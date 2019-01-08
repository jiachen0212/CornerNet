import numpy as np
import torch
import torch.nn as nn

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _decode
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss
from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer


# kp_module class 主要用来定义hourglass中会用到的一些层.. downsample,maxpooling,upsample之类的...
class kp_module(nn.Module):
    '''  
        n = 5                                      # resolution reduced 5 times
        dims    = [256, 256, 384, 384, 384, 512]   # channels   
        modules = [2, 2, 2, 2, 2, 4]               # residul数目
        out_dim = 80
    '''
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n
        curr_mod = modules[0]
        next_mod = modules[1]
        curr_dim = dims[0]
        next_dim = dims[1]
        self.up1  = make_up_layer(  # kernel_size=3
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs)    # layer=residual
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs)
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs)   # 在n>1之前一直是downsample的... 不然的话就是后面的residul+upsample

        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs)
        self.up2  = make_unpool_layer(curr_dim)   # upsample
        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)




# kp class 正式搭建 CornerNet 的主框架..
class kp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(kp, self).__init__()

        self.nstack    = nstack
        self._decode   = _decode   # decode outs, 得到：bboxes, scores, tl_scores, br_scores, clses

        curr_dim = dims[0]

        # Downsample 4 times before the hourglass part
        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(    # 调用 kp_module 开始搭建hourglass了...
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)])   # nstack = 2, 2 hourglass


        # cnv_dim=256
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags   embedding the distance, dim == 1
        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)


######## 这部分是干嘛, 感觉上是两个hourglass之间的衔接部分...
        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)   # residual
        ])
        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])


        # regress
        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]

        inter = self.pre(image)  # hourglass前的downsample, 4 times
        outs  = []

        layers = zip(
            self.kps, self.cnvs,   # ? pre features?
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_   = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

            tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            # 第一个hourglass, 中间衔接几层convrelubn, 然后进入下一个hourglass
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)   # 2line181
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_   = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp  = kp_(inter)
            cnv = cnv_(kp)


            # 第二个hourglass, 取出它的result
            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            # 第一个hourglass没有加入使用去检测corner point
            # 第一个hourglass, 再走一遍上面的hourglass流程..
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-6:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight  # 0.1
        self.push_weight = push_weight  # 0.1
        self.regr_weight = regr_weight  # 1

        # value from kp_utils.py
        self.focal_loss = focal_loss
        self.ae_loss = _ae_loss
        self.regr_loss = _regr_loss

    def forward(self, outs, targets):
        # 但是这个target的来源我还没整明白...
        # 为什么stride==6, 因为tlhp, brhp, tl-tag, br-tag, tl-reg, br-reg 6个量一起排列存的
        stride = 6

        # head map
        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]

        # embedding
        tl_tags  = outs[2::stride]
        br_tags  = outs[3::stride]

        # regress offset 
        tl_regrs = outs[4::stride]
        br_regrs = outs[5::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]

        gt_mask = targets[2]

        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]   # value to 0~1
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_heats)
        return loss.unsqueeze(0)
