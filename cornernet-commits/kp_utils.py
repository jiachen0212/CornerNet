import torch
import torch.nn as nn

from .utils import convolution, residual

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer(dim):
    return MergeUp()

def make_tl_layer(dim):
    return None

def make_br_layer(dim):
    return None

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=20):
    # k 调用时被置为100
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)   # 拉成 batch 行, 每一行取前100. 即每张 heatmap 取前100个point

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    # 返回 feature map 上的定点的整数坐标点.. topk_ys, topk_xs
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
            # 保留的k=100个points的分数, 所属的类别, index, 和他们的整点坐标值.


# decode the outs
def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, 
    K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = tl_heat.size()     # bs, c, H, W

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    # 做一个3x3的maxpooling, string=1
    # 在heatmap上利用3x3的maxpooling来完成相当于NMS的操作
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    # 每张heatmap取前100个points (因为sigmoid后满足的点可能还蛮多的, 超过20个是很可能的...  参数k=100)
    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    # tl_xs, tl_ys ... 这些是 feature map 上的整数定点 point 坐标
    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    # tl_regr, br_regr 是 outs 中学习的 reg offset 量
    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        # add offset 修正后的坐标, 这个值是可以反映射回原 input map 的..
        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    # 就是说不管总共含了多少类目标, 反正我一张 heat map 就是出 100 个点.. (128*128的 resolution)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists  = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)   # 首先 corner point 类别不一致的肯定不属于同一个框..

    # reject boxes based on distances
    # ae_threshold = 0.5, embeeding 后所属同一目标的tl和br的'distance'<=0.5
    dist_inds = (dists > ae_threshold)  # 然后 tl, br point emdedding 后的 distance>0.5 的话也不会是属于同一box

    # reject boxes based on widths and heights
    # 右点>左点, 下点>上点..
    width_inds  = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    # 每张 heatmap 100 个 cornet point, bs=29  够1000 的..
    scores, inds = torch.topk(scores, num_dets)   # batch 维度加进来后, 取前1000个det 结果..
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = tl_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections

def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    # paper 设置的超参, beta 设置的4
    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        # paper 设置的超参, alpha 设置的2
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

def _ae_loss(tag0, tag1, mask):
    # mask:gt为1 非gt为0吧?..
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    # 均值..
    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _regr_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss
