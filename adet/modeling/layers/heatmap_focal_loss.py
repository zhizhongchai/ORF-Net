import torch
from torch.nn import functional as F


# TODO: merge these two function
def heatmap_focal_loss(
        inputs,
        targets,
        pos_inds,
        labels,
        alpha: float = -1,
        beta: float = 4,
        gamma: float = 2,
        reduction: str = 'sum',
        sigmoid_clamp: float = 1e-4,
        ignore_high_fp: float = -1.,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs:  (sum_l N*Hl*Wl, C)
        targets: (sum_l N*Hl*Wl, C)
        pos_inds: N
        labels: N
    Returns:
        Loss tensor with the reduction option applied.
    """
    pred = torch.clamp(inputs.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
    neg_weights = torch.pow(1 - targets, beta)
    pos_pred_pix = pred[pos_inds]  # N x C
    pos_pred = pos_pred_pix.gather(1, labels.unsqueeze(1))
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights

    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    if reduction == "sum":
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss

    return - pos_loss, - neg_loss


def heatmap_focal_loss(
        inputs,
        targets,
        pos_inds,
        labels,
        alpha: float = -1,
        beta: float = 4,
        gamma: float = 2,
        reduction: str = 'sum',
        sigmoid_clamp: float = 1e-4,
        ignore_high_fp: float = -1.,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs:  (sum_l N*Hl*Wl, C)
        targets: (sum_l N*Hl*Wl, C)
        pos_inds: N
        labels: N
    Returns:
        Loss tensor with the reduction option applied.
    """
    pred = torch.clamp(inputs.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
    neg_weights = torch.pow(1 - targets, beta)
    pos_pred_pix = pred[pos_inds]  # N x C
    pos_pred = pos_pred_pix.gather(1, labels.unsqueeze(1))
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights

    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    if reduction == "sum":
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss

    return - pos_loss, - neg_loss


heatmap_focal_loss_jit = torch.jit.script(heatmap_focal_loss)


# heatmap_focal_loss_jit = heatmap_focal_loss

def binary_heatmap_focal_loss(
        inputs,
        targets,
        pos_inds,
        alpha: float = -1,
        beta: float = 4,
        gamma: float = 2,
        sigmoid_clamp: float = 1e-4,
        ignore_high_fp: float = -1.,
):
    """
    Args:
        inputs:  (sum_l N*Hl*Wl,)
        targets: (sum_l N*Hl*Wl,)
        pos_inds: N
    Returns:
        Loss tensor with the reduction option applied.
    """
    pred = torch.clamp(inputs.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
    neg_weights = torch.pow(1 - targets, beta)
    pos_pred = pred[pos_inds]  # N
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights
    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    pos_loss = - pos_loss.sum()
    neg_loss = - neg_loss.sum()

    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss

    return pos_loss, neg_loss


binary_heatmap_focal_loss_jit = torch.jit.script(binary_heatmap_focal_loss)


def weight_binary_heatmap_focal_loss(
        inputs,
        targets,
        pos_inds,
        weights,
        alpha: float = -1,
        beta: float = 4,
        gamma: float = 2,
        sigmoid_clamp: float = 1e-4,
        ignore_high_fp: float = -1.,
        reduction: str = 'sum',
):
    """
    Args:
        inputs:  (sum_l N*Hl*Wl,)
        targets: (sum_l N*Hl*Wl,)
        pos_inds: N
    Returns:
        Loss tensor with the reduction option applied.
    """

    pred = torch.clamp(inputs.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
    neg_weights = torch.pow(1 - targets, beta)
    pos_pred = pred[pos_inds]  # N
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma) * weights[pos_inds]
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * weights
    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    if reduction == "sum":
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        # if pos_pred.shape[0] != 0:
        #     pos_loss = pos_loss.sum() / torch.sum(weights[pos_inds])
        # else:
        #     pos_loss = pos_loss.sum() * 0
        # neg_loss = neg_loss.sum() / torch.sum(weights)

    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss

    return -pos_loss, -neg_loss


weight_binary_heatmap_focal_loss_jit = torch.jit.script(weight_binary_heatmap_focal_loss)


def binary_heatmap_focal_loss_omni(
        inputs,
        targets,
        scores,
        pos_inds,
        alpha: float = -1,
        beta: float = 4,
        gamma: float = 2,
        sigmoid_clamp: float = 1e-4,
        ignore_high_fp: float = -1.,
):
    """
    Args:
        inputs:  (sum_l N*Hl*Wl,)
        targets: (sum_l N*Hl*Wl,)
        pos_inds: N
    Returns:
        Loss tensor with the reduction option applied.
    """
    pred = torch.clamp(inputs.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)

    # targets[targets > 1] = 1
    neg_weights = torch.pow(1 - targets, beta)
    # neg_weights = torch.exp(1 - targets)
    # neg_weights[pos_inds]=neg_weights[pos_inds]-1
    # pos_weights = scores[pos_inds]
    pos_weights = torch.exp(scores)[pos_inds]

    pos_pred = pred[pos_inds]  # N
    # pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma)
    # neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights

    pos_loss = torch.log(pos_pred)
    # pos_loss = torch.log(pos_pred*scores[pos_inds])

    temp_targets = torch.zeros_like(scores)
    temp_targets[pos_inds]=1
    neg_inds = temp_targets==0

    neg_loss = torch.log(1 - pred[neg_inds]*neg_weights[neg_inds]) * torch.pow(pred[neg_inds], gamma)
    # print(225, neg_weights[neg_inds].requires_grad)
    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    pos_loss = - pos_loss.sum()
    neg_loss = - neg_loss.sum()

    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss

    return pos_loss, neg_loss


binary_heatmap_focal_loss_jit_omni = torch.jit.script(binary_heatmap_focal_loss_omni)
