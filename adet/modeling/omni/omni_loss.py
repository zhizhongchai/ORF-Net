import logging
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit
from ..layers.heatmap_focal_loss import heatmap_focal_loss_jit
from ..layers.heatmap_focal_loss import binary_heatmap_focal_loss_jit, binary_heatmap_focal_loss_jit_omni
import numpy as np
from adet.utils.comm import reduce_sum, reduce_mean, compute_ious

sigmoid_clamp = 1e-4
gamma = 2


def normalize(x):
    return (x - 0. + 1e-4) / (x.max() - 0. + 1e-4)


def get_omni_weight(logits_box, logits_dot=None, logits_mask=None, logits_unlabel=None, targets=None):
    weight_list = []
    weight_list_dot = []
    weight_list_box = []
    weight_list_mask = []
    weight_list_unlabel = []

    for im_i in range(len(targets)):
        score_list = []
        if logits_dot is not None:
            dot_scores = logits_dot.sigmoid()
            score_list.append(dot_scores)
            weight_list_dot.append(dot_scores)
        if logits_box is not None:
            box_scores = logits_box.sigmoid()
            score_list.append(box_scores)
            weight_list_box.append(box_scores)

        if logits_unlabel is not None:
            unlabel_scores = logits_unlabel.sigmoid()
            score_list.append(unlabel_scores)
            weight_list_unlabel.append(unlabel_scores)

        if logits_mask is not None:
            mask_scores = logits_mask.sigmoid()
            score_list.append(mask_scores)
            weight_list_mask.append(mask_scores)

        if len(score_list) == 1:
            weight_list.append(score_list[0])
        else:
            sum_weight = torch.ones_like(score_list[0])
            for i, s in enumerate(score_list):
                sum_weight = sum_weight * s
            temp_weight = torch.pow(sum_weight, 1 / len(score_list))
            weight_list.append(temp_weight)

    weight = cat([x.reshape(-1, x.shape[-1]) for x in weight_list], dim=0, ) if len(
        weight_list) > 0 else None
    weight_dot = cat([x.reshape(-1, x.shape[-1]) for x in weight_list_dot], dim=0, ) if len(
        weight_list_dot) > 0 else None
    weight_box = cat([x.reshape(-1, x.shape[-1]) for x in weight_list_box], dim=0, ) if len(
        weight_list_box) > 0 else None
    weight_mask = cat([x.reshape(-1, x.shape[-1]) for x in weight_list_mask], dim=0, ) if len(
        weight_list_mask) > 0 else None
    weight_unlabel = cat([x.reshape(-1, x.shape[-1]) for x in weight_list_unlabel], dim=0, ) if len(
        weight_list_unlabel) > 0 else None
    return weight, weight_dot, weight_box, weight_mask, weight_unlabel


class BOXLoss(nn.Module):
    def __init__(self, cfg):
        super(BOXLoss, self).__init__()
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        self.cfg = cfg

    def forward(self, logits_pred, scores, train_info, targets=None):
        losses = {}
        is_in_boxes_list = train_info['is_in_boxes']
        num_pos_avg = train_info['num_pos_avg']
        IoUMap = train_info['IoUMap']

        pos_weights = []
        # box_neg_weights = []
        neg_weights = []

        box_pred = torch.clamp(logits_pred.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            labels_per_im = targets_per_im.gt_classes
            bboxes = targets_per_im.gt_boxes.tensor
            num_gt = labels_per_im.size(0)
            is_in_boxes = is_in_boxes_list[im_i]

            if bboxes.numel() == 0:
                pos_weights.append(scores.new_zeros((scores.size(0), num_gt)))
                # box_neg_weights.append(scores.new_zeros((scores.size(0), num_gt)))
                neg_weights.append(torch.ones_like(scores))
                continue
            alignment_metrics = scores[:, labels_per_im]
            alignment_metrics = alignment_metrics * is_in_boxes

            iou_alignment_metrics = IoUMap.detach()[:, labels_per_im]
            iou_alignment_metrics = iou_alignment_metrics * is_in_boxes

            temp_alignment_metrics_list = []

            neg_alignment_metric = torch.zeros_like(scores)
            for i in range(alignment_metrics.shape[-1]):

                temp_iou_alignment_metrics = iou_alignment_metrics[:, i]

                temp_alignment_metrics = alignment_metrics[:, i]
                temp_is_in_boxes = is_in_boxes[:, i] * 1
                temp_metrics = torch.zeros_like(temp_alignment_metrics)

                if torch.sum(temp_is_in_boxes * 1) != 0:
                    confidence = torch.max(temp_alignment_metrics[temp_is_in_boxes > 0])
                    temp_metrics[temp_is_in_boxes > 0] = normalize(
                        torch.pow((temp_alignment_metrics[temp_is_in_boxes > 0]), confidence) *
                        temp_iou_alignment_metrics[temp_is_in_boxes > 0])

                temp_alignment_metrics_list.append(temp_metrics)
                neg_alignment_metric[:, 0] = neg_alignment_metric[:, 0] + temp_is_in_boxes

            pos_weights.append(torch.stack(temp_alignment_metrics_list, dim=-1))

            temp_neg_metrics = torch.zeros_like(scores)
            temp_neg_metrics[neg_alignment_metric == 0] = torch.ones_like(scores[neg_alignment_metric == 0])
            neg_weights.append(temp_neg_metrics)

        neg_weights = cat([x.reshape(-1, x.shape[-1]) for x in neg_weights], dim=0)

        neg_loss = torch.sum(-torch.log(1 - box_pred[neg_weights > 0]) * torch.pow(box_pred[neg_weights > 0],
                                                                                   self.focal_loss_gamma)) * (
                           1 - self.focal_loss_alpha)  # certain negaitve loss
        if pos_weights[0].shape[-1] > 0:
            pos_weights = cat([x.reshape(-1, x.shape[-1]) for x in pos_weights], dim=0, )
            pos_loss = 0
            box_neg_loss = 0

            for i in range(pos_weights.shape[-1]):
                temp_pos_weight = pos_weights[:, i]

                if torch.sum(temp_pos_weight) > 0:
                    temp_pos_weight[temp_pos_weight > 0] = temp_pos_weight[temp_pos_weight > 0].clamp(sigmoid_clamp,
                                                                                                      1 - sigmoid_clamp)

                    pos_inds = temp_pos_weight > 0.0

                    if torch.sum(pos_inds * 1) > 0:
                        pos_loss = pos_loss - torch.sum(torch.log(
                            box_pred[:, 0][pos_inds]) * torch.pow(
                            temp_pos_weight[pos_inds] * (1 - box_pred[:, 0][pos_inds]),
                            gamma)) * self.focal_loss_alpha

                        box_neg_loss = box_neg_loss - torch.sum(torch.log(
                            (1 - box_pred[:, 0][pos_inds])) * torch.pow(
                            (1 - temp_pos_weight[pos_inds]) * box_pred[:, 0][pos_inds],
                            gamma)) * self.focal_loss_alpha

            pos_loss = pos_loss / num_pos_avg
            neg_loss = neg_loss / num_pos_avg
            box_neg_loss = box_neg_loss / num_pos_avg
            losses['loss_box_cls'] = neg_loss + pos_loss + box_neg_loss
        else:
            neg_loss = neg_loss / num_pos_avg
            losses['loss_box_cls'] = neg_loss

        return losses


class MASKLoss(nn.Module):
    def __init__(self, cfg):
        super(MASKLoss, self).__init__()
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        self.cfg = cfg

    def forward(self, logits_pred, scores, box_train_info, mask_train_info, targets=None):
        losses = {}
        is_in_maskes_list = mask_train_info['is_in_boxes']
        num_pos_avg = box_train_info['num_pos_avg']
        IoUMap = box_train_info['IoUMap']

        pos_weights = []
        neg_weights = []
        box_pred = torch.clamp(logits_pred.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            labels_per_im = targets_per_im.gt_classes
            bboxes = targets_per_im.gt_boxes.tensor
            num_gt = labels_per_im.size(0)
            is_in_boxes = is_in_maskes_list[im_i]

            if bboxes.numel() == 0:
                pos_weights.append(scores.new_zeros((scores.size(0), num_gt)))
                neg_weights.append(torch.ones_like(scores))
                continue
            alignment_metrics = scores[:, labels_per_im]
            alignment_metrics = alignment_metrics * is_in_boxes

            iou_alignment_metrics = IoUMap.detach()[:, labels_per_im]
            iou_alignment_metrics = iou_alignment_metrics * is_in_boxes

            temp_alignment_metrics_list = []
            neg_alignment_metric = torch.zeros_like(scores)
            for i in range(alignment_metrics.shape[-1]):

                temp_iou_alignment_metrics = iou_alignment_metrics[:, i]

                temp_alignment_metrics = alignment_metrics[:, i]
                temp_is_in_boxes = is_in_boxes[:, i] * 1
                temp_metrics = torch.zeros_like(temp_alignment_metrics)

                if torch.sum(temp_is_in_boxes * 1) != 0:
                    confidence = torch.max(temp_alignment_metrics[temp_is_in_boxes > 0])
                    temp_metrics[temp_is_in_boxes > 0] = normalize(
                        torch.pow(temp_alignment_metrics[temp_is_in_boxes > 0], confidence) *
                        temp_iou_alignment_metrics[temp_is_in_boxes > 0])

                temp_alignment_metrics_list.append(temp_metrics)
                neg_alignment_metric[:, 0] = neg_alignment_metric[:, 0] + temp_is_in_boxes

            temp_alignment_metrics = torch.stack(temp_alignment_metrics_list, dim=-1)
            pos_weights.append(temp_alignment_metrics)

            ####czz
            temp_neg_metrics = torch.zeros_like(scores)
            temp_neg_metrics[neg_alignment_metric == 0] = torch.ones_like(scores[neg_alignment_metric == 0])
            neg_weights.append(temp_neg_metrics)

        neg_weights = cat([x.reshape(-1, x.shape[-1]) for x in neg_weights], dim=0, )

        neg_loss = torch.sum(-torch.log(1 - box_pred[neg_weights > 0]) * torch.pow(box_pred[neg_weights > 0],
                                                                                   self.focal_loss_gamma)) * (
                           1 - self.focal_loss_alpha)
        if pos_weights[0].shape[-1] > 0:
            pos_weights = cat([x.reshape(-1, x.shape[-1]) for x in pos_weights], dim=0, )
            pos_loss = 0
            box_neg_loss = 0

            for i in range(pos_weights.shape[-1]):
                temp_pos_weight = pos_weights[:, i]
                if torch.sum(temp_pos_weight) > 0:
                    temp_pos_weight[temp_pos_weight > 0] = temp_pos_weight[temp_pos_weight > 0].clamp(sigmoid_clamp,
                                                                                                      1 - sigmoid_clamp)

                    pos_inds = temp_pos_weight > 0.0
                    if torch.sum(pos_inds * 1) > 0:
                        pos_loss = pos_loss - torch.sum(torch.log(
                            box_pred[:, 0][pos_inds]) * torch.pow(
                            temp_pos_weight[pos_inds] * (1 - box_pred[:, 0][pos_inds]),
                            gamma)) * self.focal_loss_alpha

                        box_neg_loss = box_neg_loss - torch.sum(torch.log(
                            (1 - box_pred[:, 0][pos_inds])) * torch.pow(
                            (1 - temp_pos_weight[pos_inds]) * box_pred[:, 0][pos_inds],
                            gamma)) * self.focal_loss_alpha

            pos_loss = pos_loss / num_pos_avg
            neg_loss = neg_loss / num_pos_avg
            box_neg_loss = box_neg_loss / num_pos_avg
            losses['loss_mask_cls'] = neg_loss + pos_loss + box_neg_loss
        else:
            neg_loss = neg_loss / num_pos_avg
            losses['loss_mask_cls'] = neg_loss

        return losses

class DOTLoss(nn.Module):
    def __init__(self, cfg):
        super(DOTLoss, self).__init__()
        self.pos_weight = cfg.MODEL.CENTERNET.POS_WEIGHT
        self.neg_weight = cfg.MODEL.CENTERNET.NEG_WEIGHT
        self.hm_focal_alpha = cfg.MODEL.CENTERNET.HM_FOCAL_ALPHA
        self.hm_focal_beta = cfg.MODEL.CENTERNET.HM_FOCAL_BETA
        self.loss_gamma = cfg.MODEL.CENTERNET.LOSS_GAMMA
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.cfg = cfg

    def forward(self, logits_dots, scores, train_info):
        losses = {}
        dot_pos_inds = train_info['pos_inds']
        num_pos_avg = train_info['num_pos_avg']
        targets = torch.zeros_like(scores)
        targets[dot_pos_inds] = 1

        pred = torch.clamp(logits_dots.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
        scores = normalize(scores.detach())
        scores = torch.clamp(scores, min=sigmoid_clamp, max=1 - sigmoid_clamp)

        uncertain_inds = (targets < 1)[:, 0] * (scores >= 0.0)[:, 0]

        pos_pred = pred[dot_pos_inds]  # N
        pos_loss = -torch.sum(torch.log(pos_pred) * torch.pow(1 - pos_pred, self.loss_gamma)) * self.focal_loss_alpha

        uncertain_neg_loss = -torch.sum(
            torch.log((1 - pred[uncertain_inds])) * torch.pow(
                (1 - scores[uncertain_inds]) * pred[uncertain_inds],
                gamma)) * self.focal_loss_alpha
        uncertain_pos_loss = -torch.sum(
            torch.log(pred[uncertain_inds]) * torch.pow(
                scores[uncertain_inds] * (1 - pred[uncertain_inds]),
                gamma)) * self.focal_loss_alpha

        num_avg_uncertain = torch.sum(torch.ones_like(scores[scores > 0.5]))

        uncertain_pos_loss = uncertain_pos_loss / num_avg_uncertain
        uncertain_neg_loss = uncertain_neg_loss / num_avg_uncertain
        pos_loss = pos_loss / num_pos_avg
        uncertain_loss = uncertain_neg_loss + uncertain_pos_loss + pos_loss
        losses['loss_dot_cls'] = uncertain_loss

        return losses


class UNLABELLoss(nn.Module):
    def __init__(self, cfg):
        super(UNLABELLoss, self).__init__()
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        self.cfg = cfg
        self.hm_focal_alpha = cfg.MODEL.CENTERNET.HM_FOCAL_ALPHA

    def forward(self, logits_pred, scores):
        losses = {}

        confidence = torch.max(scores)
        scores = normalize(scores)
        scores = torch.clamp(scores, min=sigmoid_clamp, max=1 - sigmoid_clamp)

        unlabel_pred = torch.clamp(logits_pred.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)

        uncertain_inds = scores >= 0.0

        pos_loss = -torch.sum(torch.log(
            unlabel_pred[uncertain_inds]) *
                              torch.pow(scores[uncertain_inds] * (1 - unlabel_pred)[uncertain_inds],
                                        gamma)) * self.focal_loss_alpha
        neg_loss = -torch.sum(torch.log(
            (1 - unlabel_pred[uncertain_inds])) * torch.pow(
            (1 - scores[uncertain_inds]) * unlabel_pred[uncertain_inds],
            gamma) * self.focal_loss_alpha)

        num_avg_uncertain = torch.sum(torch.ones_like(scores[scores > 0.5]))

        pos_loss = pos_loss / num_avg_uncertain
        neg_loss = neg_loss / num_avg_uncertain

        unlabel_loss = pos_loss + neg_loss

        losses['loss_unlabel_cls'] = unlabel_loss * confidence

        return losses



