# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head
from detectron2.layers import cat

from .mask import MaskBranch
from .unlabel import UnlabelBranch
from .dot import DotBranch
from torch.nn import functional as F
from adet.utils.comm import reduce_sum, reduce_mean, compute_ious
from ..layers.heatmap_focal_loss import binary_heatmap_focal_loss_jit
from .omni_loss import BOXLoss, DOTLoss, MASKLoss, UNLABELLoss, get_omni_weight
from adet.layers import ml_nms, IOULoss

__all__ = ["Omni"]


@META_ARCH_REGISTRY.register()
class Omni(nn.Module):
    """
    Main class for BlendMask architectures (see https://arxiv.org/abd/1901.02446).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.instance_loss_weight = cfg.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT

        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_branch = MaskBranch(cfg, self.backbone.output_shape())
        self.dot_branch = DotBranch(cfg, self.backbone.output_shape())
        self.unlabel_branch = UnlabelBranch(cfg, self.backbone.output_shape())
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES

        self.loc_loss_func = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)

        self.BOX_loss = BOXLoss(cfg)
        self.DOT_loss = DOTLoss(cfg)
        self.MASK_loss = MASKLoss(cfg)
        self.UNLABEL_loss = UNLABELLoss(cfg)

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
        if self.combine_on:
            self.panoptic_module = build_sem_seg_head(cfg, self.backbone.output_shape())
            self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
            self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
            self.combine_instances_confidence_threshold = (cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH)

        # build top module
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES
        attn_size = cfg.MODEL.BLENDMASK.ATTN_SIZE
        attn_len = num_bases * attn_size * attn_size
        self.top_layer = nn.Conv2d(
            in_channels, attn_len,
            kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.top_layer.weight, std=0.01)
        torch.nn.init.constant_(self.top_layer.bias, 0)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES

    def forward(self, batched_inputs, iter=0, branch='supervised'):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            instances: Instances
            sem_seg: semantic segmentation ground truth.
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: each dict is the results for one image. The dict
                contains the following keys:
                "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                    See the return value of
                    :func:`combine_semantic_and_instance_outputs` for its format.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        if self.training:
            return self.forward_train_merge(images, features, batched_inputs, iter, branch)
        else:
            return self.forward_test(images, features, batched_inputs, branch=branch)

    def normalize(self, x):
        return (x - x.min() + 1e-4) / (x.max() - x.min() + 1e-4)

    def forward_train_merge(self, images, features, batched_inputs, iter=0, branch='supervised_box'):
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if "basis_sem" in batched_inputs[0]:
            basis_sem = [x["basis_sem"].to(self.device) for x in batched_inputs]
            basis_sem = ImageList.from_tensors(
                basis_sem, self.backbone.size_divisibility, 0).tensor
        else:
            basis_sem = None

        center_losses = {}
        mask_losses = {}
        unlabel_losses = {}

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, self.top_layer)

        if self.cfg.MODEL.OMNI.DOT_ON:
            center_losses = self.dot_branch(images, features, gt_instances)
            dot_train_info = center_losses['train_info']
            del center_losses['train_info']
            logits_dots = dot_train_info['logits_pred']
        else:
            logits_dots = None

        if self.cfg.MODEL.OMNI.UNLABEL_ON:
            logits_unlabel, _, _ = self.unlabel_branch(features)
        else:
            logits_unlabel = None

        if self.cfg.MODEL.OMNI.MASK_ON:
            basis_out, basis_losses = self.mask_branch(images, features, gt_instances, basis_sem, branch)
            mask_train_info = basis_losses['train_info']
            del basis_losses['train_info']
            logits_masks = mask_train_info['logits_pred']
        else:
            logits_masks = None

        if self.cfg.MODEL.OMNI.BOX_WEIGHT_ON:
            box_train_info = proposal_losses['train_info']
            del proposal_losses['train_info']
            logits_boxes = box_train_info['logits_pred']
        else:
            logits_boxes = None

        omni_weight, weight_dot, weight_box, weight_mask, weight_unlabel = get_omni_weight(
            logits_boxes, logits_dots, logits_masks, logits_unlabel, gt_instances)

        weight_box = weight_box.detach()
        if self.cfg.MODEL.OMNI.DOT_ON:
            weight_dot = weight_dot.detach()
        if self.cfg.MODEL.OMNI.UNLABEL_ON:
            weight_unlabel = weight_unlabel.detach()
        if self.cfg.MODEL.OMNI.MASK_ON:
            weight_mask = weight_mask.detach()

        losses = {}

        if self.cfg.MODEL.OMNI.BOX_WEIGHT_ON and 'box' in branch:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if gt_instances[0].gt_boxes.tensor.shape[0] > 0:
                if self.cfg.MODEL.OMNI.DOT_ON and self.cfg.MODEL.OMNI.UNLABEL_ON and self.cfg.MODEL.OMNI.MASK_ON:
                    weight_for_box = torch.pow(weight_dot * weight_unlabel * weight_mask, 1 / 3)
                elif self.cfg.MODEL.OMNI.DOT_ON and self.cfg.MODEL.OMNI.MASK_ON:
                    weight_for_box = torch.pow(weight_dot * weight_mask, 1 / 2)
                elif self.cfg.MODEL.OMNI.DOT_ON and self.cfg.MODEL.OMNI.UNLABEL_ON:
                    weight_for_box = torch.pow(weight_dot * weight_unlabel, 1 / 2)
                elif self.cfg.MODEL.OMNI.MASK_ON:
                    weight_for_box = weight_mask
                elif self.cfg.MODEL.OMNI.DOT_ON:
                    weight_for_box = weight_dot
                elif self.cfg.MODEL.OMNI.UNLABEL_ON:
                    weight_for_box = weight_unlabel

                box_weight_loss = self.BOX_loss(logits_boxes, weight_for_box, box_train_info, gt_instances)
                proposal_losses.update(box_weight_loss)
                losses.update(proposal_losses)
            else:
                box_pred = logits_boxes.sigmoid()
                losses['loss_box_cls'] = torch.sum(
                    -torch.log(1 - box_pred) * torch.pow(box_pred, self.focal_loss_gamma)) * (1 - self.focal_loss_alpha)

        if self.cfg.MODEL.OMNI.MASK_ON and 'mask' in branch:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if gt_instances[0].gt_boxes.tensor.shape[0] > 0:
                if self.cfg.MODEL.OMNI.DOT_ON and self.cfg.MODEL.OMNI.UNLABEL_ON and self.cfg.MODEL.OMNI.BOX_ON:
                    weight_for_mask = torch.pow(weight_dot * weight_unlabel * weight_box, 1 / 3)
                elif self.cfg.MODEL.OMNI.BOX_ON and self.cfg.MODEL.OMNI.DOT_ON:
                    weight_for_mask = torch.pow(weight_dot * weight_box, 1 / 2)
                elif self.cfg.MODEL.OMNI.BOX_ON:
                    weight_for_mask = weight_box
                mask_weight_loss = self.MASK_loss(logits_masks, weight_for_mask, box_train_info,
                                                  mask_train_info, gt_instances)
                mask_losses.update(mask_weight_loss)
                losses.update(mask_losses)
            else:
                mask_pred = logits_masks.sigmoid()
                losses['loss_mask_cls'] = torch.sum(
                    -torch.log(1 - mask_pred) * torch.pow(mask_pred, self.focal_loss_gamma)) * (
                                                  1 - self.focal_loss_alpha)

        if self.cfg.MODEL.OMNI.DOT_ON and 'dot' in branch:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if gt_instances[0].gt_boxes.tensor.shape[0] > 0:
                if self.cfg.MODEL.OMNI.BOX_ON and self.cfg.MODEL.OMNI.UNLABEL_ON and self.cfg.MODEL.OMNI.MASK_ON:
                    weight_for_dot = torch.pow(weight_box * weight_unlabel * weight_mask, 1 / 3)
                elif self.cfg.MODEL.OMNI.BOX_ON and self.cfg.MODEL.OMNI.MASK_ON:
                    weight_for_dot = torch.pow(weight_box * weight_mask, 1 / 2)
                elif self.cfg.MODEL.OMNI.BOX_ON and self.cfg.MODEL.OMNI.UNLABEL_ON:
                    weight_for_dot = torch.pow(weight_box * weight_unlabel, 1 / 2)
                elif self.cfg.MODEL.OMNI.BOX_ON:
                    weight_for_dot = weight_box

                dot_weight_loss = self.DOT_loss(logits_dots, weight_for_dot, dot_train_info)
                center_losses.update(dot_weight_loss)
                losses.update(center_losses)
            else:
                dot_pred = logits_dots.sigmoid()
                losses['loss_dot_cls'] = torch.sum(
                    -torch.log(1 - dot_pred) * torch.pow(dot_pred, self.focal_loss_gamma)) * (1 - self.focal_loss_alpha)

        if self.cfg.MODEL.OMNI.UNLABEL_ON and 'unsup' in branch:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if gt_instances[0].gt_boxes.tensor.shape[0] > 0:
                if self.cfg.MODEL.OMNI.BOX_ON and self.cfg.MODEL.OMNI.DOT_ON and self.cfg.MODEL.OMNI.MASK_ON:
                    weight_for_unlabel = torch.pow(weight_box * weight_dot * weight_mask, 1 / 3)
                elif self.cfg.MODEL.OMNI.BOX_ON and self.cfg.MODEL.OMNI.DOT_ON:
                    weight_for_unlabel = torch.pow(weight_box * weight_dot, 1 / 2)
                elif self.cfg.MODEL.OMNI.BOX_ON:
                    weight_for_unlabel = weight_box

                unlabel_weight_loss = self.UNLABEL_loss(logits_unlabel, weight_for_unlabel)
                unlabel_losses.update(unlabel_weight_loss)
                losses.update(unlabel_losses)
            else:
                unlabel_pred = logits_unlabel.sigmoid()
                losses['loss_unlabel_cls'] = torch.sum(
                    -torch.log(1 - unlabel_pred) * torch.pow(unlabel_pred, self.focal_loss_gamma)) * (1 - self.focal_loss_alpha)

        if 'mask' in branch:
            losses['loss_fcos_loc'] = proposal_losses['loss_fcos_loc']

        return losses

    def forward_test(self, images, features, batched_inputs, branch):
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        logits_mask, logits_dot, logits_unlabel = None, None, None
        if self.cfg.MODEL.OMNI.MASK_ON:
            logits_mask, basis_losses = self.mask_branch(images, features, gt_instances, None, branch)
        if self.cfg.MODEL.OMNI.DOT_ON:
            logits_dot = self.dot_branch(images, features, gt_instances)
        if self.cfg.MODEL.OMNI.UNLABEL_ON:
            logits_unlabel, _ = self.unlabel_branch(features)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, self.top_layer,
                                                             logits_mask=logits_mask,
                                                             logits_dot=logits_dot, logits_unlabel=logits_unlabel)

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
