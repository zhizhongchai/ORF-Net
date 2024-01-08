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

# from .blender import build_blender
from .basis_module import build_basis_module
# from .detectmask import DetectMask
from .tag import Tag
# from .dot import Dot
from .centernet import CenterNet
from torch.nn import functional as F
from adet.utils.comm import reduce_sum, reduce_mean, compute_ious
from ..layers.heatmap_focal_loss import binary_heatmap_focal_loss_jit
from .omni_loss import BOXLoss, DOTLoss, MASKLoss, TAGLoss, get_omni_weight, OMNI_BOX_Loss, TAG_DATA_Loss, \
    BOX_DATA_Loss, DOT_DATA_Loss
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

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.centernet = CenterNet(cfg, self.backbone.output_shape())
        self.tag = Tag(cfg, self.backbone.output_shape())
        self.basis_module = build_basis_module(cfg, self.backbone.output_shape())
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES

        self.loc_loss_func = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)

        self.BOX_loss = BOXLoss(cfg)
        self.DOT_loss = DOTLoss(cfg)
        self.MASK_loss = MASKLoss(cfg)
        self.TAG_loss = TAGLoss(cfg)
        self.BOX_DATA_loss = BOX_DATA_Loss(cfg)
        self.DOT_DATA_loss = DOT_DATA_Loss(cfg)
        self.TAG_DATA_loss = TAG_DATA_Loss(cfg)
        self.OMNI_BOX_loss = OMNI_BOX_Loss(cfg)

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
        basis_losses = {}

        proposals, proposal_losses, box_results, box_uncertainty = self.proposal_generator(images, features,
                                                                                           gt_instances,
                                                                                           self.top_layer, branch)

        if self.cfg.MODEL.OMNI.DOT_ON:
            center_proposals, center_losses, dot_uncertainty = self.centernet(images, features, gt_instances,
                                                                              branch=branch)

        if self.cfg.MODEL.OMNI.TAG_ON:
            _, tag_losses, tag_uncertainty = self.tag(features, gt_instances, branch, box_results)

        if self.cfg.MODEL.OMNI.MASK_ON:
            # basis_out, basis_losses = self.detectmask(images, features, gt_instances, basis_sem, branch)
            basis_out, basis_losses = self.basis_module(features, basis_sem, branch)

        if self.cfg.MODEL.OMNI.BOX_WEIGHT_ON:
            box_train_info = proposal_losses['train_info']
            del proposal_losses['train_info']
            logits_boxes = box_train_info['logits_pred']
            is_in_pred_boxes = box_train_info['is_in_pred_boxes']
        else:
            logits_boxes = None
            is_in_pred_boxes = None

        if self.cfg.MODEL.OMNI.DOT_ON:
            dot_train_info = center_losses['train_info']
            del center_losses['train_info']
            logits_dots = dot_train_info['logits_pred']
        else:
            logits_dots = None

        if self.cfg.MODEL.OMNI.TAG_ON:
            tag_train_info = tag_losses['train_info']
            del tag_losses['train_info']
            logits_tags = tag_train_info['logits_pred']
        else:
            logits_tags = None

        if self.cfg.MODEL.OMNI.MASK_ON:
            mask_train_info = basis_losses['train_info']
            del basis_losses['train_info']
            logits_masks = mask_train_info['logits_pred']
        else:
            logits_masks = None

        omni_weight, weight_dot, weight_box, weight_mask, weight_tag, weight_pos, weight_neg, branch_num = get_omni_weight(
            logits_boxes, logits_dots, logits_masks, logits_tags, gt_instances)

        if not self.cfg.MODEL.OMNI.TAG_ON:
            pred_tag = None
        else:
            pred_tag = weight_tag.clone()
        pred_box = weight_box.clone()
        if not self.cfg.MODEL.OMNI.DOT_ON:
            pred_dot = None
        else:
            pred_dot = weight_dot.clone()

        weight_box = weight_box.detach()
        if self.cfg.MODEL.OMNI.DOT_ON:
            weight_dot = weight_dot.detach()
        if self.cfg.MODEL.OMNI.TAG_ON:
            weight_tag = weight_tag.detach()

        if self.cfg.MODEL.OMNI.BOX_WEIGHT_ON and ('box' in branch or 'neg' in branch):
            if self.cfg.MODEL.OMNI.DOT_ON and self.cfg.MODEL.OMNI.TAG_ON:
                weight_for_box = torch.pow(weight_dot * weight_tag, 1 / 2)
            elif self.cfg.MODEL.OMNI.DOT_ON:
                weight_for_box = weight_dot
            elif self.cfg.MODEL.OMNI.TAG_ON:
                weight_for_box = weight_tag

            # if iter < 1000:
            #     weight_for_box = weight_box

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            box_weight_loss = self.BOX_loss(logits_boxes, weight_for_box, box_train_info, gt_instances, pred_dot,
                                            pred_tag)
            proposal_losses.update(box_weight_loss)

        if self.cfg.MODEL.OMNI.DOT_ON and 'dot' in branch:
            weight_for_dot = torch.pow(weight_tag * weight_box, 1 / 2) if self.cfg.MODEL.OMNI.TAG_ON else weight_box
            # if iter < 1000:
            #     weight_for_dot = weight_box
            dot_weight_loss = self.DOT_loss(logits_dots, weight_for_dot, dot_train_info, pred_box, pred_tag)
            center_losses.update(dot_weight_loss)
        if self.cfg.MODEL.OMNI.TAG_ON and 'unsup' in branch:
            weight_for_tag = torch.pow(weight_dot * weight_box, 1 / 2) if self.cfg.MODEL.OMNI.DOT_ON else weight_box
            # if iter < 1000:
            #     weight_for_tag = weight_box
            tag_weight_loss = self.TAG_loss(logits_tags, weight_for_tag, tag_train_info, box_results, branch, iter)
            tag_losses.update(tag_weight_loss)

        losses = {}
        if 'box' in branch or 'neg' in branch:
            losses.update(proposal_losses)
        if 'dot' in branch:
            losses.update(center_losses)
        if 'unsup' in branch:
            losses.update(tag_losses)

        return losses

    def forward_test(self, images, features, batched_inputs, branch):
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, self.top_layer,
                                                             branch=branch)

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    # def forward_test(self, images, features, batched_inputs):
    #     if "instances" in batched_inputs[0]:
    #         gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
    #     else:
    #         gt_instances = None
    #     # cls_feats, reg_feats, dot_feats, mask_feats = self.share_block(features)
    #     proposals, proposal_losses,logits_box = self.proposal_generator(images, features, gt_instances, self.top_layer)
    #
    #     tag_proposals, tag_losses,logits_tag = self.tag(images, features, gt_instances)
    #
    #     dot_proposals, dot_losses,logits_dot = self.dot(images, features, gt_instances)
    #
    #     processed_results = []
    #     for results_per_image, input_per_image, image_size in zip(
    #             proposals, batched_inputs, images.image_sizes
    #     ):
    #         height = input_per_image.get("height", image_size[0])
    #         width = input_per_image.get("width", image_size[1])
    #         r = detector_postprocess(results_per_image, height, width)
    #         processed_results.append({"instances": r})
    #     return logits_box,logits_dot,logits_tag,logits_box
