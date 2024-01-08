import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from .tood_outputs import TOODOutputs

__all__ = ["TOOD"]

INF = 100000000
EPS = 1e-12


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8, conv_cfg=None, norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.la_conv1 = nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1)
        self.norm = nn.GroupNorm(32, self.feat_channels)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reduction_conv = nn.Conv2d(self.in_channels,
                                        self.feat_channels,
                                        1,
                                        stride=1,
                                        padding=0)

        # self.reduction_conv = nn.ModuleList()
        # tower = []
        # tower.append(nn.Conv2d(self.in_channels,
        #                        self.feat_channels,
        #                        1,
        #                        stride=1,
        #                        padding=0))
        # tower.append(nn.GroupNorm(32, self.in_channels))
        # tower.append(nn.ReLU())
        # self.reduction_conv.append(nn.Sequential(*tower))

        for modules in [
            self.la_conv1, self.la_conv2,
            self.reduction_conv,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, feat_list, avg_feat=None):
        feat = torch.cat(feat_list, 1)
        # b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        task_convs = []
        weight = weight[:, :, 0, 0]
        for i in range(self.stacked_convs):
            task_convs.append(feat_list[i] * weight[:, i])

        task_feat = torch.cat(task_convs, 1)
        task_feat = self.reduction_conv(task_feat)
        task_feat = self.norm(task_feat)
        task_feat = self.relu(task_feat)

        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs.
        # conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
        #               self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs,
        #                                                       self.feat_channels)
        # conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        # feat = feat.reshape(b, self.in_channels, h * w)
        # feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        # if self.norm_cfg is not None:
        #     feat = self.reduction_conv.norm(feat)
        # feat = self.reduction_conv.activate(feat)

        return task_feat


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


@PROPOSAL_GENERATOR_REGISTRY.register()
class TOOD(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS.YIELD_PROPOSAL
        self.yield_box_feats = cfg.MODEL.FCOS.YIELD_BOX_FEATURES

        self.tood_head = TOODHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.tood_head.in_channels_to_top_module

        self.tood_outputs = TOODOutputs(cfg)
        self.cfg = cfg

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness, top_feats, center_attns, bbox_towers, weight_logits = self.tood_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

    def forward(self, images, cls_features, reg_features, gt_instances=None, top_module=None, logits_dots=None, logits_masks=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        cls_features = [cls_features[f] for f in self.in_features]
        reg_features = [reg_features[f] for f in self.in_features]
        locations = self.compute_locations(reg_features)
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.tood_head(
            cls_features, reg_features, top_module, self.yield_proposal or self.yield_box_feats
        )

        if self.training:
            results, losses = self.tood_outputs.losses(
                logits_pred, reg_pred, ctrness_pred,
                locations, gt_instances, top_feats, logits_dots, logits_masks
            )

            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.tood_outputs.predict_proposals(
                        logits_pred, reg_pred, ctrness_pred,
                        locations, images.image_sizes, top_feats
                    )
            if self.yield_box_feats:
                results["box_feats"] = {
                    f: b for f, b in zip(self.in_features, bbox_towers)
                }
            return results, losses
        else:
            results = self.tood_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, images.image_sizes, top_feats
            )
            extras = {}
            if self.yield_box_feats:
                extras["box_feats"] = {
                    f: b for f, b in zip(self.in_features, bbox_towers)
                }
            return results, extras

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations


class TOODHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.cfg = cfg
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.in_channels_to_top_module = in_channels

        self.stacked_convs = 6
        self.feat_channels = in_channels
        self.inter_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            tower = []
            tower.append(nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=1,
                padding=1, bias=True
            ))
            tower.append(nn.GroupNorm(32, in_channels))
            tower.append(nn.ReLU())
            self.inter_convs.append(nn.Sequential(*tower))

        # self.cls_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8)
        # self.reg_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8)
        # self.dot_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8)
        # self.mask_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8)

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )

        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        for modules in [self.cls_logits, self.bbox_pred, self.ctrness, ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, cls_x, reg_x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(cls_x):
            logits.append(self.cls_logits(feature))

        for l, feature in enumerate(reg_x):
            bbox_tower = feature
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)
            reg = self.bbox_pred(bbox_tower)
            ctrness.append(self.ctrness(bbox_tower))
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))

        return logits, bbox_reg, ctrness, top_feats, bbox_towers
