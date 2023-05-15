from __future__ import division
from __future__ import with_statement
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.utils import pointnet2_utils


def split_xyz_features(point_cloud):
    """
    split xyz coordinate and features
    :param point_cloud: 
    :return: 
    """
    xyz = point_cloud[..., 0:3].contiguous()
    features = point_cloud[..., 3:].transpose(1, 2).contiguous() if point_cloud.size(-1) > 3 else None

    return xyz, features


class LabModule(nn.Module):

    def __init__(self,
                 channels,
                 dropout_rate):
        super(LabModule, self).__init__()

        count = 4
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels // count, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(channels // count, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, unknow_feats):
        unknow_g = F.max_pool1d(unknow_feats, kernel_size=unknow_feats.shape[2])
        unknow_g = self.mlp(unknow_g)
        unknow_feats = unknow_g * unknow_feats

        return unknow_feats


class FPModule(nn.Module):

    def __init__(self,
                 mlp,
                 concat_channels,
                 vdchannels,
                 dropout_rate):
        super(FPModule, self).__init__()

        if vdchannels is not None:
            self.vdchannels = vdchannels
            mlp[0] += vdchannels
        else:
            self.vdchannels = None

        self.labmodule = LabModule(concat_channels, dropout_rate)

        self.mlp = nn.Sequential()
        for i in range(len(mlp) - 1):
            self.mlp.add_module("conv_" + str(i), nn.Conv2d(mlp[i], mlp[i + 1], kernel_size=1, bias=False))
            self.mlp.add_module("bn_" + str(i), nn.BatchNorm2d(mlp[i + 1]))
            self.mlp.add_module("relu_" + str(i), nn.ReLU(inplace=True))

    def forward(self,
                unknown,
                known,
                unknow_feats,
                known_feats,
                vdfeatures):
        """
        model forward
        :param unknown: (batch, n, 3)
        :param known: (batch, m, 3)
        :param unknow_feats: (batch, C1, n)
        :param known_feats: (batch, C2, m)
        :param vdfeatures: 
        :return: (batch, mlp[-1], n)
        """
        unknown_num_points = unknown.shape[1]

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            # interpolated_feats: [batch, C, num_points]
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*(known_feats.size()[0:2] + [unknown.size(1)]))

        if unknow_feats is not None:
            unknow_feats = self.labmodule(unknow_feats)
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)     # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        if self.vdchannels is not None:
            repeat_times = unknown_num_points // vdfeatures.shape[2]
            vdfeatures = vdfeatures.repeat(1, 1, repeat_times)
            new_features = torch.cat([new_features, vdfeatures], dim=1)

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class ConvBlock(nn.Module):

    def __init__(self,
                 radius,
                 npoint,
                 neighbour_points,
                 use_xyz,
                 mlp_list):
        """

        :param radius: 
        :param npoint: 
        :param neighbour_points: 
        :param use_xyz: whether use xyz
        :param mlp_list: mlp channels
        """
        super(ConvBlock, self).__init__()

        self.radius = radius
        self.neighbour_points = neighbour_points
        self.use_xyz = use_xyz
        self.npoint = npoint

        self.groupers = pointnet2_utils.QueryAndGroup(self.radius,
                                                      self.neighbour_points,
                                                      use_xyz=self.use_xyz)

        if self.use_xyz:
            mlp_list[0] += 3

        self.mlp = nn.Sequential()
        for i in range(len(mlp_list) - 1):
            self.mlp.add_module("conv_" + str(i), nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1, bias=False))
            self.mlp.add_module("bn_" + str(i), nn.BatchNorm2d(mlp_list[i + 1]))
            self.mlp.add_module("relu_" + str(i), nn.ReLU(inplace=True))

    def forward(self,
                xyz,
                features):
        """
        model forward
        :param xyz: 
        :param features: 
        :return: 
        """
        # xyz.shape: [batch, N, 3], new_idx.shape: [batch, npoint]
        new_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)

        # xyz_flipped.shape: [batch, 3, N], new_xyz.shape: [batch, npoint, 3]
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, new_idx).transpose(1, 2).contiguous()

        # features.shape: [batch, C, N], new_features.shape: [batch, C, npoint, nsample]
        new_features = self.groupers(xyz, new_xyz, features).contiguous()
        new_features = self.mlp(new_features)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(-1)

        return new_xyz, new_features


class VDModule(nn.Module):

    def __init__(self,
                 use_xyz,
                 channels):
        super(VDModule, self).__init__()

        self.use_xyz = use_xyz
        count = 4
        self.mlp = nn.Conv2d(channels, channels // count, kernel_size=1, bias=False)

    def forward(self,
                xyz,
                new_xyz,
                features):
        num_points = xyz.shape[1]
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)

        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                # new_features.shape: [B, 3 + C, 1, N]
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        new_features = self.mlp(new_features)

        center = new_xyz[:, -1, :].unsqueeze(1)
        dist = torch.sum((new_xyz - center) ** 2, -1)
        (_, idx) = torch.topk(dist, num_points, dim=-1, largest=False)
        idx = idx.int()
        new_features = pointnet2_utils.grouping_operation(new_features.squeeze(2).contiguous(), idx.unsqueeze(1))

        new_features = F.max_pool2d(new_features, kernel_size=[1, num_points // 4], stride=[1, num_points // 4])

        return new_features.squeeze(2)


class SegModel(nn.Module):

    def __init__(self,
                 use_xyz,
                 input_channels,
                 vdchannels,
                 radius_list=[],
                 npoint_list=[],
                 neighbour_points_list=[],
                 all_mlp_list=[],
                 num_classes=None,
                 dropout_rate=None):
        super(SegModel, self).__init__()

        print("SegModel")
        self.use_xyz = use_xyz
        self.vdchannels = vdchannels

        """down-sampling"""
        self.conv_0 = ConvBlock(radius_list[0], npoint_list[0],
                                neighbour_points_list[0], self.use_xyz, all_mlp_list[0])
        self.conv_1 = ConvBlock(radius_list[1], npoint_list[1],
                                neighbour_points_list[1], self.use_xyz, all_mlp_list[1])
        self.conv_2 = ConvBlock(radius_list[2], npoint_list[2],
                                neighbour_points_list[2], self.use_xyz, all_mlp_list[2])
        self.conv_3 = ConvBlock(radius_list[3], npoint_list[3],
                                neighbour_points_list[3], self.use_xyz, all_mlp_list[3])

        if self.vdchannels:
            self.vdmodule = VDModule(False, 512)
        self.concat_channels = [input_channels, 64, 128, 256]

        """up-sampling"""
        self.up_sample_3 = FPModule([512 + self.concat_channels[3], 512, 256],
                                    self.concat_channels[3], vdchannels, dropout_rate)
        self.up_sample_2 = FPModule([256 + self.concat_channels[2], 256, 256],
                                    self.concat_channels[2], None, dropout_rate)
        self.up_sample_1 = FPModule([256 + self.concat_channels[1], 256, 128],
                                    self.concat_channels[1], None, dropout_rate)
        self.up_sample_0 = FPModule([128 + self.concat_channels[0], 128, 128, 128],
                                    self.concat_channels[0], None, dropout_rate)

        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(128, num_classes, kernel_size=1, bias=True)
        )

        self.init_weights()

    def forward(self, point_cloud):
        """
        model forward
        :param point_cloud: 
        :return:
        """
        xyz, features = self.split_xyz_features(point_cloud)

        xyz_dict = {}
        features_dict = {}

        new_xyz, new_features = self.conv_0(xyz, features)
        xyz_dict["conv_0"] = new_xyz
        features_dict["conv_0"] = new_features

        new_xyz, new_features = self.conv_1(new_xyz, new_features)
        xyz_dict["conv_1"] = new_xyz
        features_dict["conv_1"] = new_features

        new_xyz, new_features = self.conv_2(new_xyz, new_features)
        xyz_dict["conv_2"] = new_xyz
        features_dict["conv_2"] = new_features

        new_xyz, new_features = self.conv_3(new_xyz, new_features)
        xyz_dict["conv_3"] = new_xyz
        features_dict["conv_3"] = new_features

        if self.vdchannels is not None:
            vdfeatures = self.vdmodule(new_xyz,
                                       new_xyz,
                                       new_features)
        else:
            vdfeatures = None

        new_features = self.up_sample_3(xyz_dict["conv_2"],
                                        xyz_dict["conv_3"],
                                        features_dict["conv_2"],
                                        features_dict["conv_3"],
                                        vdfeatures)
        features_dict["conv_2"] = new_features
        new_features = self.up_sample_2(xyz_dict["conv_1"],
                                        xyz_dict["conv_2"],
                                        features_dict["conv_1"],
                                        features_dict["conv_2"],
                                        vdfeatures)
        features_dict["conv_1"] = new_features
        new_features = self.up_sample_1(xyz_dict["conv_0"],
                                        xyz_dict["conv_1"],
                                        features_dict["conv_0"],
                                        features_dict["conv_1"],
                                        vdfeatures)
        features_dict["conv_0"] = new_features
        new_features = self.up_sample_0(xyz,
                                        xyz_dict["conv_0"],
                                        features,
                                        features_dict["conv_0"],
                                        vdfeatures)

        logits = self.fc(new_features).transpose(1, 2).contiguous()

        return logits

    def split_xyz_features(self, point_cloud):
        """
        split attributes
        :param point_cloud: 
        :return: 
        """
        # [batch, N, 3]
        xyz = point_cloud[..., 0:3].contiguous()
        # [batch, C, N]
        features = point_cloud[..., 3:].transpose(1, 2).contiguous() if point_cloud.size(-1) > 3 else None

        return xyz, features

    def init_weights(self):
        """
        weight initialization
        :return: 
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SALoss(nn.Module):

    def __init__(self,
                 num_classes,
                 weight,
                 epsilon):
        super(SALoss, self).__init__()

        self.num_classes = num_classes
        self.weight = weight
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self,
                logits,
                targets):
        log_probs = self.logsoftmax(logits)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        targets = targets * self.weight
        loss = (-targets * log_probs).mean(0).sum()

        return loss
