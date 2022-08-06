# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, gamma=0.5, focal_loss_enable=False, focal_temp=1.0):
        super(JointsMSELoss, self).__init__()
        if focal_loss_enable:
            self.criterion = nn.MSELoss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.gamma = gamma
        self.focal_loss_enable = focal_loss_enable
        self.softmax = torch.nn.Softmax(dim=0)
        self.focal_temp = focal_temp

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        H, W = output.size(2), output.size(3)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                _loss = self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                       heatmap_gt.mul(target_weight[:, idx]))
                if self.focal_loss_enable:
                    _max_ind = heatmap_gt.topk(1)[1]
                    w_gt, h_gt = _max_ind % W, torch.floor(_max_ind / W).int()
                    _max_ind = heatmap_pred.topk(1)[1]
                    w, h = _max_ind % W, torch.floor(_max_ind / W).int()
                    dist_weight = torch.pow(torch.square(w - w_gt) + torch.square(h - h_gt), self.gamma) / ((H+W)/2)
                    dist_weight = self.softmax(dist_weight/self.focal_temp)
                    _loss = torch.sum(_loss.mean(dim=1, keepdim=True) * dist_weight.detach())
            else:
                _loss = self.criterion(heatmap_pred, heatmap_gt)
            loss += 0.5 * _loss

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
