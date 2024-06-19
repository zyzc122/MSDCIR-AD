# -*- coding: utf-8 -*-
# @Date : 2022-07-07
# @Author : zyz
# @File : loss

import torch
from torch import nn
from torch.functional import F

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, target, pt):
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt + 1e-3) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + 1e-3)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class ColorLoss(nn.Module):
    def __init__(self, ndim=-1):
        super(ColorLoss, self).__init__()
        self.ndim = ndim

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        _img1 = img1.permute(0, 2, 3, 1).reshape(-1, channel)
        _img2 = img2.permute(0, 2, 3, 1).reshape(-1, channel)
        return 1 - F.cosine_similarity(_img1, _img2, dim=self.ndim).mean()

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        grad_x_1 = torch.abs(img1[:, :, :, :-1] - img1[:, :, :, 1:])
        grad_y_1 = torch.abs(img1[:, :, :-1, :] - img1[:, :, 1:, :])

        grad_x_2 = torch.abs(img2[:, :, :, :-1] - img2[:, :, :, 1:])
        grad_y_2 = torch.abs(img2[:, :, :-1, :] - img2[:, :, 1:, :])

        grad_x = grad_x_1 * torch.exp(-grad_x_2) + grad_x_2 * torch.exp(-grad_x_1)
        grad_y = grad_y_1 * torch.exp(-grad_y_2) + grad_y_2 * torch.exp(-grad_y_1)

        return grad_x.mean() + grad_y.mean()

class StdLoss(nn.Module):
    def __init__(self):
        super(StdLoss, self).__init__()

    def forward(self, img1, img2):
        b, c, m, n = img1.size()
        r = torch.mean(torch.square(img1 - img2), 1)
        r = r.reshape(b, -1)
        return torch.mean(torch.std(r, -1))