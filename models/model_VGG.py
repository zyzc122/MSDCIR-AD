# -*- coding: utf-8 -*-
# @Date : 2022-07-07
# @Author : zyz
# @File : Constrained Image Construction

import torch
import torch.nn as nn
from torchvision import models
from quantize import VQVAEQuantize, GumbelQuantize
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, nrow, ncol):
        super(PositionalEncoding, self).__init__()
        self.num_hiddens = num_hiddens
        self.dropout = nn.Dropout(dropout)
        P_row = torch.zeros((1, nrow, 1, num_hiddens))
        X = torch.arange(nrow, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        X = X.unsqueeze(1)
        P_row[:, :, :, 0::2] = torch.sin(X)
        P_row[:, :, :, 1::2] = torch.cos(X)

        P_col = torch.zeros((1, 1, ncol, num_hiddens))
        X = torch.arange(ncol, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        P_col[:, :, :, 0::2] = torch.sin(X)
        P_col[:, :, :, 1::2] = torch.cos(X)
        self.P = torch.einsum('b h c n, b c w n -> b n h w', P_row, P_col)

    def forward(self, X):
        X = X + self.P.to(X.device)
        return self.dropout(X)

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(decoder_block, self).__init__()

        self.conv_relu = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=(3, 3), padding=1),
                                       nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
                                       nn.Conv2d(in_c, in_c, kernel_size=(3, 3), padding=1),
                                       nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
                                       nn.Conv2d(in_c, in_c, kernel_size=(3, 3), padding=1))
        self.up = nn.Sequential(nn.BatchNorm2d(2*in_c),  nn.ReLU(inplace=True),
                                nn.Upsample(scale_factor=2),
                                nn.Conv2d(2*in_c, out_c, kernel_size=(3, 3), padding=1),
                                nn.BatchNorm2d(out_c),  nn.ReLU(inplace=True))

        # self.reduce_conv = nn.Sequential(nn.Conv2d(2 * out_c, out_c, kernel_size=(1, 1), padding=0),
        #                                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x):
        y = self.conv_relu(x)
        x = torch.cat([x, y], 1)
        return self.up(x)

class CVAE(nn.Module):
    def __init__(self, n_dim1, n_dim2, n_embedding1, n_embedding2, size, seed=3407):
        super(CVAE, self).__init__()
        self.seed = seed
        self.size = size
        self.backbone = models.vgg16(pretrained=True)
        self.layer1 = self.backbone.features[:24]
        self.layer2 = self.backbone.features[24:]

        self.layer_11 = nn.Sequential(nn.Conv2d(512, n_dim1, kernel_size=(3, 3), padding=1),
                                      nn.BatchNorm2d(n_dim1), nn.ReLU(inplace=True))

        self.layer_22 = nn.Sequential(nn.Conv2d(512, n_dim2, kernel_size=(3, 3), padding=1),
                                      nn.BatchNorm2d(n_dim2), nn.ReLU(inplace=True),
                                      nn.Conv2d(n_dim2, n_dim2, kernel_size=(3, 3), padding=1),
                                      nn.BatchNorm2d(n_dim2), nn.ReLU(inplace=True),
                                      nn.Conv2d(n_dim2, n_dim2, kernel_size=(3, 3), padding=1),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      nn.BatchNorm2d(n_dim2), nn.ReLU(inplace=True),
                                      nn.Conv2d(n_dim2, n_dim2, kernel_size=(3, 3), padding=1),
                                      nn.BatchNorm2d(n_dim2), nn.ReLU(inplace=True))

        # self.layer_33 = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4))

        self.Pd = PositionalEncoding(n_dim1, 0.25, self.size // 16, self.size // 16)
        self.Ps = PositionalEncoding(n_dim2, 0.25, self.size // 64, self.size // 64)

        self.fc_mu_1 = nn.Conv2d(n_dim1, n_dim1, kernel_size=(1, 1), padding=0)
        self.fc_var_1 = nn.Conv2d(n_dim1, n_dim1, kernel_size=(1, 1), padding=0)
        self.fc_mu_2 = nn.Conv2d(n_dim2, n_dim2, kernel_size=(1, 1), padding=0)
        self.fc_var_2 = nn.Conv2d(n_dim2, n_dim2, kernel_size=(1, 1), padding=0)

        self.quantize_d_mean = VQVAEQuantize(n_dim1, n_embedding1, n_dim1)
        self.quantize_s_mean = VQVAEQuantize(n_dim2, n_embedding2, n_dim2)

        self.quantize_d_var = VQVAEQuantize(n_dim1, n_embedding1, n_dim1)
        self.quantize_s_var = VQVAEQuantize(n_dim2, n_embedding2, n_dim2)

        n_dim1_2 = n_dim1 // 2
        n_dim1_4 = n_dim1 // 4
        self.decoder_0 = decoder_block(n_dim2, n_dim2)
        self.decoder_1 = decoder_block(n_dim2, n_dim2)
        self.decoder_2 = decoder_block(n_dim2 + n_dim1, n_dim2)
        self.decoder_3 = decoder_block(n_dim2, n_dim1)
        self.decoder_4 = decoder_block(n_dim1, n_dim1_2)
        self.decoder_5 = decoder_block(n_dim1_2, n_dim1_4)
        self.conv_6 = nn.Sequential(nn.Conv2d(n_dim1_4, 3, kernel_size=(3, 3), padding=1), nn.Sigmoid())

    def encoder(self, x):
        z1 = self.layer1(x)
        z2 = self.layer2(z1)
        z1 = self.layer_11(z1)
        z2 = self.layer_22(z2)
        z1 = self.Pd(z1)
        z2 = self.Ps(z2)
        mu_1 = self.fc_mu_1(z1)
        var_1 = self.fc_var_1(z1)
        mu_2 = self.fc_mu_2(z2)
        var_2 = self.fc_var_2(z2)
        return mu_1, var_1, mu_2, var_2

    def sampleing(self, mu, log_var, train):
        if not train:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, x1, x2):
        x2 = self.decoder_0(x2)
        x2 = self.decoder_1(x2)
        x2 = torch.cat([x1, x2], 1)
        x2 = self.decoder_2(x2)
        x2 = self.decoder_3(x2)
        x2 = self.decoder_4(x2)
        x2 = self.decoder_5(x2)
        x2 = self.conv_6(x2)
        return x2

    def forward(self, x, train=True):
        mu1, var1, mu2, var2, = self.encoder(x)

        KLD = -0.015 * (torch.mean(1 + var1 - mu1.pow(2) - var1.exp()) +
                        torch.mean(1 + var2 - mu2.pow(2) - var2.exp()))

        mu1, diff_mu_1, id_t = self.quantize_d_mean(mu1)
        mu2, diff_mu_2, id_t = self.quantize_s_mean(mu2)

        var1, diff_var_1, id_t = self.quantize_d_var(var1)
        var2, diff_var_2, id_t = self.quantize_s_var(var2)

        diff = diff_mu_1 + diff_mu_2 + diff_var_1 + diff_var_2

        mu1 = nn.Dropout(0.5)(mu1)
        var1 = nn.Dropout(0.5)(var1)
        mu2 = nn.Dropout(0.5)(mu2)
        var2 = nn.Dropout(0.5)(var2)

        x2 = self.sampleing(mu2, var2, train)
        x1 = self.sampleing(mu1, var1, train)

        y = self.decoder(x1, x2)
        return y, diff.unsqueeze(0), KLD
