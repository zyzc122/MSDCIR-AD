# -*- coding: utf-8 -*-
# @Date : 2022-07-07
# @Author : zyz
# @File : Compare Models

import torch, kornia
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from utils.GaussionBlur import get_gaussian_kernel
from loss.ssim_loss import SSIM

class ProjectNet(nn.Module):
	def __init__(self, mode=2):
		super(ProjectNet, self).__init__()
		self.mode = mode
		if mode == 0:
			self.n_layer = 8
		elif mode == 1:
			self.n_layer = 9
		else:
			self.n_layer = 11
		self.avg = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
		self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

		self.pool_layer = nn.AvgPool2d((2, 2), (2, 2))
		backbone = models.vgg16(pretrained=True)
		self.layer1_1 = backbone.features[:2]
		self.layer1_2 = backbone.features[2:5]

		self.layer2_1 = backbone.features[5:7]
		self.layer2_2 = backbone.features[7:9]

		self.layer3_1 = backbone.features[10:11]
		self.layer3_2 = backbone.features[12:13]
		self.layer3_3 = backbone.features[14:15]

		self.layer4_1 = backbone.features[17:18]
		self.layer4_2 = backbone.features[19:20]
		self.layer4_3 = backbone.features[21:22]

		self.layer5_1 = backbone.features[24:25]
		self.layer5_2 = backbone.features[26:27]
		self.layer5_3 = backbone.features[28:29]
		self.acitivatelayer = nn.Sigmoid()

	def project(self, x):
		z_1_1 = self.layer1_1(x)
		z_1_2 = self.layer1_2(z_1_1)

		z_2_1 = self.layer2_1(z_1_2)
		# z_2_1 = self.acitivatelayer(z_2_1)
		z_2_2 = self.layer2_2(z_2_1)
		# z_2_2 = self.acitivatelayer(z_2_2)

		z_3_1 = self.layer3_1(self.pool_layer(z_2_2))
		z_3_1 = self.acitivatelayer(z_3_1)
		z_3_2 = self.layer3_2(z_3_1)
		z_3_2 = self.acitivatelayer(z_3_2)
		z_3_3 = self.layer3_3(z_3_2)
		z_3_3 = self.acitivatelayer(z_3_3)

		z_4_1 = self.layer4_1(self.pool_layer(z_3_3))
		z_4_1 = self.acitivatelayer(z_4_1)
		z_4_2 = self.layer4_2(z_4_1)
		z_4_2 = self.acitivatelayer(z_4_2)
		z_4_3 = self.layer4_3(z_4_2)
		z_4_3 = self.acitivatelayer(z_4_3)

		z_5_1 = self.layer5_1(self.pool_layer(z_4_3))
		z_5_1 = self.acitivatelayer(z_5_1)
		z_5_2 = self.layer5_2(z_5_1)
		z_5_2 = self.acitivatelayer(z_5_2)
		z_5_3 = self.layer5_3(z_5_2)
		z_5_3 = self.acitivatelayer(z_5_3)

		if self.mode == 1:
			return [z_3_1, z_3_2, z_3_3, z_4_1, z_4_2, z_4_3, z_5_1, z_5_2, z_5_3]
		if self.mode == 0:
			return [z_2_1, z_2_2, z_3_1, z_3_2, z_3_3, z_4_1, z_4_2, z_4_3]

		if self.mode == 3:
			return [z_2_2, z_3_3, z_4_3, z_5_3]
		return [z_2_1, z_2_2, z_3_1, z_3_2, z_3_3, z_4_1, z_4_2, z_4_3, z_5_1, z_5_2, z_5_3]
	# return [torch.cat((z_2_1, z_2_2), 1), torch.cat((z_3_1, z_3_2, z_3_3), 1), torch.cat((z_4_1, z_4_2, z_4_3), 1), torch.cat((z_5_1, z_5_2, z_5_3), 1)]

	def forward(self, x):
		x = (x - self.avg.to(x.device)) / self.std.to(x.device)

		return self.project(x)

def Covariance_Recursive(x, U, C, n):
	(_, c, nx, ny) = x.size()
	_x = x.permute(0, 2, 3, 1).reshape(-1, c)
	u_x = torch.mean(_x, 0, keepdim=True)
	s_x = _x - u_x
	s_x = s_x.T @ s_x
	s_u = U.T @ U
	a = 1 / (n + 1)
	s_x = a * (n * (C + s_u) + s_x)
	U = a * (n * U + u_x)
	s_u = U.T @ U
	C = s_x - s_u
	return U, C

class PCA_whitten(nn.Module):
	def __init__(self, ncom, nlayer):
		super(PCA_whitten, self).__init__()
		self.nlayer = nlayer
		self.W = [torch.zeros((1, 1), dtype=torch.float32) for l in range(self.nlayer)]
		self.ncom = [ncom for l in range(self.nlayer)]
		self.U = [torch.zeros((1, 1), dtype=torch.float32) for l in range(self.nlayer)]
		self.C = [torch.zeros((1, 1), dtype=torch.float32) for l in range(self.nlayer)]
		self.n = 0

	def p_fit(self, f):
		for l in range(self.nlayer):
			if self.C[l].size()[0] == 1:
				c = f[l].size()[1]
				self.C[l] = torch.zeros((c, c), dtype=torch.float32).to(f[l].device)
				self.U[l] = torch.zeros((1, c), dtype=torch.float32).to(f[l].device)
			self.U[l], self.C[l] = Covariance_Recursive(f[l], self.U[l], self.C[l], self.n)
		self.n = self.n + 1


	def fit(self):
		for l in range(self.nlayer):
			U, S, V = torch.linalg.svd(self.C[l])
			self.W[l] = torch.matmul(U, torch.diag_embed(torch.sqrt(1 / S))[:, :self.ncom[l]])
		return

	def forward(self, X):
		for l in range(self.nlayer):
			n, c, h, w = X[l].size()
			x = X[l].permute(0, 2, 3, 1).reshape(-1, c)
			x = torch.matmul(x - self.U[l].to(x.device), self.W[l].to(x.device))
			X[l] = x.reshape(1, h, w, -1).permute(0, 3, 1, 2)
		return X


class CompareModel(nn.Module):
	def __init__(self, size, n_layer, device):
		super(CompareModel, self).__init__()
		self.n_layer = n_layer
		self.n_blur = 6
		self.device = device
		self.U = [torch.zeros((1, 1), dtype=torch.float32).to(device) for l in range(self.n_layer)]
		self.S = [torch.zeros((1, 1), dtype=torch.float32).to(device) for l in range(self.n_layer)]
		self.C = [torch.zeros((1, 1), dtype=torch.float32).to(device) for l in range(self.n_layer)]
		self.Avg = [torch.zeros((1, ), dtype=torch.float32).to(device) for l in range(self.n_layer)]
		self.Std = [torch.ones((1, ), dtype=torch.float32).to(device) for l in range(self.n_layer)]
		self.Color_Avg = [torch.zeros((1, ), dtype=torch.float32).to(device) for l in range(self.n_blur)]
		self.Color_Std = [torch.ones((1, ), dtype=torch.float32).to(device) for l in range(self.n_blur)]
		self.SSIM_Avg = [torch.zeros((1, ), dtype=torch.float32).to(device) for l in range(self.n_blur)]
		self.SSIM_Std = [torch.ones((1, ), dtype=torch.float32).to(device) for l in range(self.n_blur)]
		self.P_CNN = [0, 1]
		self.P_COLOR = [0, 1]
		self.P_SSIM = [0, 1]
		self.size = size
		self.blur_layer = [get_gaussian_kernel(kernel_size=4*i+5).to(device) for i in range(self.n_blur-1)]
		self.ssim_layer = SSIM(window_size=11, size_average=False).to(device)

		self.avg_layer = nn.AvgPool2d((21, 21), stride=(1, 1), padding=10)
		self.th = 0

	def blur(self, x):
		y = [self.blur_layer[i](x) for i in range(self.n_blur-1)]
		return torch.cat(y, dim=0)

	def RGB2LAB(self, x):
		return kornia.color.rgb_to_lab(x)

	def CNN_M_distance_para_fit(self, f, l, n):
		if self.C[l].size()[0] == 1:
			c = f.size()[1]
			self.C[l] = torch.zeros((c, c), dtype=torch.float32).to(self.device)
			self.U[l] = torch.zeros((1, c), dtype=torch.float32).to(self.device)
		self.U[l], self.C[l] = Covariance_Recursive(f, self.U[l], self.C[l], n)
		s_mean = self.C[l].mean() * 0.01
		self.S[l] = torch.linalg.pinv(self.C[l] + s_mean)
		return

	def Normalization_para_fit(self, X_CNN, X_COLOR, X_SSIM):
		self.P_CNN = [torch.mean(X_CNN).to(self.device),
		              torch.std(X_CNN).to(self.device)]
		self.P_COLOR = [torch.mean(X_COLOR).to(self.device),
		                torch.std(X_COLOR).to(self.device)]
		self.P_SSIM = [torch.mean(X_SSIM).to(self.device),
		               torch.std(X_SSIM).to(self.device)]
		return

	def Layer_Normalization_para_fit(self, X_CNN, X_COLOR, X_SSIM):
		for l in range(self.n_layer):
			d = torch.cat(X_CNN[l], 0)
			X_CNN[l] = []
			self.Avg[l] = d.mean()
			self.Std[l] = d.std()
		for l in range(self.n_blur):
			d = torch.cat(X_COLOR[l], 0)
			X_COLOR[l] = []
			self.Color_Avg[l] = d.mean()
			self.Color_Std[l] = d.std()
			d = torch.cat(X_SSIM[l], 0)
			X_SSIM[l] = []
			self.SSIM_Avg[l] = d.mean()
			self.SSIM_Std[l] = d.std()
		return

	def reshape(self, x, y):
		assert y.size() == x.size()
		(_, channel, nx, ny) = x.size()
		_x = x.permute(0, 2, 3, 1).reshape(-1, channel)
		_y = y.permute(0, 2, 3, 1).reshape(-1, channel)
		return _x, _y, nx, ny

	def weight_est(self, t):
		t_max = torch.max(t)
		t_min = torch.min(t)
		t_max = t_max - t_min
		t = (t - t_min) / t_max
		t_max = F.max_pool2d(t, 32, stride=32)
		return torch.max(t_max) - torch.min(t_max)

	def M_distance(self, x, y, l):
		x, y, nx, ny = self.reshape(x, y)
		d = x - y - self.U[l]
		d = torch.sum(torch.mul(torch.mm(d, self.S[l]), d), 1).reshape(1, 1, nx, ny)
		d[d<0]=0
		return torch.sqrt(d)

	def Color_distance(self, x, y):
		x, y, nx, ny = self.reshape(x, y)
		return 1 - torch.cosine_similarity(x, y, 1).reshape(1, 1, nx, ny)

	def ssim_distance(self, x, y):
		return 1 - self.ssim_layer(x, y)

	def texture_color_distance(self, x0, x1):
		d_color = self.Color_distance(x0, x1)
		d_color = (d_color - self.Color_Avg[0]) / self.Color_Std[0]
		d_ssim = self.ssim_distance(x0, x1)
		d_ssim = (d_ssim - self.SSIM_Avg[0]) / self.SSIM_Std[0]
		x_list_0 = self.blur(x0)
		x_list_1 = self.blur(x1)
		for i in range(self.n_blur-1):
			t = self.Color_distance(x_list_0[i].unsqueeze(0), x_list_1[i].unsqueeze(0)).to(self.device)
			t = (t - self.Color_Avg[i+1]) / self.Color_Std[i+1]
			d_color = d_color + t
			t = self.ssim_distance(x_list_0[i].unsqueeze(0), x_list_1[i].unsqueeze(0)).to(self.device)
			t = (t - self.SSIM_Avg[i + 1]) / self.SSIM_Std[i + 1]
			d_ssim = d_ssim + t
		return d_color, d_ssim

	def forward(self, x0, x1, f0, f1, flag=False, ablation=False):
		d_color, d_ssim = self.texture_color_distance(x0, x1)
		d_cnn = torch.zeros((1, 1, self.size, self.size)).to(self.device)
		for l in range(self.n_layer):
			if torch.isnan(self.Std[l]):
				continue
			t = (self.M_distance(f0[l], f1[l], l) - self.Avg[l]) / self.Std[l]
			t = F.interpolate(t, self.size, mode='bilinear', align_corners=False)
			d_cnn = d_cnn + t.squeeze()
		d = torch.mean(torch.square(x0-x1), 1, keepdim=True).to(self.device)
		if flag:
			d_ssim = ((d_ssim - self.P_SSIM[0]) / self.P_SSIM[1])
			d_color = ((d_color - self.P_COLOR[0]) / self.P_COLOR[1])
			# d_cnn = self.avg_layer(d_cnn)
			# d_ssim = self.avg_layer(d_ssim)
			# d_color = self.avg_layer(d_color)
			d_seg = (d_cnn + d_ssim + d_color)
			if self.size == 512:
				d_seg = self.avg_layer(d_seg)

			d_det = d_seg.reshape([1, -1])
			d_det = torch.topk(d_det, 512).values
			if ablation == True:
				return {'cnn': d_cnn,
						'color': d_color,
						'ssim': d_ssim,
						'cnn_color': d_cnn + d_color,
						'cnn_ssim': d_cnn + d_ssim,
						'color_ssim': d_color + d_ssim,
						'seg': d_seg,
						'det_std': d_seg.std() + d_det.mean(),
						'det_max': d_det.mean()}
			else:
				return {'seg': d_seg,
						'det_std': 0,
						'det_max': d_det.mean()}
		return d_cnn, d_color, d_ssim
