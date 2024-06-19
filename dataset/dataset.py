# -*- coding: utf-8 -*-
# @Date : 2022-07-01
# @Author : zyz
# @File : dataset

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from torchvision import transforms
import numpy as np
import pandas as pd

object_list = ['cable', 'zipper', 'capsule', 'transistor', 'bottle', 'hazelnut',
               'metal_nut',  'toothbrush', 'pill', 'screw']
text_list = ['carpet',  'tile', 'leather', 'grid', 'wood']

btad_list = ['btad03', 'btad01', 'btad02']
visa_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

def get_dataset_list(dataset_name):
    if dataset_name == 'MVTAD':
        dataset_list = object_list + text_list
    elif dataset_name == 'BTAD':
        dataset_list = btad_list
    elif dataset_name == 'visa':
        dataset_list = visa_list
    else:
        dataset_list = []
    return dataset_list


class MVTecAD(Dataset):
    def __init__(self, root_dir, class_name, size, mode="train"):
        self.root_dir = root_dir
        self.class_name = class_name
        self.mode = mode
        self.size = size
        # self.H = 1024
        # self.n_patch = self.H // self.size
        # self.n_patch = self.n_patch ** 2

        self.transform_gt = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.CenterCrop((self.size, self.size)),
            transforms.ToTensor()])

        if mode in ['train']:
            if class_name in ['toothbrush', 'transistor', 'zipper']:
                self.transform = transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01),
                    transforms.ToTensor(),
                ])
            elif class_name in ['bottle', 'hazelnut', 'screw']:
                self.transform = transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.RandomRotation(90),
                    transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01),
                    transforms.ToTensor(),
                ])
            elif class_name in ['grid', 'leather', 'tile', 'carpet', 'wood', 'metal_nut', 'screw_bag', 'pushpins', 'splicing_connectors']:
                self.transform = transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomRotation(3),
                    transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.ToTensor(),
                    transforms.RandomRotation(3),
                    transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01)
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:

            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor()
            ])

        self.image_names, self.y, self.gt = self.load_dataset()

    def __len__(self):
        return len(self.image_names)

    def load_dataset(self):
        x, y, gt = [],  [], []
        image_path = os.path.join(self.root_dir, self.class_name, self.mode)
        gt_path = os.path.join(self.root_dir, self.class_name, 'ground_truth')
        image_types = sorted(os.listdir(image_path))
        end_with = '.png'
        gt_end_with = '_mask.png'
        # if self.class_name in btad_list[:2]:
        # 	end_with = '.bmp'
        # if self.class_name in btad_list[1:]:
        # 	gt_end_with = '.png'
        # elif self.class_name == btad_list[0]:
        # 	gt_end_with = '.bmp'

        for type in image_types:
            image_type_path = os.path.join(image_path, type)
            if not os.path.isdir(image_type_path):
                continue
            image_list = sorted([os.path.join(image_type_path, f)
                                 for f in os.listdir(image_type_path) if f.endswith(end_with)])
            x.extend(image_list)

            # y.extend([0] * len(image_list))
            # gt.extend([None] * len(image_list))
            if type == 'good':
                y.extend([0]*len(image_list))
                gt.extend([None] * len(image_list))
            else:
                y.extend([1] * len(image_list))
                image_name_list = [os.path.splitext(os.path.basename(f))[0] for f in image_list]

                gt_list = [os.path.join(gt_path, type, image_name + gt_end_with)
                           for image_name in image_name_list]
                gt.extend(gt_list)
            assert len(x) == len(y) and len(x) == len(gt)
        return list(x), list(y), list(gt)

    def __getitem__(self, idx):
        image_names, y, gt = self.image_names[idx], self.y[idx], self.gt[idx]
        x = Image.open(image_names).convert('RGB')
        x = self.transform(x)
        if y == 0:
            mask = torch.zeros([1, self.size, self.size])
        else:
            mask = Image.open(gt).convert('1')
            mask = self.transform_gt(mask)
            # mask.to(torch.bool)
        return x, y, mask

class MVTecAD_3D(Dataset):
    def __init__(self, root_dir, class_name, size, mode="Train"):
        self.root_dir = root_dir
        self.class_name = class_name
        self.mode = mode
        self.size = size

        self.transform_gt = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor()])

        if mode in ['train', 'validation']:
            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(3),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor()
            ])

        self.image_names, self.y, self.gt = self.load_dataset()

    def __len__(self):
        return len(self.image_names)

    def load_dataset(self):
        x, y, gt = [],  [], []
        image_path = os.path.join(self.root_dir, self.class_name, self.mode)
        image_types = sorted(os.listdir(image_path))
        end_with = '.png'

        for type in image_types:
            image_list = glob(f'{self.root_dir}/{self.class_name}/{self.mode}/{type}/rgb/*.png')
            x.extend(image_list)
            if type == 'good':
                y.extend([0]*len(image_list))
                gt.extend([None] * len(image_list))
            else:
                y.extend([1] * len(image_list))
                image_name_list = [os.path.splitext(os.path.basename(f))[0] for f in image_list]
                gt_list = [f'{self.root_dir}/{self.class_name}/{self.mode}/{type}/gt/{image_name}.png' for image_name in image_name_list]
                gt.extend(gt_list)
            assert len(x) == len(y) and len(x) == len(gt)
        return list(x), list(y), list(gt)

    def __getitem__(self, idx):
        image_names, y, gt = self.image_names[idx], self.y[idx], self.gt[idx]
        x = Image.open(image_names).convert('RGB')
        x = self.transform(x)
        if y == 0:
            mask = torch.zeros([1, self.size, self.size])
        else:
            mask = Image.open(gt).convert('1')
            mask = self.transform_gt(mask)
            # mask.to(torch.bool)
        return x, y, mask

class BTAD(Dataset):
    def __init__(self, root_dir, class_name, size, mode="Train"):
        self.root_dir = root_dir
        self.class_name = class_name
        self.mode = mode
        self.size = size

        self.transform_gt = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.CenterCrop((self.size, self.size)),
            transforms.ToTensor()])

        if mode in ['Train', 'Eva']:
            if class_name in ['btad01', 'batd03']:
                self.transform = transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01),
                    transforms.ToTensor(),
                ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor()
            ])

        self.image_names, self.y, self.gt = self.load_dataset()

    def __len__(self):
        return len(self.image_names)

    def load_dataset(self):
        x, y, gt = [],  [], []
        image_path = os.path.join(self.root_dir, self.class_name, self.mode)
        gt_path = os.path.join(self.root_dir, self.class_name, 'ground_truth')
        image_types = sorted(os.listdir(image_path))
        end_with = '.png'
        gt_end_with = '_mask.png'
        if self.class_name in btad_list[:2]:
            end_with = '.bmp'
        if self.class_name in btad_list[1:]:
            gt_end_with = '.png'
        elif self.class_name == btad_list[0]:
            gt_end_with = '.bmp'

        for type in image_types:
            image_type_path = os.path.join(image_path, type)
            if not os.path.isdir(image_type_path):
                continue
            image_list = sorted([os.path.join(image_type_path, f)
                                 for f in os.listdir(image_type_path) if f.endswith(end_with)])
            x.extend(image_list)
            if type == 'ok':
                y.extend([0]*len(image_list))
                gt.extend([None] * len(image_list))
            else:
                y.extend([1] * len(image_list))
                image_name_list = [os.path.splitext(os.path.basename(f))[0] for f in image_list]
                gt_list = [os.path.join(gt_path, type, image_name + gt_end_with)
                           for image_name in image_name_list]
                gt.extend(gt_list)
            assert len(x) == len(y) and len(x) == len(gt)
        return list(x), list(y), list(gt)

    def __getitem__(self, idx):
        image_names, y, gt = self.image_names[idx], self.y[idx], self.gt[idx]
        x = Image.open(image_names).convert('RGB')
        x = self.transform(x)
        if y == 0:
            mask = torch.zeros([1, self.size, self.size])
        else:
            mask = Image.open(gt).convert('1')
            mask = self.transform_gt(mask)
            # mask.to(torch.bool)
        return x, y, mask

class VISA(Dataset):
    def __init__(self, root_dir, class_name, size, mode="train"):
        self.root_dir = root_dir
        self.class_name = class_name
        self.mode = mode
        self.size = size

        self.transform_gt = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor()])

        if mode in ['train']:
            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor()
            ])

        self.image_names, self.y, self.gt = self.load_dataset()

    def __len__(self):
        return len(self.image_names)

    def load_dataset(self):
        data_path = pd.read_csv(f'{self.root_dir}split_csv/1cls.csv')
        data_path = data_path[
            (data_path.object == self.class_name) & (data_path.split == self.mode)]
        x = data_path['image'].values
        y = data_path['label'].values
        gt = data_path['mask'].values
        return list(x), list(y), list(gt)

    def __getitem__(self, idx):
        image_names, y, gt = self.image_names[idx], self.y[idx], self.gt[idx]
        x = Image.open(f'{self.root_dir}{image_names}').convert('RGB')
        x = self.transform(x)
        if y == 'normal':
            mask = torch.zeros([1, self.size, self.size])
            y = 0
        else:
            mask = Image.open(f'{self.root_dir}{gt}')
            mask = np.array(mask)
            mask[mask != 0] = 255
            mask = Image.fromarray(mask)
            mask.save(f'{self.root_dir}{gt}')
            mask = Image.open(f'{self.root_dir}{gt}').convert('1')
            mask = self.transform_gt(mask)
            y = 1
            # mask.to(torch.bool)
        return x, y, mask

import cv2
if __name__ == '__main__':
    H = 1024
    s = 512
    N = H // s
    for class_name in text_list:
        files = glob(f'./dataset/{class_name}/Test1/*/*.png')
        for file in files:
            ext = file.split('/')[-2]
            id = file.split('/')[-1].split('.')[0]
            if not os.path.exists(f'./dataset/{class_name}/Test/{ext}/'):
                os.makedirs(f'./dataset/{class_name}/Test/{ext}/')
            x = cv2.imread(file)
            if x.shape[1] != H:
                x = cv2.resize(x, (H, H))
            start_x = 0
            for lx in range(N):
                start_y = 0
                for ly in range(N):
                    cv2.imwrite(f'./dataset/{class_name}/Test/{ext}/{lx}_{ly}_{id}.png',
                                x[start_x:start_x+s, start_y:start_y+s, :])
                    start_y += s
                start_x += s
        files = glob(f'./dataset/{class_name}/ground_truth1/*/*.png')
        for file in files:
            ext = file.split('/')[-2]
            id = file.split('/')[-1].split('.')[0]
            if not os.path.exists(f'./dataset/{class_name}/ground_truth/{ext}/'):
                os.makedirs(f'./dataset/{class_name}/ground_truth/{ext}/')
            x = cv2.imread(file)
            if x.shape[1] != H:
                x = cv2.resize(x, (H, H))
            start_x = 0
            for lx in range(N):
                start_y = 0
                for ly in range(N):
                    cv2.imwrite(f'./dataset/{class_name}/ground_truth/{ext}/{lx}_{ly}_{id}.png',
                                x[start_x:start_x+s, start_y:start_y+s, :])
                    start_y += s
                start_x += s
        # files = glob(f'./dataset/{class_name}/Eva/*/*.png')
        # for file in files:
        # 	ext = file.split('/')[-2]
        # 	id = file.split('/')[-1].split('.')[0]
        # 	x = cv2.imread(file)
        # 	x = np.transpose(x, (1, 0, 2))
        # 	cv2.imwrite(f'./dataset/{class_name}/Eva/{ext}/1{id}.png', x)
        #
        # files = glob(f'./dataset/{class_name}/Train/*/*.png')
        # for file in files:
        # 	ext = file.split('/')[-2]
        # 	id = file.split('/')[-1].split('.')[0]
        # 	x = cv2.imread(file)
        # 	x = np.transpose(x, (1, 0, 2))
        # 	cv2.imwrite(f'./dataset/{class_name}/Train/{ext}/1{id}.png', x)



