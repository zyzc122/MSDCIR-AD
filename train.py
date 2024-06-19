# -*- coding: utf-8 -*-
# @Date : 2022-07-07
# @Author : zyz
# @File : train
import random
from torch import optim
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm
from models.model_ResNet18 import CVAE
from models.model_compare import ProjectNet
from utils.scheduler import *
from loss.ssim_loss import *
from dataset.dataset import *
from loss.loss import *
import argparse
import os.path
import os
import numpy as np
from utils.imgs2video import imgs2video

seed = 3407
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def train_one_epoch(epoch, Train_loader, Eva_loader, Test_loader, model, project_model, optimizer, scheduler, device,
          dataset_name):
    Train_loader = tqdm(Train_loader)
    mse_loss = nn.MSELoss().to(device)

    mse_loss_weight = 0.15
    ssim_loss_1 = SSIM(window_size=11).to(device)
    ssim_loss_weight = 0.85

    color_loss = ColorLoss().to(device)
    smooth_loss = SmoothLoss().to(device)

    sample_size = min(args.batch_size, 8)

    mse_sum = 0
    mse_n = 0

    for i, (img, y, label) in enumerate(Train_loader):
        model.zero_grad()
        img = img.to(device)
        out, latent_loss, KLD = model(img)
        # f0 = project_model(img)
        # f1 = project_model(out)
        # d_l = mse_loss(f0[0], f1[0]) + mse_loss(f0[1], f1[1]) + mse_loss(f0[2], f1[2]) + mse_loss(f0[3], f1[3])
        recon_loss = mse_loss_weight * (mse_loss(out, img) + color_loss(out, img)) + \
                     ssim_loss_weight * (1 - ssim_loss_1(out, img)) + 0.1 * smooth_loss(out, img)

        if latent_loss == None and KLD == None:
            loss = recon_loss
        else:
            KLD_loss = KLD.mean()
            latent_loss = latent_loss.mean()

            loss = args.recon_weight * recon_loss + args.latent_weight * latent_loss + \
                   args.KLD_weight * KLD_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        Train_loader.set_description(
            (
                f'epoch: {epoch + 1}; '
                f'recon: {recon_loss.item():.4f}; '
                f'Latent: {latent_loss.item():.4f}; '
                f'KLD: {KLD.item():.4f};'
                f'lr: {lr:.5f}'
            )
        )

    model.eval()
    sample = []
    for j, (img_test, y, label) in enumerate(Test_loader):
        sample.append(img_test[:sample_size].to(device))
        break

    with torch.no_grad():
        out, _, _ = model(sample[0])

    utils.save_image(
        torch.cat([sample[0], out, torch.abs(sample[0] - out)], 0),
        f'sample/{dataset_name}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
        nrow=8,
        normalize=True
    )
    model.train()
    recon_loss_eva = 0
    for j, (img_eva, y, label) in enumerate(Eva_loader):
        img_eva = img_eva.to(device)
        with torch.no_grad():
            out, _, _ = model(img_eva)
            eva_loss = mse_loss_weight * (mse_loss(out, img_eva) + color_loss(out, img_eva)) + \
                       ssim_loss_weight * (1 - ssim_loss_1(out, img_eva)) + \
                       0.1 * smooth_loss(out, img_eva)

        recon_loss_eva += eva_loss.item() * img_eva.shape[0]
        mse_n += img_eva.shape[0]
    return recon_loss_eva / mse_n

def train_main(args, dataset_list):
    for dataset_name in dataset_list:
        if not os.path.exists(f'checkpoint/{dataset_name}'):
            os.makedirs(f'checkpoint/{dataset_name}')
        if not os.path.exists(f'sample/{dataset_name}'):
            os.makedirs(f'sample/{dataset_name}')

        if args.dataset == 'MVTAD':
            Train_dataset = MVTecAD(args.path, dataset_name, args.size, 'train')
            Eva_dataset = MVTecAD(args.path, dataset_name, args.size, 'validation')
            Test_dataset = MVTecAD(args.path, dataset_name, args.size, 'test')
        elif args.dataset == 'visa':
            Train_dataset = VISA(args.path, dataset_name, args.size, 'train')
            Eva_dataset = VISA(args.path, dataset_name, args.size, 'validation')
            Test_dataset = VISA(args.path, dataset_name, args.size, 'test')

        elif args.dataset == 'BTAD':
            Train_dataset = BTAD(args.path, dataset_name, args.size, 'Train')
            Eva_dataset = BTAD(args.path, dataset_name, args.size, 'Eva')
            Test_dataset = BTAD(args.path, dataset_name, args.size, 'Test')
        else:
            return

        Train_loader = DataLoader(Train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        Eva_loader = DataLoader(Eva_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        Test_loader = DataLoader(Test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        construct_model = CVAE(n_dim1=args.n_dim1, n_dim2=args.n_dim2,
                               n_embedding1=args.n_embedding1,
                               n_embedding2=args.n_embedding2, size=args.size).to(args.device)

        if os.path.exists(f'checkpoint/{dataset_name}/pre_train.pt'):
            construct_model.load_state_dict(torch.load(f'checkpoint/{dataset_name}/pre_train.pt'))
        construct_model = nn.DataParallel(construct_model)

        project_model = ProjectNet(mode=3).to(args.device)
        for param in project_model.parameters():
            param.requires_grad = False

        optimizer = optim.AdamW(construct_model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = CycleScheduler(optimizer, lr_max=args.lr, divider=50, warmup_proportion=0.25, n_iter=args.n_iter)

        for i in range(args.epoch):
            score = train_one_epoch(i, Train_loader, Eva_loader, Test_loader,
                                    construct_model, project_model, optimizer, scheduler, args.device,
                                    dataset_name)
            print(f'{i}: {score}')
        torch.save(construct_model.module.state_dict(), f'checkpoint/{dataset_name}/Cvae.pt')
        imgs2video(dataset_name)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Train parameter
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_iter', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='BTAD')  # MVTAD, visa, BTAD

    # Model parameter
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--n_dim1', type=int, default=256)
    parser.add_argument('--n_dim2', type=int, default=512)
    parser.add_argument('--n_embedding1', type=int, default=128)
    parser.add_argument('--n_embedding2', type=int, default=128)
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=10.0)
    parser.add_argument('--latent_weight', type=float, default=0.1)
    parser.add_argument('--KLD_weight', type=float, default=0.1)
    args = parser.parse_args()
    args.path = f'./dataset/{args.dataset}/'
    print(args)
    dataset_list = get_dataset_list(args.dataset)
    train_main(args, dataset_list)

