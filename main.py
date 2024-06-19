# -*- coding: utf-8 -*-
# @Date : 2022-07-07
# @Author : zyz
# @File : train
import random
from torch import optim
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm
from model_ResNet18 import CVAE
from model_compare import ProjectNet, CompareModel, PCA_whitten
from scheduler import *
from ssim_loss import *
from dataset import *

from loss import *
import argparse
import os.path
import os, glob, csv
from aupro import *
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from imgs2video import imgs2video
import pickle as pkl


seed = 3407
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def Fusion_Para_Est(loader, construct_model, project_model, pca_model, compare_model, device):
    construct_model.eval()
    project_model.eval()
    with torch.no_grad():
        for i, (img, y, m) in enumerate(tqdm(loader)):
            img = img.to(device)
            z, _, _ = construct_model(img)
            f0 = project_model(img)
            f0 = pca_model(f0)
            f1 = project_model(z)
            f1 = pca_model(f1)
            l = 0
            for x0, x1 in zip(f0, f1):
                d = x0 - x1
                compare_model.CNN_M_distance_para_fit(d, l, i)
                l += 1

    D_list = [list() for l in range(project_model.n_layer)]
    C_list = [list() for l in range(compare_model.n_blur + 1)]
    S_list = [list() for l in range(compare_model.n_blur + 1)]
    with torch.no_grad():
        for i, (img, y, m) in enumerate(tqdm(loader)):
            img = img.to(device)
            z, _, _ = construct_model(img)
            f0 = project_model(img)
            f0 = pca_model(f0)
            f1 = project_model(z)
            f1 = pca_model(f1)
            C_list[0].append(compare_model.Color_distance(img, z))
            S_list[0].append(compare_model.ssim_distance(img, z))
            x0 = compare_model.blur(img)
            x1 = compare_model.blur(z)
            for i in range(compare_model.n_blur-1):
                C_list[i+1].append(compare_model.Color_distance(x0[i].unsqueeze(0), x1[i].unsqueeze(0)).to(device))
                S_list[i+1].append(compare_model.ssim_distance(x0[i].unsqueeze(0), x1[i].unsqueeze(0)).to(device))
            l = 0
            for x0, x1 in zip(f0, f1):
                _, c, h, w = x0.size()
                d = compare_model.M_distance(x0, x1, l)
                D_list[l].append(d)
                l += 1
        compare_model.Layer_Normalization_para_fit(D_list, C_list, S_list)
        del D_list, C_list, S_list

    D_CNN = []
    D_COLOR = []
    D_SSIM = []
    with torch.no_grad():
        for i, (img, y, m) in enumerate(tqdm(loader)):
            img = img.to(device)
            z, _, _ = construct_model(img)
            f0 = project_model(img)
            f0 = pca_model(f0)
            f1 = project_model(z)
            f1 = pca_model(f1)
            d_cnn, d_color, d_ssim = compare_model(img, z, f0, f1)

            D_CNN.append(d_cnn.cpu())
            D_COLOR.append(d_color.cpu())
            D_SSIM.append(d_ssim.cpu())
        D_CNN_arrary = torch.cat(D_CNN, 0)
        D_COLOR_arrary = torch.cat(D_COLOR, 0)
        D_SSIM_arrary = torch.cat(D_SSIM, 0)
        del D_CNN, D_SSIM, D_COLOR
        compare_model.Normalization_para_fit(D_CNN_arrary, D_COLOR_arrary, D_SSIM_arrary)

    Seg = []
    with torch.no_grad():
        for i, (img, y, m) in enumerate(tqdm(loader)):
            img = img.to(device)
            z, _, _ = construct_model(img)

            f0 = project_model(img)
            f0 = pca_model(f0)
            f1 = project_model(z)
            f1 = pca_model(f1)
            result = compare_model(img, z, f0, f1, flag=True, ablation=True)
            Seg.append(result['cnn_ssim'])
        D_Seg_arrary = torch.cat(Seg, 0).reshape([1, -1])
        del Seg
        compare_model.th = [D_Seg_arrary.mean() + D_Seg_arrary.std(),
                            D_Seg_arrary.mean() + 15 * D_Seg_arrary.std()]
    return


def Project_Whitten_Est(loader, pca_model, project_model, device):
    project_model.eval()
    with torch.no_grad():
        for i, (img, y, m) in enumerate(tqdm(loader)):
            img = img.to(device)
            f = project_model(img)
            pca_model.p_fit(f)
        pca_model.fit()
    return


def train(epoch, Train_loader, Eva_loader, Test_loader, model, project_model, optimizer, scheduler, device,
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
        normalize=True,
        range=(0, 1),
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


def save_img_heatmap(img, anomaly_socre, t, filename):
    anomaly_socre = anomaly_socre.cpu().numpy()[0, 0, :, :]
    anomaly_socre[anomaly_socre > t[1].cpu().numpy()] = t[1].cpu().numpy()
    anomaly_socre[anomaly_socre < t[0].cpu().numpy()] = t[0].cpu().numpy()

    A2max = t[1].cpu().numpy()
    A2min = t[0].cpu().numpy()

    anomaly_socre = np.round(255 * (anomaly_socre - A2min) / (A2max - A2min))
    anomaly_socre = anomaly_socre.astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_socre, colormap=cv2.COLORMAP_JET)

    img = img.permute(0, 2, 3, 1).squeeze(0)
    img = img.cpu().numpy()
    img = img[:, :, ::-1]

    img = img * 200
    img = img.astype(np.uint8)

    overlay = img.copy()
    img = cv2.addWeighted(heatmap, 0.3, overlay, 0.85, 0, overlay)
    cv2.imwrite(filename, img)
    return


def test(loader, construct_model, project_model, compare_model, pca_model, device, pro_flag, dataset_name, t):
    label_true_list = []
    mask_true_list = []
    label_pred_list_std = []
    label_pred_list_max = []
    mask_pred_list = []
    mask_cnn_list = []
    mask_color_list = []
    mask_ssim_list = []
    mask_cns_list = []
    mask_cnc_list = []
    mask_cs_list = []
    if not os.path.exists(f'sample/{dataset_name}'):
        os.makedirs(f'sample/{dataset_name}')
    if not os.path.exists(f'sample/{dataset_name}/original/'):
        os.makedirs(f'sample/{dataset_name}/original/')
    if not os.path.exists(f'sample/{dataset_name}/reconstruction/'):
        os.makedirs(f'sample/{dataset_name}/reconstruction/')
    if not os.path.exists(f'sample/{dataset_name}/segmentation/'):
        os.makedirs(f'sample/{dataset_name}/segmentation/')
    if not os.path.exists(f'sample/{dataset_name}/ground_truth/'):
        os.makedirs(f'sample/{dataset_name}/ground_truth/')
    if not os.path.exists(f'sample/{dataset_name}/result/'):
        os.makedirs(f'sample/{dataset_name}/result/')
    import time
    total_time = 0
    with torch.no_grad():
        for i, (img, y, m) in enumerate(tqdm(loader)):
            img = img.to(device)
            torch.cuda.synchronize()
            start = time.time()
            z, _, _ = construct_model(img)
            f = project_model(img)
            f = pca_model(f)
            f1 = project_model(z)
            f1 = pca_model(f1)
            result = compare_model(img, z, f, f1, flag=True, ablation=True)
            torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)

            mask_pred_list.append(result['seg'])
            mask_cnn_list.append(result['cnn'])
            mask_color_list.append(result['color'])
            mask_ssim_list.append(result['ssim'])
            mask_cnc_list.append(result['cnn_color'])
            mask_cns_list.append(result['cnn_ssim'])
            mask_cs_list.append(result['color_ssim'])

            label_true_list.append(y)
            mask_true_list.append(m)
            label_pred_list_std.append(result['det_std'])
            label_pred_list_max.append(result['det_max'])

            if 1:

                # utils.save_image(
                #     img,
                #     f'sample/{dataset_name}/original/{dataset_name}_{str(i).zfill(5)}.png',
                #     normalize=True,
                #     value_range=(0, 1))
                #
                # utils.save_image(
                #     z,
                #     f'sample/{dataset_name}/reconstruction/{dataset_name}_{str(i).zfill(5)}.png',
                #     normalize=True,
                #     value_range=(0, 1))
                #
                # utils.save_image(
                #     m,
                #     f'sample/{dataset_name}/ground_truth/{dataset_name}_{str(i).zfill(5)}.png',
                #     normalize=True,
                #     value_range=(0, 1))

                save_img_heatmap(img, result['seg'], t,
                                 f'sample/{dataset_name}/segmentation/{dataset_name}_{str(i).zfill(5)}.png')
                # save_img_heatmap(img, result['cnn'], t,
                #                  f'sample/{dataset_name}/segmentation/{dataset_name}_{str(i).zfill(5)}_cnn.png')
                # save_img_heatmap(img, result['cnn_color'], t,
                #                  f'sample/{dataset_name}/segmentation/{dataset_name}_{str(i).zfill(5)}_cnn_color.png')
                # save_img_heatmap(img, result['cnn_ssim'], t,
                #                  f'sample/{dataset_name}/segmentation/{dataset_name}_{str(i).zfill(5)}_cnn_ssim.png')
    print(f'total_time:{total_time},  fps:{len(loader) / total_time}')
    label_true = torch.cat(label_true_list, 0).numpy()
    label_pred_std = torch.tensor(label_pred_list_std).numpy()
    label_pred_max = torch.tensor(label_pred_list_max).numpy()
    mask_true = torch.cat(mask_true_list, 0).numpy().astype(bool)
    mask_pred = torch.cat(mask_pred_list, 0).cpu().numpy()
    det_score_std = roc_auc_score(label_true, label_pred_std)
    print(det_score_std)
    det_score_max = roc_auc_score(label_true, label_pred_max)
    print(det_score_max)
    seg_score = roc_auc_score(mask_true.flatten(), mask_pred.flatten())
    print(seg_score)
    pro = 0
    if pro_flag:
        pro = auc_pro(mask_pred, mask_true)
        print(pro)

    mask_pred = torch.cat(mask_cnn_list, 0).cpu().numpy()
    cnn_score = roc_auc_score(mask_true.flatten(), mask_pred.flatten())
    mask_pred = torch.cat(mask_color_list, 0).cpu().numpy()
    color_score = roc_auc_score(mask_true.flatten(), mask_pred.flatten())
    mask_pred = torch.cat(mask_ssim_list, 0).cpu().numpy()
    ssim_score = roc_auc_score(mask_true.flatten(), mask_pred.flatten())
    mask_pred = torch.cat(mask_cnc_list, 0).cpu().numpy()
    cnc_score = roc_auc_score(mask_true.flatten(), mask_pred.flatten())
    mask_pred = torch.cat(mask_cns_list, 0).cpu().numpy()
    cns_score = roc_auc_score(mask_true.flatten(), mask_pred.flatten())
    mask_pred = torch.cat(mask_cs_list, 0).cpu().numpy()
    cs_score = roc_auc_score(mask_true.flatten(), mask_pred.flatten())

    result = {'det_std': det_score_std * 100,
              'det_max': det_score_max * 100,
              'seg': seg_score * 100,
              'seg_cnn': cnn_score * 100,
              'seg_color': color_score * 100,
              'seg_ssim': ssim_score * 100,
              'seg_color_cnn': cnc_score * 100,
              'seg_cnn_ssim': cns_score * 100,
              'seg_color_ssim': cs_score * 100,
              'pro': pro * 100}
    return result


def train_main(args, dataset_list):
    for dataset_name in dataset_list:
        if not os.path.exists(f'checkpoint/{dataset_name}'):
            os.makedirs(f'checkpoint/{dataset_name}')
        if not os.path.exists(f'sample/{dataset_name}'):
            os.makedirs(f'sample/{dataset_name}')

        Train_dataset = MVTecAD(f'./dataset/MVTAD/', dataset_name, args.size, 'train')
        Train_loader = DataLoader(Train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        Eva_dataset = MVTecAD(f'./dataset/MVTAD/', dataset_name, args.size, 'validation')
        Eva_loader = DataLoader(Eva_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        Test_dataset = MVTecAD(f'./dataset/MVTAD/', dataset_name, args.size, 'test')
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
            score = train(i, Train_loader, Eva_loader, Test_loader,
                          construct_model, project_model, optimizer, scheduler, args.device,
                          dataset_name)
            print(f'{i}: {score}')
        torch.save(construct_model.module.state_dict(), f'checkpoint/{dataset_name}/Cvae.pt')
        imgs2video(dataset_name)
    return


def test_main(args, dataset, datasetList):
    N = 1 # test times
    headers = ['name', 'index', 'det_std', 'det_max', 'seg', 'seg_cnn', 'seg_color', 'seg_ssim', 'seg_color_cnn',
               'seg_cnn_ssim', 'seg_color_ssim', 'pro']
    if N == 1:
        seed = 123
    else:
        seed = 0

    mode = 2

    with open(f"./full_result_{args.pca_ncom}.csv", 'w', newline='') as f:
        csv_writer = csv.DictWriter(f, headers)
        csv_writer.writeheader()
        for dataset_name in datasetList:
            path = f'./checkpoint/{dataset}/{dataset_name}'
            # load reconstruction model
            weight = f'{path}/Cvae.pt'
            print(dataset, dataset_name)
            Score_Mat = np.zeros((N, len(headers) - 2), dtype=float)
            construct_model = CVAE(n_dim1=args.n_dim1, n_dim2=args.n_dim2,
                                   n_embedding1=args.n_embedding1,
                                   n_embedding2=args.n_embedding2, size=args.size, seed=seed).to(args.device)
            construct_model.load_state_dict(torch.load(weight))
            construct_model.eval()

            # load project model
            project_model = ProjectNet(mode).to(args.device)
            project_model.eval()

            # load PCA model
            if os.path.exists(f'{path}/pca_model_{args.pca_ncom}.pkl'):
                with open(f'{path}/pca_model_{args.pca_ncom}.pkl', "rb") as f:
                    pca_model = pkl.load(f)
            else:
                pca_model = PCA_whitten(args.pca_ncom, project_model.n_layer)
                # Evaset = MVTecAD(args.path, dataset_name, args.size, 'validation')
                Evaset = VISA(args.path, dataset_name, args.size, 'validation')
                # Evaset = MVTecAD_3D(args.path, dataset_name, args.size, 'validation')
                # Evaset = BTAD(args.path, dataset_name, args.size, 'Eva')
                loader = DataLoader(Evaset, batch_size=1, num_workers=1)
                Project_Whitten_Est(loader, pca_model, project_model, args.device)
                with open(f'{path}/pca_model_{args.pca_ncom}.pkl', "wb") as f:
                    pkl.dump(pca_model, f)

            # load compare model
            if os.path.exists(f'{path}/compare_model_{args.pca_ncom}.pkl'):
                with open(f'{path}/compare_model_{args.pca_ncom}.pkl', "rb") as f:
                    compare_model = pkl.load(f)
            else:
                compare_model = CompareModel(args.size, project_model.n_layer, args.device)
                # Evaset = MVTecAD(args.path, dataset_name, args.size, 'validation')
                # Evaset = BTAD(args.path, dataset_name, args.size, 'Eva')
                Evaset = VISA(args.path, dataset_name, args.size, 'validation')
                loader = DataLoader(Evaset, batch_size=1, num_workers=1)
                Fusion_Para_Est(loader, construct_model, project_model, pca_model, compare_model, args.device)
                with open(f'{path}/compare_model_{args.pca_ncom}.pkl', "wb") as f:
                    pkl.dump(compare_model, f)

            # test N times
            for l in range(N):
                # testset = MVTecAD(args.path, dataset_name, args.size, 'test')
                # testset = MVTecAD_3D(args.path, dataset_name, args.size, 'test')
                # testset = BTAD(args.path, dataset_name, args.size, 'Test')
                testset = VISA(args.path, dataset_name, args.size, 'test')
                loader = DataLoader(testset, batch_size=1, num_workers=1)

                score = test(loader, construct_model, project_model, compare_model, pca_model,
                             args.device, args.proplot, dataset_name, compare_model.th)
                score['name'] = dataset_name
                score['index'] = dataset
                csv_writer.writerow(score)

                Score_Mat[l:] = np.array([score['det_std'], score['det_max'], score['seg'], score['seg_cnn'],
                                          score['seg_color'], score['seg_ssim'], score['seg_color_cnn'],
                                          score['seg_cnn_ssim'],
                                          score['seg_color_ssim'], score['pro']], dtype=float)
            if N > 1:
                Score_mean = np.mean(Score_Mat, 0)
                Score_std = np.std(Score_Mat, 0)
                score_mean_dict = {'det_std': Score_mean[0],
                                   'det_max': Score_mean[1],
                                   'seg': Score_mean[2],
                                   'seg_cnn': Score_mean[3],
                                   'seg_color': Score_mean[4],
                                   'seg_ssim': Score_mean[5],
                                   'seg_color_cnn': Score_mean[6],
                                   'seg_cnn_ssim': Score_mean[7],
                                   'seg_color_ssim': Score_mean[8],
                                   'pro': Score_mean[9],
                                   'name': dataset_name + '_mean',
                                   'index': dataset}
                score_std_dict = {'det_std': Score_std[0],
                                  'det_max': Score_std[1],
                                  'seg': Score_std[2],
                                  'seg_cnn': Score_std[3],
                                  'seg_color': Score_std[4],
                                  'seg_ssim': Score_std[5],
                                  'seg_color_cnn': Score_std[6],
                                  'seg_cnn_ssim': Score_std[7],
                                  'seg_color_ssim': Score_std[8],
                                  'pro': Score_std[9],
                                  'name': dataset_name + '_std',
                                  'index': dataset
                                  }

                csv_writer.writerow(score_mean_dict)
                csv_writer.writerow(score_std_dict)
    return


def calc_complex():
    from ptflops import get_model_complexity_info
    from thop import profile
    project_model = ProjectNet(mode=2).to(args.device)
    # compare_model = CompareModel(args.size, project_model.n_layer, args.device).to(args.device)
    # if os.path.exists(f'./checkpoint/bottle/compare_model_96.pkl'):
    #     with open(f'./checkpoint//bottle/compare_model_96.pkl', "rb") as f:
    #         compare_model = pkl.load(f)
    construct_model = CVAE(n_dim1=args.n_dim1, n_dim2=args.n_dim2,
                           n_embedding1=args.n_embedding1,
                           n_embedding2=args.n_embedding2, size=args.size, seed=seed).to(args.device)
    pca_model = PCA_whitten(args.pca_ncom, project_model.n_layer).to(args.device)
    # construct_model.load_state_dict(torch.load(weight))
    flops, params = get_model_complexity_info(construct_model, (3, args.size, args.size), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    # stat(construct_model, (3, 256, 256))
    input = torch.randn(1, 3, args.size, args.size).to(args.device)
    flops, params = profile(construct_model, inputs=(input,))
    print(flops / 1024 / 1024 / 1024)
    print('Params: ', params / 1024 / 1024)
    flops, params = profile(project_model, inputs=(input,))
    print('Flops:  ', flops / 1024 / 1024 / 1024)
    print('Params: ', params / 1024 / 1024)
    f = project_model(input)
    pca_model.fit(f, 1)
    # f = pca_model(f)

    flops, params = profile(pca_model, inputs=(f, ))
    print('Flops:  ', flops)
    print('Params: ', params)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_iter', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_dim1', type=int, default=256)
    parser.add_argument('--n_dim2', type=int, default=512)
    parser.add_argument('--n_embedding1', type=int, default=128)
    parser.add_argument('--n_embedding2', type=int, default=128)
    parser.add_argument('--recon_weight', type=float, default=10.0)
    parser.add_argument('--latent_weight', type=float, default=0.1)
    parser.add_argument('--KLD_weight', type=float, default=0.1)
    parser.add_argument('--pca_ncom', type=float, default=96)
    parser.add_argument('--proplot', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--path', type=str, default='./dataset/visa/')
    # parser.add_argument('--path', type=str, default='./dataset/MVTAD/')
    # parser.add_argument('--path', type=str, default='./dataset/BTAD/')
    args = parser.parse_args()
    print(args)
    # train_main(args, text_list+object_list)
    test_main(args, f'{args.size}/VISA', visa_list)
    # calc_complex()
