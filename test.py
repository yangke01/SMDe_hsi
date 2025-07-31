import os
import torch
import torch.nn as nn

import torch.utils.data as D
from hsir.data.dataset import HSITestDataset, HSITrainDataset

from tqdm import tqdm
from hsir import metrics
import time

import pandas as pd
from scipy.io import savemat

from models.HSIDCNN import hsid_cnn
from models.QRNN3D import qrnn3d
from models.T3SC import build_t3sc
from models.TRQ3D import trq3d
from models.SST import sst
from models.SERT import sert_base
from models.testnet import testnet
from models.Hybrid import hybrid
from models.MAN import man_m
from models.HSDT import hsdt, hsdt_l


class IdentityModel(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, x):
        return x
    
def noisy():
    net = IdentityModel()
    net.use_2dconv=True
    return net


def save_mat(path, array, key='cube'):
    savemat(path, {key: array})


def MSIQA(X, Y):
    psnr = metrics.mpsnr(X, Y)
    ssim = metrics.mssim(X, Y)
    sam = metrics.sam(X, Y)
    return psnr, ssim, sam


def test(model, dataloader, device, save=False, save_list=None, save_dir="./denoised_results"):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_sam = 0.0

    all_metrics = []

    if save:
        os.makedirs(save_dir, exist_ok=True)

    if save_list is None:
        save_list = []

    pbar = tqdm(dataloader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            filename = batch.get('filename', [f"sample_{batch_idx}"])[0]

            outputs = model(inputs)

            if len(outputs.shape) == 5:
                outputs = outputs.squeeze(1)
                targets = targets.squeeze(1)

            psnr, ssim, sam = MSIQA(outputs, targets)
            total_psnr += psnr
            total_ssim += ssim
            total_sam += sam

            avg_psnr = total_psnr / (batch_idx + 1)
            avg_ssim = total_ssim / (batch_idx + 1)
            avg_sam = total_sam / (batch_idx + 1)

            pbar.set_postfix({
                "PSNR": f"{avg_psnr:.4f}",
                "SSIM": f"{avg_ssim:.4f}",
                "SAM": f"{avg_sam:.4f}"
            })

            if save:
                filename_only = os.path.basename(filename)
                name_without_ext = os.path.splitext(filename_only)[0]
                if name_without_ext in save_list:
                    output_np = outputs[0].detach().cpu().permute(1, 2, 0).numpy()
                    save_path = os.path.join(save_dir, f"{name_without_ext}_denoised.mat")
                    save_mat(save_path, output_np, key='output')

            all_metrics.append({
                "filename": filename,
                "psnr": psnr,
                "ssim": ssim,
                "sam": sam
            })

    print(f"Test Result => PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, SAM: {avg_sam:.4f}")
    return all_metrics


if __name__ == '__main__' :
    """Hyperparameters"""
    # batch_size = 8
    # lr = 1e-4
    # max_epochs = 100
    # gpu_ids = '0'
    # num_worker = 8

    # train_root = '/media/yangke/689FC88261149B0B/yangke/data/ICVL128_31.db'
    # test_root = '/media/yangke/689FC88261149B0B/yangke/code/SST/data/HSI_Data/icvl_noise_50'

    # test_root = '/media/yangke/689FC88261149B0B/yangke/data/icvl_test'
    test_root = '/media/yangke/689FC88261149B0B/yangke/data/cave_test'
    # test_root = '/media/yangke/689FC88261149B0B/yangke/data/urban'
    checkpoint_path = 'checkpoint'
    task_name = ['gaussian', 'complex']

    """Set Up"""
    noise = task_name[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    model = sert_base()   ## model ##
    name = 'SERTv2'    ## name of model ##

    model.to(device)
    use_chw = model.use_2dconv

    print("=> creating model ..")
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.4fM" % (total / 1e6))

    resume_path = os.path.join(checkpoint_path, name, )
    if noise == 'gaussian':
        resume_path = os.path.join(resume_path, 'gaussian')
    elif noise == 'complex':
        resume_path = os.path.join(resume_path, 'mixture')
    resume_path = os.path.join(resume_path, 'model_best.pth')
    checkpoint = torch.load(resume_path,  map_location=device)

    for key in checkpoint.keys():
        print(key)

    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    """Test Data"""
    num_size = 51
    # complex_test = ['icvl_512_10', 'icvl_512_30', 'icvl_512_50', 'icvl_512_70', 'icvl_512_blind']
    # complex_test = ['icvl_512_noniid', 'icvl_512_stripe', 'icvl_512_deadline', 'icvl_512_impulse', 'icvl_512_mixture']
    complex_test = ['cave_512_noniid', 'cave_512_stripe', 'cave_512_deadline', 'cave_512_impulse', 'cave_512_mixture', 'cave_512_test_after']
    # complex_test = ['urban_304']
    if noise == 'gaussian':
        test_name = 'icvl_512_blind'
        test_root = os.path.join(test_root, test_name)
        dataset = HSITestDataset(test_root, size=num_size, use_chw=use_chw, return_name=True)
        test_loader = D.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    elif noise == 'complex':
        test_name = complex_test[5]
        test_root = os.path.join(test_root, test_name)
        dataset = HSITestDataset(test_root, size=num_size, use_chw=use_chw, return_name=True)
        test_loader = D.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # model=noisy()
    # model.to(device)

    save_list = ['tree_0822-0853', 'balloons_ms', 'Urban_93_123', 'Urban_176_206', 'objects_0924-1607', 'chart_and_stuffed_toy_ms']
    start_time = time.time()
    metrics_list = test(model, test_loader, device, save=True, save_list=save_list,)
    end_time = time.time()
    test_time = end_time - start_time
    print('cost-time:', (test_time / float(num_size)))

    # # 保存结果到 CSV
    # df = pd.DataFrame(metrics_list)
    # csv_path = f"results_{name}_{noise}.csv"
    # df.to_csv(csv_path, index=False)
    # print(f"Saved metrics to {csv_path}")