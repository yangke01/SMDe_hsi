import os

import torch
import torch.nn as nn
import numpy as np

import hsir.data.transform.noise as N
import hsir.data.transform.general as G
import torchvision.transforms as T
import torch.utils.data as D
from hsir.data.dataset import HSITestDataset, HSITrainDataset
from hsir.data.utils import worker_init_fn
from hsir.trainer import Trainer
from hsir.scheduler import MultiStepSetLR, adjust_learning_rate, get_learning_rate

from pydoc import locate
from Hybrid import hybrid

# from models.HSIDCNN import hsid_cnn
# from models.QRNN3D import qrnn3d
# from models.T3SC import build_t3sc
# from models.TRQ3D import trq3d
# from models.SST import sst
# from models.SERT import sert_base
# from models.testnet import testnet
# from models.Hybrid import hybrid
# from models.SSUMamba import ssumamba
# from models.rethink import yknet
# from models.RGSST import kknet


if __name__ == '__main__' :
    """Hyperparameters"""
    batch_size = 1
    lr = 1e-3
    max_epochs = 100
    gpu_ids = '0'
    num_worker = 1

    train_root = '/media/yangke/689FC88261149B0B/yangke/data/ICVL128_31.db'
    # train_root = '/media/yangke/689FC88261149B0B/yangke/data/CAVE64_31.db'    
    # train_root = '/media/yangke/689FC88261149B0B/yangke/data/ICVL128_31.db'
    test_root = '/media/yangke/689FC88261149B0B/yangke/data/icvl_test'
    # test_root = '/media/yangke/689FC88261149B0B/yangke/data/cave_test'    
    checkpoint_path = 'checkpoint'
    task_name = ['gaussian', 'complex']

    """Set Up"""
    noise = task_name[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = newmixer()   ## model ##
    # name = 'NewMixer'    ## name of model ##
    model = hybrid()   ## model ##
    name = 'SMDe'    ## name of model ##
    use_chw = model.use_2dconv
    # schedule_path = 'hsir.schedule.denoise_mixer_complex'  ## shcedule ##
    # schedule_path = 'hsir.schedule.denoise_hybrid_complex'  ## shcedule ##
    # schedule_path = 'hsir.schedule.denoise_after'
    schedule_path = 'hsir.schedule.denoise_yknet_complex'
    schedule = locate(schedule_path)

    trainer = Trainer(
        net=model,
        lr=schedule.base_lr,
        save_dir=os.path.join(checkpoint_path, name),
        gpu_ids=gpu_ids,
    )

    # if os.path.exists():
    #     print(f"Loading checkpoint from {checkpoint_path}...")
    
    resume_path = os.path.join(checkpoint_path, name, 'model_latest.pth')
    # resume_path = os.path.join(checkpoint_path, name, 'mixture/model_best.pth')
    if os.path.exists(resume_path):
        trainer.load(resume_path)

    """Dataset"""
    if noise == 'gaussian':
        common_transform = G.Identity()
        train_transform_1 = T.Compose([
            N.AddNoise(50),
            G.HSI2Tensor(use_chw=use_chw)
        ])
        train_transform_2 = T.Compose([
            N.AddNoiseBlind([10, 30, 50, 70]),
            G.HSI2Tensor(use_chw=use_chw)
        ])
        target_transform = G.HSI2Tensor(use_chw=use_chw)
        train_dataset_1 = HSITrainDataset(train_root, train_transform_1, target_transform, common_transform)
        train_dataset_2 = HSITrainDataset(train_root, train_transform_2, target_transform, common_transform)
        train_loader1 = D.DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                          pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
        train_loader2 = D.DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                          pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)

    elif noise == 'complex':
        sigmas = [10, 30, 50, 70]
        common_transform = G.Identity()
        train_transform = T.Compose([
            N.AddNoiseNoniid(sigmas),
            G.SequentialSelect(
                transforms=[
                    lambda x: x,
                    N.AddNoiseImpulse(),
                    N.AddNoiseStripe(),
                    N.AddNoiseDeadline()
                ]
            ),
            G.HSI2Tensor(use_chw=use_chw)
        ])
        target_transform = G.HSI2Tensor(use_chw=use_chw)
        train_dataset = HSITrainDataset(train_root, train_transform, target_transform, common_transform)
        train_loader = D.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                     pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    else:
        raise ValueError("Invalid nois        # test_name = 'cave_512_test_after' e type. Choose 'gaussian' or 'complex'.")

    """Test Data"""
    if noise == 'gaussian':
        test_name = 'icvl_512_blind'
        test_root = os.path.join(test_root, test_name)
        dataset = HSITestDataset(test_root, size=50, use_chw=use_chw)
        test_loader = D.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    elif noise == 'complex':
        test_name = 'icvl_512_mixture'
        # test_name = 'cave_512_test_after'        
        test_root = os.path.join(test_root, test_name)
        dataset = HSITestDataset(test_root, size=50, use_chw=use_chw)
        test_loader = D.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    """Training and Testing"""
    if os.path.exists(resume_path):
        pass
    else:
        if lr:
            adjust_learning_rate(trainer.optimizer, lr)  # override lr
    lr_scheduler = MultiStepSetLR(trainer.optimizer, schedule.lr_schedule, epoch=trainer.epoch)
    epoch_per_save = 5
    best_psnr = 0
    while trainer.epoch < schedule.max_epochs:
        np.random.seed()  # reset seed per epoch, otherwise the noise will be added with a specific pattern
        trainer.logger.print('Epoch [{}] Use lr={}'.format(trainer.epoch, get_learning_rate(trainer.optimizer)))

        # train
        if noise == 'gaussian':
            if trainer.epoch == 30:
                best_psnr = 0
            if trainer.epoch < 30:
                trainer.train(train_loader1)
            else:
                trainer.train(train_loader2, warm_up=trainer.epoch == 30)
        elif noise == 'complex':
            trainer.train(train_loader, warm_up=False)
            # trainer.train(train_loader, warm_up=trainer.epoch == 80)

        # save ckpt
        metrics = trainer.validate(test_loader, test_name)
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            print(f"\033[1;32mBest Result Saving...\033[0m")
            trainer.save_checkpoint('model_best.pth')
        # if trainer.epoch % epoch_per_save == 0:
        #     trainer.save_checkpoint()
        trainer.save_checkpoint('model_latest.pth')

        lr_scheduler.step()


