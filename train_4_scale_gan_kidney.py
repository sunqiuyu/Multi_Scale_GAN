import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet_training import weights_init
from utils.callbacks import LossHistory
from nets.Mutil_Scale_GAN import GeneratorUNet_4scale,DiscriminatorUNet
from utils.dataloader_single import UnetDataset, unet_dataset_collate
from utils.fit_three_diffv1 import fit_one_epoch


if __name__ == "__main__":
    Cuda = True
    num_classes = 2
    input_shape = [128, 128]
    Init_Epoch = 0
    MAX_Epoch = 200
    batch_size = 16
    init_lr = 1e-4
    VOCdevkit_path = 'Kits_Datasets'
    dice_loss = True
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 6

    modelG = GeneratorUNet_4scale(num_classes=num_classes).train()
    modelD1 = DiscriminatorUNet(num_classes=num_classes).train()
    modelD2 = DiscriminatorUNet(num_classes=num_classes).train()
    modelD3 = DiscriminatorUNet(num_classes=num_classes).train()
    modelD4 = DiscriminatorUNet(num_classes=num_classes).train()
    weights_init(modelG)
    weights_init(modelD1)
    weights_init(modelD2)
    weights_init(modelD3)
    weights_init(modelD4)

    modelG_train = modelG.train()
    modelD1_train = modelD1.train()
    modelD2_train = modelD2.train()
    modelD3_train = modelD3.train()
    modelD4_train = modelD4.train()
    if Cuda:
        modelG_train = torch.nn.DataParallel(modelG)
        cudnn.benchmark = True
        modelG_train = modelG_train.cuda()

        modelD1_train = torch.nn.DataParallel(modelD1)
        cudnn.benchmark = True
        modelD1_train = modelD1_train.cuda()

        odelD2_train = torch.nn.DataParallel(modelD2)
        cudnn.benchmark = True
        modelD2_train = modelD2_train.cuda()

        odelD3_train = torch.nn.DataParallel(modelD3)
        cudnn.benchmark = True
        modelD3_train = modelD3_train.cuda()

        odelD4_train = torch.nn.DataParallel(modelD4)
        cudnn.benchmark = True
        modelD4_train = modelD4_train.cuda()

    loss_history = LossHistory("Kits_logs/")
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    if True:
        batch_size = batch_size
        lr = init_lr
        start_epoch = Init_Epoch
        end_epoch = MAX_Epoch

        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizerG = optim.Adam(modelG_train.parameters(), lr)
        lr_schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=1, gamma=0.96)

        optimizerD1 = optim.Adam(modelD1_train.parameters(), lr)
        lr_schedulerD1 = optim.lr_scheduler.StepLR(optimizerD1, step_size=1, gamma=0.96)

        optimizerD2 = optim.Adam(modelD2_train.parameters(), lr)
        lr_schedulerD2 = optim.lr_scheduler.StepLR(optimizerD2, step_size=1, gamma=0.96)

        optimizerD3 = optim.Adam(modelD3_train.parameters(), lr)
        lr_schedulerD3 = optim.lr_scheduler.StepLR(optimizerD3, step_size=1, gamma=0.96)

        optimizerD4 = optim.Adam(modelD4_train.parameters(), lr)
        lr_schedulerD4 = optim.lr_scheduler.StepLR(optimizerD4, step_size=1, gamma=0.96)

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=False, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=False, collate_fn=unet_dataset_collate)

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(modelG_train, modelG,
                          modelD1_train, modelD1,
                          modelD2_train, modelD2,
                          modelD3_train, modelD3,
                          modelD4_train, modelD4,
                          loss_history, optimizerG,
                          optimizerD1, optimizerD2,optimizerD3,optimizerD4,
                          epoch,epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights,
                          num_classes)
            lr_schedulerG.step()
            lr_schedulerD1.step()
            lr_schedulerD2.step()
            lr_schedulerD3.step()
            lr_schedulerD4.step()
