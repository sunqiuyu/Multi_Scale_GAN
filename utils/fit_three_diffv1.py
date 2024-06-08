import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

from losses import SoftDiceLoss
from utils.utils import get_lr
from utils.utils_metrics import f_score
import torch.nn as nn

c_loss = nn.BCELoss().cuda()
d_loss = SoftDiceLoss().cuda()
cro_loss = nn.CrossEntropyLoss().cuda()
def fit_one_epoch(modelG_train, modelG,
                  modelD1_train, modelD1,
                  modelD2_train, modelD2,
                  modelD3_train, modelD3,
                  modelD4_train, modelD4,
                  loss_history,
                  optimizerG,optimizerD1,
                  optimizerD2,
                  optimizerD3,
                  optimizerD4,epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, dice_loss, focal_loss, cls_weights, num_classes):
    total_loss = 0
    total_f_score = 0

    train_loss1 = 0
    tarin_f_score1 = 0

    val_loss = 0
    val_f_score = 0

    modelG_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs,mask,one_hot_mask = batch

            with torch.no_grad():
                imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
                mask = torch.from_numpy(np.array(mask)).long()
                one_hot_mask = torch.from_numpy(np.array(one_hot_mask)).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    mask = mask.cuda()
                    one_hot_mask = one_hot_mask.cuda()
                    weights = weights.cuda()
            ##########################
            # DISCRIMINATOR TRAINING #
            ##########################
            optimizerD1.zero_grad()
            optimizerD2.zero_grad()
            optimizerD3.zero_grad()
            optimizerD4.zero_grad()
            cpmap,layer1,layer2,layer3 = modelG_train(imgs)  # batch_size,2,128,128
            #cpmap = nn.Softmax2d()(cpmap)

            N = cpmap.size()[0]
            H = cpmap.size()[2]
            W = cpmap.size()[3]

            cpmap1 = nn.functional.interpolate(layer1,(H,W),mode='bilinear')
            cpmap2 = nn.functional.interpolate(layer2, (H, W), mode='bilinear')
            cpmap3 = nn.functional.interpolate(layer3, (H, W), mode='bilinear')

            # Generate the Real and Fake Labels
            targetf = Variable(torch.zeros([N,1,H, W], dtype=torch.float32), requires_grad=False)
            targetr = Variable(torch.ones([N,1,H, W], dtype=torch.float32), requires_grad=False)
            if cuda:
                targetf = targetf.cuda()
                targetr = targetr.cuda()
            # Train on Real
            confr = modelD1_train(imgs, one_hot(mask,num_classes))  ##batch_size,1,128,128
            LDr = c_loss(confr, targetr)
            # Train on Fake
            conff = modelD1_train(imgs, cpmap)
            LDf = c_loss(conff, targetf)
            D_train_loss = (LDf + LDr) * 0.5
            D_train_loss = D_train_loss.mean()

            # Train on Real
            confr2 = modelD2_train(imgs, one_hot(mask, num_classes))  ##batch_size,1,128,128
            LDr2 = c_loss(confr2, targetr)
            # Train on Fake
            conff2 = modelD2_train(imgs, cpmap1)
            LDf2 = c_loss(conff2, targetf)
            D_train_loss2 = (LDf2 + LDr2) * 0.5
            D_train_loss2 = D_train_loss2.mean()

            # Train on Real
            confr3 = modelD3_train(imgs, one_hot(mask, num_classes))  ##batch_size,1,128,128
            LDr3 = c_loss(confr3, targetr)
            # Train on Fake
            conff3 = modelD3_train(imgs, cpmap2)
            LDf3 = c_loss(conff3, targetf)
            D_train_loss3 = (LDf3 + LDr3) * 0.5
            D_train_loss3 = D_train_loss3.mean()

            # Train on Real
            confr4 = modelD4_train(imgs, one_hot(mask, num_classes))  ##batch_size,1,128,128
            LDr4 = c_loss(confr4, targetr)
            # Train on Fake
            conff4 = modelD4_train(imgs, cpmap3)
            LDf4 = c_loss(conff4, targetf)
            D_train_loss4 = (LDf4 + LDr4) * 0.5
            D_train_loss4 = D_train_loss4.mean()



            D_train_loss = D_train_loss + D_train_loss3 + D_train_loss2 + D_train_loss4
            D_train_loss.backward(retain_graph=True)
            # D_train_loss4.backward(retain_graph=True)
            # D_train_loss3.backward(retain_graph=True)
            # D_train_loss2.backward(retain_graph=True)
            # D_train_loss.backward(retain_graph=True)
            optimizerD1.step()
            optimizerD2.step()
            optimizerD3.step()
            optimizerD4.step()

            ######################
            # GENERATOR TRAINING #
            #####################
            optimizerG.zero_grad()
            # optimizerD.zero_grad()
            cmap,layerg1,layerg2,layerg3 = modelG_train(imgs)
            cpmapg1 = nn.functional.interpolate(layerg1, (H, W), mode='bilinear')
            cpmapg2 = nn.functional.interpolate(layerg2, (H, W), mode='bilinear')
            cpmapg3 = nn.functional.interpolate(layerg3, (H, W), mode='bilinear')

            cpmapsmax = nn.Softmax2d()(cmap)
            cpmapsmax1 = nn.Softmax2d()(cpmapg1)
            cpmapsmax2 = nn.Softmax2d()(cpmapg2)
            cpmapsmax3 = nn.Softmax2d()(cpmapg3)

            #cpmapsig = torch.sigmoid(cmap)
            conff = modelD1_train(imgs, cpmapsmax)
            conff1 = modelD2_train(imgs, cpmapsmax1)
            conff2 = modelD3_train(imgs, cpmapsmax2)
            conff3 = modelD3_train(imgs, cpmapsmax3)

            LGadv1 = c_loss(conff, targetr)
            LGadv2 = c_loss(conff1, targetr)
            LGadv3 = c_loss(conff2, targetr)
            LGadv4 = c_loss(conff3, targetr)

            loss_dice1 = Dice_loss(cmap, one_hot_mask)
            loss_fce1 = cro_loss(cmap, mask) * ((1 - conff) ** 4 + abs(1 - conff))
            loss_seg1 = loss_dice1 + loss_fce1

            loss_dice2 = Dice_loss(cpmapg1, one_hot_mask)
            loss_fce2 = cro_loss(cpmapg1, mask)* ((1 - conff1) ** 4 + abs(1 - conff1))
            loss_seg2 = loss_dice2 + loss_fce2

            loss_dice3 = Dice_loss(cpmap2, one_hot_mask)
            loss_fce3 = cro_loss(cpmap2, mask)* ((1 - conff2) ** 4 + abs(1 - conff2))
            loss_seg3 = loss_dice3 + loss_fce3

            loss_dice4 = Dice_loss(cpmap3, one_hot_mask)
            loss_fce4 = cro_loss(cpmap3, mask)* ((1 - conff3) ** 4 + abs(1 - conff3))
            loss_seg4 = loss_dice4 + loss_fce4

            lossseg1 = loss_seg1.mean()
            lossseg2 = loss_seg2.mean()
            lossseg3 = loss_seg3.mean()
            lossseg4 = loss_seg4.mean()

            #loss_seg = lossseg1 + lossseg2 + lossseg3 + lossseg4

            G_adv_loss1 = LGadv1.mean()
            G_adv_loss2 = LGadv2.mean()
            G_adv_loss3 = LGadv3.mean()
            G_adv_loss4 = LGadv4.mean()

            # G_adv_loss = G_adv_loss1 + G_adv_loss2 + G_adv_loss3 + G_adv_loss4
            # #loss_seg = loss_seg.mean()
            #G_train_loss = 0.05 * G_adv_loss + loss_seg
            #G_train_loss.backward()
            G_train_loss4 = 0.05 * G_adv_loss4 + lossseg4
            G_train_loss3 = 0.05 * G_adv_loss3 + lossseg3
            G_train_loss2 = 0.05 * G_adv_loss2 + lossseg2
            G_train_loss1 = 0.05 * G_adv_loss1 + lossseg1
            G_train_loss4.backward(retain_graph=True)
            G_train_loss3.backward(retain_graph=True)
            G_train_loss2.backward(retain_graph=True)
            G_train_loss1.backward()
            optimizerG.step()

            #print("[{}][{}] LD: {:.4f} LDfake: {:.4f} LD_real: {:.4f} LG: {:.4f} LG_ce: {:.4f} LG_adv: {:.4f}" \
            #      .format(epoch, iteration, (LDr + LDf).item(), LDr.item(), LDf.item(), LGseg.item(), LGce.data.item(),
            #             LGadv.data.item()))
            total_loss += G_train_loss1.item()
            total_f_score += G_adv_loss1.item()
            train_loss1 += lossseg1.item()
            tarin_f_score1 += loss_dice1.item()

            pbar.set_postfix(**{
                'G_train_loss': total_loss / (iteration + 1),
                'G_adv_loss': total_f_score/ (iteration + 1),
                'loss_seg': train_loss1/ (iteration + 1),
                 'dice_loss': tarin_f_score1 / (iteration + 1),
                'lr': get_lr(optimizerG)})
            pbar.update(1)
    print('Finish Train')
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step:
                break
            imgs, mask, one_hot_mask = batch

            with torch.no_grad():
                modelG_train.eval()
                imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
                mask = torch.from_numpy(np.array(mask)).long()
                one_hot_mask = torch.from_numpy(np.array(one_hot_mask)).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    mask = mask.cuda()
                    one_hot_mask = one_hot_mask.cuda()
                    weights = weights.cuda()
                outputs,outputs1,outputs2, outputs3 = modelG_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, mask, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, mask, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, one_hot_mask)
                    loss = loss + main_dice
                # -------------------------------#
                #   计算f_score
                # -------------------------------#

                _f_score = f_score(outputs, one_hot_mask)

                val_loss += main_dice.item()
                val_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizerG)})
            pbar.update(1)
    if (epoch + 1) % 50 == 0:
        torch.save(modelG.state_dict(),
                   'Kits_logs/modelG_AUGunet_4scale_4P_kidney_ep%03d.pth' % ((epoch + 1)))
        # torch.save(modelD1.state_dict(),
        #            'Kits_log/modelD_end_msgan_ep%03d.pth' % ((epoch + 1)))



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs,
                                                                                                 temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

def one_hot(label,num_classes):
    label = label.cpu().numpy()
    one_hot = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot).cuda()
