#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-25
# @Author  : Jehovah
# @File    : test1111.py
# @Software: PyCharm


# -*- coding: utf-8 -*-
# @Author: JacobShi777

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import random
import argparse
import random
import functools
import time

from torch.autograd import Variable
from data import *
from model import *
import option
from myutils import utils
from myutils.vgg16 import Vgg16
from myutils.lcnn import LCNN
from myutils.Unet2 import *
import torchvision.transforms as transforms
import torchvision.utils as vutils
from myutils.vgg16 import Vgg16
opt = option.init()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.myGpu


def train(print_every=10):
    checkpaths(opt)

    # traindata, testdata = load_dataset(opt)
    train_set = DatasetFromFolder(opt, True)
    test_set = DatasetFromFolder(opt, False)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    norm_layer = get_norm_layer(norm_type='batch')

    netD1 = NLayerDiscriminator(opt.input_nc, opt.ndf, n_layers=1, norm_layer=norm_layer, use_sigmoid=False,
                               gpu_ids=opt.gpu_ids)
    netD2 = NLayerDiscriminator(opt.input_nc, opt.ndf, n_layers=1, norm_layer=norm_layer, use_sigmoid=False,
                                gpu_ids=opt.gpu_ids)
    # netD3 = NLayerDiscriminator(opt.input_nc, opt.ndf, n_layers=1, norm_layer=norm_layer, use_sigmoid=False,
    #                             gpu_ids=opt.gpu_ids)

    netG1 = MyUnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False,
                           gpu_ids=opt.gpu_ids)
    netG2 = MyUnetGenerator2(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False,
                             gpu_ids=opt.gpu_ids)
    # netG3 = MyUnetGenerator2(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False,
    #                          gpu_ids=opt.gpu_ids)

    netE1 = MyEncoder(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False,
                     gpu_ids=opt.gpu_ids)
    netE2 = MyEncoder(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False,
                     gpu_ids=opt.gpu_ids)
    # netE3 = MyEncoder(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False,
    #                   gpu_ids=opt.gpu_ids)

    # netVGG = Vgg16()
    # utils.init_vgg16(opt.model_dir)
    # netVGG.load_state_dict(torch.load(os.path.join(opt.model_dir, "vgg16.weight")))
    VGG = make_encoder(model_file=opt.model_vgg)
    perceptual_loss = PerceptualLoss(VGG, 3)
    VGG.cuda()
    # netG3.cuda()
    netG1.cuda()
    netG2.cuda()

    # netD3.cuda()
    netD1.cuda()
    netD2.cuda()

    # netE3.cuda()
    netE1.cuda()
    netE2.cuda()

    # netG3.apply(weights_init)
    netG1.apply(weights_init)
    netG2.apply(weights_init)
    
    # netD3.apply(weights_init)
    netD1.apply(weights_init)
    netD2.apply(weights_init)
    
    # netE3.apply(weights_init)
    netE1.apply(weights_init)
    netE2.apply(weights_init)

    criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan)
    criterionL1 = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    criterionCEL = nn.CrossEntropyLoss()

    # initialize optimizers
    # optimizer_G3 = torch.optim.Adam(netG3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_G1 = torch.optim.Adam(netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_G2 = torch.optim.Adam(netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizer_D3 = torch.optim.Adam(netD3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D1 = torch.optim.Adam(netD1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D2 = torch.optim.Adam(netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizer_E3 = torch.optim.Adam(netE3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_E1 = torch.optim.Adam(netE1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_E2 = torch.optim.Adam(netE2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print('=========== Networks initialized ============')
    # print_network(netG)
    # print_network(netD)
    print('=============================================')
    lam1 = 1.0
    lam2 = 1.0
    lam3 = 1.0

    f = open('./checkpoint/loss.txt', 'w')
    # f2 = open('./checkpoint/recognition.txt', 'w')
    strat_time = time.time()
    for epoch in range(1, opt.n_epoch + 1):
        D_running_loss = 0.0
        G_running_loss = 0.0
        G2_running_loss = 0.0
        downFilters = createFilters(256, 1)
        for (i, batch) in enumerate(training_data_loader, 1):
            real_p, real_s = Variable(batch[0]), Variable(batch[1])
            if opt.cuda:
                real_p, real_s = real_p.cuda(), real_s.cuda()

            real_pgray = 0.299 * real_p[:, 0, :, :] + 0.587 * real_p[:, 1, :, :] + 0.114 * real_p[:, 2, :, :]
            real_pgray = real_pgray.unsqueeze(0)

            # netD1
            optimizer_D1.zero_grad()
            parsing_feature = netE1(real_p[:, 3:, :, :])
            fake_s1 = netG1.forward(real_p[:, 0:3, :, :], parsing_feature)
            fake_ps1 = torch.cat([fake_s1, real_p], 1)
            pred_fake = netD1.forward(fake_ps1.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            real_ps = torch.cat([real_s, real_p], 1)
            pred_real = netD1.forward(real_ps.detach())
            loss_D_real = criterionGAN(pred_real, True)

            loss_D1 = (loss_D_real + loss_D_fake) * 0.5
            loss_D1.backward()
            optimizer_D1.step()

            # netD2
            optimizer_D2.zero_grad()
            parsing_feature2 = netE2(real_p[:, 3:, :, :])
            fake_s2 = netG2.forward(fake_ps1[:, 0:4, :, :].detach(), parsing_feature2)
            fake_ps2 = torch.cat((fake_s2, real_p), 1)
            pred_fake = netD2.forward(fake_ps2.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            pred_real = netD2.forward(real_ps)
            loss_D_real = criterionGAN(pred_real, True)

            loss_D2 = (loss_D_real + loss_D_fake) * 0.5
            loss_D2.backward()
            optimizer_D2.step()

            '''
            optimize netG
            '''
            optimizer_G1.zero_grad()
            optimizer_E1.zero_grad()

            loss_global = criterionL1(fake_s1, real_s)
            loss_local = localLossL1(fake_s1, real_s, real_p, criterionL1)
            loss_G_L1 = opt.alpha1 * loss_global + (1 - opt.alpha1) * loss_local
            loss_G_L1 *= opt.lambda1

            pred_fake = netD1.forward(fake_ps1)
            loss_G_GAN = criterionGAN(pred_fake, True)

            # VGGFace
            b,c,w,h = fake_s1.shape
            yh = fake_s1.expand(b, 3, w, h)
            ys = real_s.expand(b, 3, w, h)
            _mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).expand_as(yh)).cuda()
            _var = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).expand_as(yh)).cuda()
            yh = yh / 2 + 0.5
            ys = ys / 2 + 0.5
            yh = (yh - _mean) / _var
            ys = (ys - _mean) / _var
            loss_recog = perceptual_loss(yh, ys)

            loss_G = loss_G_GAN + loss_G_L1 + opt.styleParam * loss_recog
            loss_G.backward()
            optimizer_G1.step()
            optimizer_E1.step()

            optimizer_G2.zero_grad()
            optimizer_E2.zero_grad()

            loss_global2 = criterionL1(fake_s2, real_s)
            loss_local2 = localLossL1(fake_s2, real_s, real_p, criterionL1)
            loss_G2_L1 = opt.alpha1 * loss_global2 + (1 - opt.alpha1) * loss_local2
            loss_G2_L1 *= opt.lambda1

            pred_fake = netD2.forward(fake_ps2)
            loss_G2_GAN = criterionGAN(pred_fake, True)

            # lap

            # VGGFace
            yh = fake_s2.expand(b, 3, w, h)
            ys = real_s.expand(b, 3, w, h)
            _mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).expand_as(yh)).cuda()
            _var = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).expand_as(yh)).cuda()
            yh = yh / 2 + 0.5
            ys = ys / 2 + 0.5
            yh = (yh - _mean) / _var
            ys = (ys - _mean) / _var
            loss_recog = perceptual_loss(yh, ys)

            loss_G2 = loss_G2_GAN + loss_G2_L1 + opt.styleParam * loss_recog 
            # loss_G2 = loss_G2_GAN + loss_G2_L1 + loss_lap
            loss_G2.backward()
            optimizer_G2.step()
            optimizer_E2.step()


            D_running_loss += loss_D2.item()
            G_running_loss += loss_G2.item()
            G2_running_loss += loss_G2.item()
            if i % print_every == 0:
                end_time = time.time()
                time_delta = usedtime(strat_time, end_time)
                print('[%s-%d, %5d] D loss: %.3f ; G loss: %.3f' % (
                time_delta, epoch, i + 1, D_running_loss / print_every, G_running_loss / print_every))
                f.write('%d,%d,LD0:%.5f,LD:%.5f,LG:%.5f,LO:%.5f,LG2:%.5f,LO2:%.5f\r\n' % (
                epoch, i + 1, loss_D1.item(), loss_D2.item(), loss_G_GAN.item(), loss_G_L1.item(),
                loss_G2_GAN.item(), loss_G2_L1.item()))

                D_running_loss = 0.0
                G_running_loss = 0.0
                G2_running_loss = 0.0
        f.flush()

        if epoch >= 500 and epoch % 5 == 0:
            # test(netG, epoch, testing_data_loader, opt, training_data_loader2, f2, netLCNN, netG2, netE, netE2)
            test(epoch, netG1, netG2, netE1, netE2, testing_data_loader, opt)

            checkpoint(epoch, netD1, netD2, netG1, netG2, netE1, netE2)

    f.close()


def test(epoch, netG1, netG2, netE1, netE2, test_data, opt):
    mkdir(opt.output)
    save_dir_A = opt.output + "/" + str(epoch)
    mkdir(save_dir_A)

    for i, batch in enumerate(test_data):
        real_p, real_s = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            real_p, real_s= real_p.cuda(), real_s.cuda()
        parsing_feature = netE1(real_p[:, 3:, :, :])
        fake_s1 = netG1.forward(real_p[:, 0:3, :, :], parsing_feature)
        fake_ps1 = torch.cat([fake_s1, real_p], 1)
        parsing_feature2 = netE2(real_p[:, 3:, :, :])
        fake_s2 = netG2.forward(fake_ps1[:, 0:4, :, :].detach(), parsing_feature2)
        # fake_ps2 = torch.cat([fake_s2, real_p], 1)
        # parsing_feature3 = netE3(real_p[:, 3:, :, :])
        # fake_s3 = netG3.forward(fake_ps2[:, 0:4, :, :].detach(), parsing_feature3)
        output_name_A = '{:s}/{:s}{:s}'.format(
            save_dir_A, str(i + 1), '.jpg')

        vutils.save_image(fake_s2[:, :, 3:253, 28:228], output_name_A, normalize=True, scale_each=True)

    print(str(epoch) + " saved")


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    train()
