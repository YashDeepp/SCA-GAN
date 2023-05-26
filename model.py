# -*- coding: utf-8 -*-
# @Author: JacobShi777

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import random
import argparse
import random
import functools
import torch.nn as nn
from torch.autograd import Variable
import math
import torchvision.models as models
import torch.utils.model_zoo as model_zoo



# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        return self.model(input)



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # down = [downconv]
            down = [nn.Conv2d(3, inner_nc, kernel_size=4,
                             stride=2, padding=1)]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)



class UnetGenerator2(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator2, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock2(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock2(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock2(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        return self.model(input)



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock2(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock2, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # down = [downconv]
            down = [nn.Conv2d(4, inner_nc, kernel_size=4,
                             stride=2, padding=1)]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)



# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            # nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.Conv2d(12, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        return self.model(input)




# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        target_tensor = target_tensor.cuda()
        return self.loss(input, target_tensor)



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



def localLossL1(fake_s, real_s, real_p, criterionL1):
    sh0, s0 = dot_product(fake_s, real_s, real_p[:,3,:,:])
    sh1, s1 = dot_product(fake_s, real_s, real_p[:,4,:,:])
    sh2, s2 = dot_product(fake_s, real_s, real_p[:,5,:,:])
    sh3, s3 = dot_product(fake_s, real_s, real_p[:,6,:,:])
    sh4, s4 = dot_product(fake_s, real_s, real_p[:,7,:,:])
    sh5, s5 = dot_product(fake_s, real_s, real_p[:,8,:,:])
    sh6, s6 = dot_product(fake_s, real_s, real_p[:,9,:,:])
    sh7, s7 = dot_product(fake_s, real_s, real_p[:,10,:,:])

    l0 = criterionL1(sh0, s0)
    l1 = criterionL1(sh1, s1)
    l2 = criterionL1(sh2, s2)
    l3 = criterionL1(sh3, s3)
    l4 = criterionL1(sh4, s4)
    l5 = criterionL1(sh5, s5)
    l6 = criterionL1(sh6, s6)
    l7 = criterionL1(sh7, s7)

    loss = l0 + l1 + l2 + l3 + l4 + l5+ l6 + l7
    # lines = [l0.data.cpu().numpy()[0],l1.data.cpu().numpy()[0],l2.data.cpu().numpy()[0],\
    #         l3.data.cpu().numpy()[0],l4.data.cpu().numpy()[0],l5.data.cpu().numpy()[0],\
    #         l6.data.cpu().numpy()[0],l7.data.cpu().numpy()[0]]

    return loss
    # return loss, lines

def localLossL1_2(fake_s, real_s, real_p, criterionL1):
    parsing = real_p[:,3:,:,:].data
    parsing = torch.max(parsing, 1)
    probs = []
    for i in range(8):
        index = (torch.ones(parsing[1].size()) * i).cuda()
        prob = torch.eq(parsing[1].float(), index).float()
        probs.append(prob)

    sh, s = [], []
    for i in range(8):
        sh0, s0 = dot_product(fake_s, real_s, Variable(probs[i]))
        sh.append(sh0)
        s.append(s0)

    loss = 0.
    for i in range(8):
        l0 = criterionL1(sh[i], s[i])
        loss += l0

    return loss

def dot_product(fake_s, real_s, parsing):
    sh = torch.mul(fake_s, parsing)
    s = torch.mul(real_s, parsing)
    return sh, s

def avgpoolLoss(fake_s, real_s, criterionL1):
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    fake_s = avgpool(avgpool(fake_s))
    real_s = avgpool(avgpool(real_s))

    loss = criterionL1(fake_s, real_s)

    return loss

def maxpoolLoss(fake_s, real_s, criterionL1):
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    fake_s = maxpool(maxpool(fake_s))
    real_s = maxpool(maxpool(real_s))

    # fake_s = maxpool(fake_s)
    # real_s = maxpool(real_s)

    loss = criterionL1(fake_s, real_s)

    return loss

def separate2(inputs, real_p):
    parsing = real_p[:,3:,:,:].data
    parsing = torch.max(parsing, 1)
    index = (torch.ones(parsing[1].size()) * 7).cuda()
    prob7 = torch.eq(parsing[1].float(), index).float()

    prob7_ = torch.ones(prob7.size()).cuda() - prob7
    if inputs.size(1)>2:
        prob7_ = torch.cat([prob7_,prob7_,prob7_],1)
        prob7 = torch.cat([prob7,prob7,prob7],1)
    hair_no = torch.mul(inputs, Variable(prob7_))
    hair_yes = torch.mul(inputs, Variable(prob7))

    return hair_no, hair_yes

def gaussianMH():
    lists = [(0,0), (1,0), (1,1)]
    var = 1
    vals = []
    gauFilter = np.zeros((3,3))
    for item in lists:
        x, y = item
        val = float(np.exp(-(x**2+y**2)/(2.0*var**2))/(2*math.pi*(var**2)))
        vals.append(val)
    gauFilter[1,1] = vals[0]
    gauFilter[0,1] = gauFilter[1,0] = gauFilter[1,2] = gauFilter[2,1] = vals[1]
    gauFilter[0,0] = gauFilter[0,2] = gauFilter[2,0] = gauFilter[2,2] = vals[2]
    sums = np.sum(gauFilter)
    gauFilter = gauFilter / sums
    gauFilter = Variable(torch.from_numpy(gauFilter)).unsqueeze(0).unsqueeze(0).cuda()
    gauFilter = gauFilter.float()

    return gauFilter

def doGaussian(x):
    gauFilter = gaussianMH()
    channels = x.size(1)
    lists1 = []
    for i in range(channels):
        lists1.append(x[:,i,:,:].unsqueeze(1))
    lists2 = []
    for item in lists1:
        item = nn.functional.conv2d(item, gauFilter, padding=1)
        lists2.append(item)
    x = torch.cat(lists2, 1)

    return x

def multiScaleLLS(gen, gr, downSampleNums, criterionL1, downFilters):
    if downFilters[0].size(0) != gen.size(0):
        downFilters = createFilters(gen.size(2), gen.size(0))
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    if downSampleNums != 0:
        gen = doGaussian(gen)
        gr = doGaussian(gr)

        downFilters = downFilters[0] if downSampleNums == 1 else downFilters[1]
        gen = torch.mul(gen, downFilters)
        gr = torch.mul(gr, downFilters)

        for i in range(downSampleNums):
            gen = maxpool(gen)
            gr = maxpool(gr)
    lapFilters = Variable(torch.FloatTensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])).unsqueeze(0).unsqueeze(0).cuda()
    gen_lap = nn.functional.conv2d(gen, lapFilters, padding=1)
    gr_lap = nn.functional.conv2d(gr, lapFilters, padding=1)
    loss = criterionL1(gen_lap, gr_lap)

    return loss


def createFilters(to_resol, batchSize):
    lists = []
    for k in [2,4]:
        filters = np.zeros((to_resol, to_resol))
        h,w = filters.shape
        for i in range(h):
            for j in range(w):
                if (i%k == 0) and (j%k == 0):
                    filters[i,j] = 1
        lists.append(filters)
    lastLists = []
    for filters in lists:
        filters = torch.from_numpy(filters)
        filters = filters.expand(batchSize,1,to_resol,to_resol)
        filters = Variable(filters)
        filters = filters.cuda()
        lastLists.append(filters.float())

    return lastLists


class EncoderLayer(nn.Module):
    '''
    EncoderLayer

    part of VGG19 (through relu_4_1)

    ref:
    https://arxiv.org/pdf/1703.06868.pdf (sec. 6)
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    '''

    def __init__(self, batch_norm):
        super(EncoderLayer, self).__init__()
        # enc_layers = list(models.vgg.children())
        # conf = nn.Sequential(*enc_layers[:12])  # input -> relu4_1
        conf = models.vgg.cfgs['E'][:12]  # VGG through relu_4_1
        self.features = models.vgg.make_layers(conf, batch_norm=batch_norm)

    def forward(self, x):
        return self.features(x)


def make_encoder(model_file, batch_norm=True):
    '''
    make a pretrained partial VGG-19 network
    '''
    VGG_TYPE = 'vgg19_bn' if batch_norm else 'vgg19'

    enc = EncoderLayer(batch_norm)

    if model_file and os.path.isfile(model_file):
        # load weights from pre-saved model file
        enc.load_state_dict(torch.load(model_file))
    else:
        # load weights from pretrained VGG model
        vgg_weights = model_zoo.load_url(models.vgg.model_urls[VGG_TYPE])
        w = {}
        for key in enc.state_dict().keys():
            w[key] = vgg_weights[key]
        enc.load_state_dict(w)
        if not model_file:
            model_file = "encoder.model"
        torch.save(enc.state_dict(), model_file)

    return enc


class PerceptualLoss(nn.Module):
    '''
    Implement Perceptual Loss in a VGG network

    ref:
    https://github.com/ceshine/fast-neural-style/blob/master/style-transfer.ipynb
    https://arxiv.org/abs/1603.08155

    input: BxCxHxW, BxCxHxW
    output: loss type Variable
    '''

    def __init__(self, vgg_model, n_layers):
        super(PerceptualLoss, self).__init__()
        self.vgg_layers = vgg_model.features

        # use relu_1_1, 2_1, 3_1, 4_1
        if n_layers == 3:
            self.use_layer = set(['2', '25', '29'])
        elif n_layers == 2:
            self.use_layer = set(['2', '25'])
        # self.use_layer = set(['2', '9', '16', '29'])
        self.mse = torch.nn.MSELoss()

    def forward(self, g, s):
        loss = 0

        for name, module in self.vgg_layers._modules.items():

            g, s = module(g), module(s)
            if name in self.use_layer:
                s = Variable(s.data, requires_grad=False)
                loss += self.mse(g, s)

        return loss



