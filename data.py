# -*- coding: utf-8 -*-
# @Author: JacobShi777

import cv2
import os
import random
import numpy as np
import torch
import random
import scipy.io as sio
import glob
from PIL import Image
import _pickle as cPickle
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data as data


def formnames(infofile, if_train):
    infofile = infofile[0] if if_train else infofile[1]
    res = []
    with open(infofile, 'r') as f:
        for line in f:
            line = line.strip()
            res.append(line)
    return res


def input_transform(if_train, opt):
    if if_train:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    return transform


def target_transform(if_train, opt):
    if if_train:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    return transform


def load_inputs(imgpath, matpath, opt, if_train):
    # load photos and paraing
    tmp = imgpath.split("/")
    imgpath = os.path.join(opt.root, imgpath)
    # print imgpath
    matpath =  tmp[0] + "/"+ tmp[1] + "_mat" + "/" + tmp[2][:-4] + ".mat"
    matpath = os.path.join(opt.root, matpath)
    # print matpath
    img = cv2.imread(imgpath)

    # img = img.astype(np.float32)
    # img = img / 255
    # imgmean = []
    # imgstd = []
    # for i in range(3):
    # 	imgmean.append(float(img[:,:,i].mean()))
    # 	imgstd.append(float(img[:,:,i].var()))
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((\
    # 				imgmean[0], imgmean[1], imgmean[2]),(imgstd[0], imgstd[1], imgstd[2]))])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(img)
    img = np.transpose(img.numpy(), (1, 2, 0))

    img_fl = mat_merge(img, matpath)
    img_fl = np.transpose(img_fl, (2, 0, 1))
    if if_train:
        # img_fl = cv2.resize(img_fl, (opt.loadSize, opt.loadSize))
        img_fl = zero_padding(img_fl, opt.loadSize, opt.loadSize - img_fl.shape[1], opt.loadSize - img_fl.shape[2])
        img_fl = mat_process(img_fl)
    else:
        # img_fl = cv2.resize(img_fl, (opt.fineSize, opt.fineSize))
        img_fl = zero_padding(img_fl, opt.fineSize, opt.fineSize - img_fl.shape[1], opt.fineSize - img_fl.shape[2])
        img_fl = mat_process(img_fl)

    return img_fl


def load_targets(imgpath, opt, if_train):
    # load sketches
    imgpath = os.path.join(opt.root, imgpath)
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    img = img / 255
    img = img.reshape(1, img.shape[0], img.shape[1])
    if if_train:
        # img = cv2.resize(img, (opt.loadSize, opt.loadSize))
        img = zero_padding(img, opt.loadSize, opt.loadSize - img.shape[1], opt.loadSize - img.shape[2])
    else:
        # img = cv2.resize(img, (opt.fineSize, opt.fineSize))
        img = zero_padding(img, opt.fineSize, opt.fineSize - img.shape[1], opt.fineSize - img.shape[2])
    return img


def mat_merge(img, matpath):
    facelabel = sio.loadmat(matpath)
    temp = facelabel['res_label']
    img = np.concatenate((img, temp), axis=2)
    return img


def zero_padding(img, size0, pad1, pad2):
    zero_padding = np.zeros((img.shape[0], size0, size0), dtype=np.float32)
    pad1 = pad1 // 2
    pad2 = pad2 // 2
    zero_padding[:, pad1:size0-pad1, pad2:size0 - pad2] = img
    return zero_padding


def mat_process(img_fl):
    img_fl = img_fl.astype(np.float32)
    img = img_fl[0:3, :, :]

    temp = img_fl[3:, :, :]

    l0 = temp[0, :, :]

    l1 = temp[1, :, :]

    l2 = temp[7, :, :] + temp[6, :, :]
    l2 = np.where(l2 > 1, 1, l2)

    l3 = temp[5, :, :] + temp[4, :, :]
    l3 = np.where(l3 > 1, 1, l3)

    l4 = temp[2, :, :]

    l5 = temp[11, :, :] + temp[12, :, :]
    l5 = np.where(l5 > 1, 1, l5)

    l6 = temp[10, :, :]

    l7 = temp[13, :, :]

    # merge
    img = np.concatenate((img, l0.reshape(1, l0.shape[0], l0.shape[1])), axis=0)
    img = np.concatenate((img, l1.reshape(1, l1.shape[0], l1.shape[1])), axis=0)
    img = np.concatenate((img, l2.reshape(1, l2.shape[0], l2.shape[1])), axis=0)
    img = np.concatenate((img, l3.reshape(1, l3.shape[0], l3.shape[1])), axis=0)
    img = np.concatenate((img, l4.reshape(1, l4.shape[0], l4.shape[1])), axis=0)
    img = np.concatenate((img, l5.reshape(1, l5.shape[0], l5.shape[1])), axis=0)
    img = np.concatenate((img, l6.reshape(1, l6.shape[0], l6.shape[1])), axis=0)
    img = np.concatenate((img, l7.reshape(1, l7.shape[0], l7.shape[1])), axis=0)

    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, opt, if_train):
        super(DatasetFromFolder, self).__init__()

        self.if_train = if_train
        self.opt = opt
        self.imgnames = formnames(opt.infofile, self.if_train)
        self.input_transform_train = input_transform(self.if_train, self.opt)
        self.target_transform_train = target_transform(self.if_train, self.opt)

    def __getitem__(self, index):

        imgname = self.imgnames[index]
        item = imgname.split('||')
        inputs = load_inputs(item[0], item[2], self.opt, self.if_train)
        targets = load_targets(item[1], self.opt, self.if_train)
        # identity = torch.LongTensor([int(item[3])])
        if self.if_train:
            w_offset = random.randint(0, self.opt.loadSize - self.opt.fineSize - 1)
            h_offset = random.randint(0, self.opt.loadSize - self.opt.fineSize - 1)
            # w_offset = 0
            # h_offset = 0
            inputs = inputs[:, h_offset:h_offset + self.opt.fineSize, h_offset:h_offset + self.opt.fineSize]
            targets = targets[:, h_offset:h_offset + self.opt.fineSize, h_offset:h_offset + self.opt.fineSize]
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)

        return inputs, targets

    def __len__(self):
        return len(self.imgnames)


def checkpaths(opt):
    if not os.path.exists(opt.checkpoint):
        os.mkdir(opt.checkpoint)
    if not os.path.exists(opt.gen_root):
        os.mkdir(opt.gen_root)


def checkpoint(epoch, netD1, netD2, netG1, netG2, netE1, netE2):
    netD1_out_path = "./checkpoint/netD1_epoch_{}.weight".format(epoch)
    torch.save(netD1.state_dict(), netD1_out_path)
    netD2_out_path = "./checkpoint/netD2_epoch_{}.weight".format(epoch)
    torch.save(netD2.state_dict(), netD2_out_path)

    netG1_out_path = "./checkpoint/netG1_epoch_{}.weight".format(epoch)
    torch.save(netG1.state_dict(), netG1_out_path)
    netG2_out_path = "./checkpoint/netG2_epoch_{}.weight".format(epoch)
    torch.save(netG2.state_dict(), netG2_out_path)

    netE1_out_path = "./checkpoint/netE1_epoch_{}.weight".format(epoch)
    torch.save(netE1.state_dict(), netE1_out_path)
    netE2_out_path = "./checkpoint/netE2_epoch_{}.weight".format(epoch)
    torch.save(netE2.state_dict(), netE2_out_path)

    print("Checkpoint saved to {} and {}".format(netD1_out_path, netG1_out_path))


def usedtime(strat_time, end_time):
    delta = int(end_time - strat_time)
    hours = delta // 3600
    minutes = (delta - hours * 3600) // 60
    seconds = delta - hours * 3600 - minutes * 60
    return ('%2d:%2d:%2d' % (hours, minutes, seconds))

# class Outsketch_Folder_Set(data.Dataset):
#     def __init__(self, opt, isTrain=0):
#         '''
#             need set opt.img_h and opt.img_w
#         :param opt:
#         :param dirpath:
#         :param use_par:
#         :param b_checkpoint_path:
#         '''
#         super(Outsketch_Folder_Set, self).__init__()
#         self.opt=opt
#         self.root = self.opt.root
#         self.isTrain=isTrain
#         # self.datalist = sorted(glob("{}/*.png".format(self.root)), key=os.path.getctime)
#         self.datalist = os.listdir(self.root)
#         self.lens = len(self.datalist)
#
#     def __getitem__(self, index):
#         line = self.datalist[index]
#         # imgA = Image.open(line).convert("RGB")
#         inputs = Image.open(os.path.join(self.root, line)).convert("RGB")
#         if self.if_train:
#             w_offset = random.randint(0, self.opt.loadSize - self.opt.fineSize - 1)
#             h_offset = random.randint(0, self.opt.loadSize - self.opt.fineSize - 1)
#             # w_offset = 0
#             # h_offset = 0
#             inputs = inputs[:, h_offset:h_offset + self.opt.fineSize, h_offset:h_offset + self.opt.fineSize]
#         inputs = torch.from_numpy(inputs)
#
#         return inputs
#
#     def __len__(self):
#         return self.lens
#
#     def process_img(self, img, i, j, padding):
#         img = padding(img)
#         img = img.crop((j, i, j + self.opt.fineSize, i + self.opt.fineSize))
#         img = transforms.ToTensor()(img)
#         # if self.isTrain == 0:
#         img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
#         return img
