import cv2
import torch
from torch.autograd import Variable
import numpy as np
import os
import torchvision.utils as vutils
from data2 import *
from model import *
import option
from torch.utils.data import DataLoader
from myutils.Unet2 import *
opt = option.init()
norm_layer = get_norm_layer(norm_type='batch')
netG = MyUnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, \
                           use_dropout=False, gpu_ids=opt.gpu_ids)
netG2 = MyUnetGenerator2(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, \
                         use_dropout=False, gpu_ids=opt.gpu_ids)
netE = MyEncoder(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, \
                 use_dropout=False, gpu_ids=opt.gpu_ids)
netE2 = MyEncoder(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, \
                  use_dropout=False, gpu_ids=opt.gpu_ids)
fold = opt.test_epoch
netG.load_state_dict(torch.load('./checkpoint/netG1_epoch_'+fold+'.weight'))
netG2.load_state_dict(torch.load('./checkpoint/netG2_epoch_'+fold+'.weight'))

netE.load_state_dict(torch.load('./checkpoint/netE1_epoch_'+fold+'.weight'))
netE2.load_state_dict(torch.load('./checkpoint/netE2_epoch_'+fold+'.weight'))
netE.cuda()
netE2.cuda()
netG.cuda()
netG2.cuda()

test_set = DatasetFromFolder(opt, False)

testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
# netG = UnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False, gpu_ids=opt.gpu_ids)
if not os.path.exists(opt.output):
    os.makedirs(opt.output)

if not os.path.exists(opt.output):
    os.makedirs(opt.output)

save_dir_A = opt.output + "/" + fold

if not os.path.exists(save_dir_A):
    os.makedirs(save_dir_A)

for i, batch in enumerate(testing_data_loader):
    real_p= batch
    real_p = real_p.cuda()
    parsing_feature = netE(real_p[:, 3:, :, :])
    fake_s1 = netG.forward(real_p[:, 0:3, :, :], parsing_feature)
    fake_ps1 = torch.cat([fake_s1, real_p], 1)
    parsing_feature2 = netE2(real_p[:, 3:, :, :])
    fake_s2 = netG2.forward(fake_ps1[:, 0:4, :, :].detach(), parsing_feature2)

    output_name_A = '{:s}/{:s}{:s}'.format(save_dir_A, str(i + 1), '.jpg')
    print(output_name_A)
    # fake_s2 = fake_s2.squeeze(0).expand(3,256,256)

    # fake_s2 = np.transpose(fake_s2.data.cpu().numpy(), (1, 2, 0)) / 2 + 0.5
    # img = fake_s2[3:253, 28:228, :]
    # cc = (img * 255).astype(np.uint8)
    # cv2.imwrite(output_name_A, cc)
    # vutils.save_image(fake_s2[:, :, 3:253, 28:228], output_name_A, normalize=True, scale_each=True, range=(0.5, 1))
    vutils.save_image(fake_s2[:, :, 3:253, 28:228], output_name_A, normalize=True, scale_each=True)

print(" saved")
