
import cv2
import torch
from torch.autograd import Variable
import numpy as np
import os

netG = torch.load('./checkpoint/netG_epoch_500.pth')
root = './mytest'
root2 = './mygen'
if not os.path.exists(root):
	os.mkdir(root)
if not os.path.exists(root2):
	os.mkdir(root2)

imgnames = os.listdir(root)
imgnames = [x for x in imgnames if '.jpg' in x or '.png' in x]

for imgname in imgnames:
	inpath = os.path.join(root, imgname)
	outname = imgname.split('.')[0]+'_gen.jpg'
	outpath = os.path.join(root2, outname)
	img = cv2.imread(inpath)

	h, w, _ = img.shape
	img = img.astype(np.float32)
	img = img / 256
	img = cv2.resize(img, (256, 256))
	img = np.transpose(img,(2,0,1))
	img = torch.from_numpy(img)
	img = torch.unsqueeze(img, 0)
	img = img.cuda()
	img = Variable(img)
	gen_img = netG(img)
	gen_img = gen_img.data.cpu().numpy()
	img = gen_img.reshape(256, 256)
	img = cv2.resize(img, (w, h))
	img = (img*256).astype(np.uint8)
	cv2.imwrite(outpath, img)
	print('%7s Done'%imgname)


print('======Finished!======')







