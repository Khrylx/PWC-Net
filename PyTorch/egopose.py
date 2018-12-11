import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import models


pwc_model_fn = './pwc_net.pth.tar';
net = models.pwc_dc_net(pwc_model_fn)
net = net.cuda()
net.eval()


def get_flow(im1, im2):
	im_all = [im[:, :, :3] for im in [im1, im2]]

	# rescale the image size to be multiples of 64
	divisor = 64.
	H = im_all[0].shape[0]
	W = im_all[0].shape[1]
	H_ = int(ceil(H/divisor) * divisor)
	W_ = int(ceil(W/divisor) * divisor)
	for i in range(len(im_all)):
		im_all[i] = cv2.resize(im_all[i], (W_, H_))

	for _i, _inputs in enumerate(im_all):
		im_all[_i] = im_all[_i][:, :, ::-1]
		im_all[_i] = 1.0 * im_all[_i]/255.0
		im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
		im_all[_i] = torch.from_numpy(im_all[_i])
		im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
		im_all[_i] = im_all[_i].float()
	    
	im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)
	flo = net(im_all)
	flo = flo[0] * 20.0
	flo = flo.cpu().data.numpy()
	# scale the flow back to the input size 
	flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
	u_ = cv2.resize(flo[:,:,0],(W,H))
	v_ = cv2.resize(flo[:,:,1],(W,H))
	u_ *= W/ float(W_)
	v_ *= H/ float(H_)
	flo = np.dstack((u_,v_))
	return flo


def visualize_flow(flo, vis_fn):
	mag, ang = cv2.cartToPolar(flo[..., 0], flo[..., 1])
	hsv = np.zeros((flo.shape[0], flo.shape[1], 3), dtype=np.uint8)
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 1] = 255
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	cv2.imwrite(vis_fn, rgb)


im1_fn = 'data/frame_0010.png';
im2_fn = 'data/frame_0011.png';
vis_fn = 'data/vis_flow.png';
im1 = imread(im1_fn)
im2 = imread(im2_fn)
flo = get_flow(im1, im2)
visualize_flow(flo, vis_fn)





