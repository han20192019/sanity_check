import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import sigpy.mri as mr
import pywt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
import os
import pandas as pd
from PIL import Image
from deepinpy.utils.utils import h5_write
from scipy.ndimage.interpolation import rotate


def imshowgray(im, vmin=None, vmax=None):
	plt.imshow(im, cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)

def fft2c(x):
	return 1 / np.sqrt(np.prod(x.shape)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(y):
	return np.sqrt(np.prod(y.shape)) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(y)))
directory = "/mikQNAP/NYU_knee_data/singlecoil_train"

def diagonal_mask(height=640, width=372, angle=45):
    dim = max(height, width)
    mask = np.zeros((dim, dim))
    mask[:,(dim//2)-50:(dim//2)+50] += 1
    mask = rotate(mask, angle=angle)
    side = mask.shape[0]
    return mask[int((side-height)/2): int((side-height)/2)+height, int((side-width)/2): int((side-width)/2)+width]



def vardens_mask(mask):
    oneCoilMask = mr.poisson(mask.shape, accel = 2, calib = (20, 20), crop_corner = False)
    return oneCoilMask * mask


imgs = []
masks = []
ksp = []
maps = []
loss_masks = []
count = 0
num_slice = 0
for ii in os.listdir(directory):
    count += 1
    print(num_slice)
    if num_slice>12:
        break
    if ii.endswith('.h5'):
        with h5py.File(os.path.join(directory, ii), 'r') as data:
            print(list(data.keys()))
            kspace_orig = data['kspace']
            #print(kspace_orig.shape)

            if (kspace_orig.shape[1] != 640) | (kspace_orig.shape[2] != 372):
                print('scan is not 640x372 pixels')
            else:
                for num in range(8,31):
                    if num_slice>12:
                        break
                    sli_ori = kspace_orig[num]
                    img_ori = ifft2c(sli_ori)
                    width = sli_ori.shape[0]
                    height = sli_ori.shape[1]

                    mask = np.random.randn(width, height)
                    num_slice += 1

                    imgs.append(img_ori)
                    masks.append(mask)
                    ksp.append([sli_ori])
                    maps.append(np.ones(np.array([1, kspace_orig.shape[1], kspace_orig.shape[2]])))
                    #print(np.array(ksp).shape)
print(num_slice)
masks_train = np.array(masks, dtype=np.float)
#masks_train = np.random.randn(13, 640, 372)
#masks_train = np.array(masks_train, dtype=np.float)

ksp_train = np.array(ksp, dtype=np.complex64)

#ksp_train = np.random.randn(13, 1, 640, 372)
#ksp_train = np.array(ksp_train, dtype=np.complex)

maps_train = np.array(maps, dtype=np.complex64)
imgs_train = np.array(imgs, dtype=np.complex64)
print(ksp_train.shape)

data_train = {'imgs': imgs_train, 'masks': masks_train, 'maps': maps_train, 'ksp': ksp_train}
h5_write("debug_0102_sanity.h5", data_train)
