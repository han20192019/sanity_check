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
            kspace_orig = data['kspace']
            #print(kspace_orig.shape)

            if (kspace_orig.shape[1] != 640) | (kspace_orig.shape[2] != 372):
                print('scan is not 640x372 pixels')
            else:
                for sli_ori in kspace_orig[8:30]:
                    width = sli_ori.shape[0]
                    height = sli_ori.shape[1]

                    im = ifft2c(sli_ori)
                    im_mag = np.abs(im)
                    magnitude_vals = im_mag.reshape(-1)
                    mag_vals_sorted = np.sort(magnitude_vals)
                    k = int(round(0.95 * magnitude_vals.shape[0]))
                    scale_factor = mag_vals_sorted[k]    
                    im_scaled = im / scale_factor # here we normalize the complex-valued image
                    #plt.savefig("./ex.jpg")

                    sli = np.array(fft2c(im_scaled))

                    #typ = np.random.randint(3)
                    typ = num_slice%4
                    if typ == 0:
                        mask = np.array(vardens_mask(diagonal_mask(width, height, 0)))
                        loss_mask = np.array((diagonal_mask(width, height, 0)))
                    elif typ == 1:
                        mask = np.array(vardens_mask(diagonal_mask(width, height, 45)))
                        loss_mask = np.array((diagonal_mask(width, height, 45)))
                    elif typ == 2:
                        mask = np.array(vardens_mask(diagonal_mask(width, height, 90)))
                        loss_mask = np.array((diagonal_mask(width, height, 90)))
                    elif typ == 3:
                        mask = np.array(vardens_mask(diagonal_mask(width, height, 135)))
                        loss_mask = np.array((diagonal_mask(width, height, 135)))
                    num_slice += 1

                    """
                    imshowgray(abs(mask))
                    plt.show()
                    plt.savefig("./mask.jpg")
                    """

                    newsli = sli*mask

                    imgs.append(im_scaled)
                    masks.append(mask)
                    loss_masks.append(loss_mask)
                    ksp.append([sli])
                    maps.append(np.ones(np.array([1, kspace_orig.shape[1], kspace_orig.shape[2]])))
                    #print(np.array(ksp).shape)
print(num_slice)
masks_train = np.array(masks, dtype=np.float64)
ksp_train = np.array(ksp, dtype=np.complex64)
maps_train = np.array(maps, dtype=np.complex64)
imgs_train = np.array(imgs, dtype=np.complex64)

data_train = {'imgs': imgs_train, 'masks': masks_train, 'maps': maps_train, 'ksp': ksp_train}
h5_write("0102_sanity.h5", data_train)
