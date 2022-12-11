from PIL import Image
import numpy as np
from scipy.ndimage import filters
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.trains.mot import obj_cnt

def fspecial_gaussian(size, sigma):
    '''
    Function to mimic the 'fspecial' gaussian MATLAB function
    '''
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d


class GaussianBlurConv(nn.Module):
    def __init__(self, truncate=4, sigma=1.5):
        super(GaussianBlurConv, self).__init__()
        size = 2 * int(truncate * sigma + 0.5) + 1
        self.kernel = fspecial_gaussian(size, sigma)
        self.size = size

    def forward(self, x):
        channels = x.shape[1]
        pad = (self.size - 1) // 2
        self.kernel = torch.FloatTensor(self.kernel).unsqueeze(0).unsqueeze(0)
        self.kernel = np.repeat(self.kernel, channels, axis=0)
        # self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
        self.kernel = self.kernel.to(x.device)
        x = F.conv2d(x, self.kernel, padding=pad, groups=channels)
        return x

img_shape = (224, 224)
pt2d = np.zeros(img_shape, dtype=np.float32)
pt = (8, 20)
pt2d[pt[0], pt[1]] = 1.
# pt2d[220, 120] = 1.
sigma = 1.5
density = filters.gaussian_filter(pt2d, sigma, mode='constant')

# kernel = fspecial_gaussian(7, 3)
# print(np.sum(kernel))
blurconv = GaussianBlurConv(4, 1.5)
x = torch.zeros(1, 1, 224, 224)
x[:, 0, 8, 20] = 1
x[:, 0, 20, 40] = 1
density_t = blurconv(x)
cnt_density = obj_cnt(density_t, kernel=13)
cnt_density_t = cnt_density.numpy()
density_t = density_t.numpy()
print(torch.sum(density_t))
