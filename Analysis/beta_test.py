# Iteration test for problem 1

# Computer vision programming assignment 1
# This code is written by
# 20175003 Sunghyun Kang

# PSNR Analysis for this HW

import numpy as np
import cv2

# This is for progress bar
from tqdm import tqdm

#---------------------------------------------------------------------------------#

# MSE, PSNR definition
def MSE(image_gt, image_h, width, height):
    difference_square = np.sum(np.square(image_gt-image_h))
    return np.divide(difference_square, height*width)

def PSNR(image_gt, mse):
    r = np.amax(image_gt)
    return 10*np.log10(np.divide(r**2, mse))

#---------------------------------------------------------------------------------#

# import upsampled image as greyscale (I_h0)
I_h = cv2.imread('/home/Computer_Vision_PA1/upsampled.png')
I_h = cv2.cvtColor(I_h, cv2.COLOR_BGR2GRAY)
# Change the form into float type
I_h = np.array(I_h, dtype = float)

# find the height and width of the image
height, width = I_h.shape

# import the ground_truth image
I_gt = cv2.imread('/home/Computer_Vision_PA1/HR.png')
I_gt = cv2.cvtColor(I_gt, cv2.COLOR_BGR2GRAY)
# Change the form into float type
I_gt = np.array(I_gt, dtype = float)

# define low resolution image input
# which refers to bilinear downsampling(I_h)
I_l = cv2.resize(I_gt, (height//4, width//4), interpolation = cv2.INTER_LINEAR)

#---------------------------------------------------------------------------------#

# Prior part, please be careful with normalization part
# 1. define hyperparameter gamma and get gradient of I_h

Gamma = 6

x_dir = cv2.Sobel(I_h, -1, dx = 1, dy = 0)
y_dir = cv2.Sobel(I_h, -1, dx = 0, dy = 1)
un_normalized_val = np.abs(x_dir) + np.abs(y_dir)
# Normalize
maxpoint = np.max(I_h)
minpoint = np.min(I_h)
G_h0 = np.divide(np.subtract(un_normalized_val, minpoint), np.subtract(maxpoint, minpoint)) + 1e-10

# 2. get Laplacian of I_h
un_normalized_laplacian = cv2.Laplacian(I_h, -1)
# Normalize
maxpoint = np.max(un_normalized_laplacian)
minpoint = np.min(un_normalized_laplacian)
laplacian = np.divide(np.subtract(un_normalized_laplacian, minpoint), np.subtract(maxpoint, minpoint))
# laplacian = cv2.normalize(un_normalized_laplacian, None, 0, 1, cv2.NORM_MINMAX)

# 3. get the sharp edge
G_t = G_h0 - laplacian
G_t = np.clip(G_t, a_min = 0.0, a_max = 1.0)

# 4. Now get the gradient of prior
grad_prior = Gamma*laplacian*np.divide(G_t, G_h0)

#---------------------------------------------------------------------------------#

beta_val = []
mse_val = []
psnr_val = []
# MSE, PSNR Values
Alpha = 0.002
Beta = 0.0001
iter = 1000

for i in tqdm(range(1000)):

    # start the gradient descent
    # define the max iteration count
    for counter in range(0, iter):
        
        updated_laplacian = cv2.Laplacian(I_h, -1)
        
        I_dt = cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR)
        difference = I_dt - I_l

        # Get the prior and update gradient
        prior = Beta*(updated_laplacian - grad_prior)
        updated_grad = cv2.resize(difference, (height, width)) - prior
        
        # Update the value
        I_h = I_h - Alpha*updated_grad

    mse = MSE(I_gt, I_h, width, height)
    psnr = PSNR(I_gt, mse)
    mse_val.append(mse)
    psnr_val.append(psnr)
    beta_val.append(Beta)
    Beta += 0.0001

#---------------------------------------------------------------------------------#

# Save the value
np.save("/home/Computer_Vision_PA1/Analysis/beta_val_2.npy", beta_val)
np.save("/home/Computer_Vision_PA1/Analysis/beta_mse_val_2.npy", mse_val)
np.save("/home/Computer_Vision_PA1/Analysis/beta_psnr_val_2.npy", psnr_val)
