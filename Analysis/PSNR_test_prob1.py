# Computer vision programming assignment 1
# This code is written by
# 20175003 Sunghyun Kang

# PSNR Analysis for this HW

import numpy as np
import cv2

# This is for progress bar
from tqdm import tqdm

# This is for plotting
import matplotlib.pyplot as plt

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

ITERATION_TIME = 10000

# MSE, PSNR Values
alpha = 0.1
init_alpha = alpha
alpha_val = []
mse_val = []
psnr_val = []

for i in tqdm(range(0, ITERATION_TIME)):

    # start the gradient descent
    # define the max iteration count
    MAX_ITER = 1000
    for counter in range(0, MAX_ITER):
        
        # gradient value
        difference = cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR) - I_l
        grad = cv2.resize(difference, (height, width))

        # Update the value
        I_h = np.subtract(I_h, np.multiply(alpha, grad))

        # This is an error calculation part for the debugging process
        # e = np.square(difference)
        # print(np.sum(e))

    mse = MSE(I_gt, I_h, width, height)
    psnr = PSNR(I_gt, mse)
    mse_val.append(mse)
    psnr_val.append(psnr)
    alpha_val.append(alpha)
    alpha += 0.000024

#---------------------------------------------------------------------------------#

# # Plotting part
# plt.figure(figsize = (15, 10))
# plt.plot(alpha_val, mse_val, color = 'black')
# plt.title("Tendency between MSE and step size (at 1000 iteration)", fontsize = 20)
# plt.axhline(500, color='black')
# plt.axvline(init_alpha, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('MSE value', fontsize = 17)

# plt.savefig("Problem1_MSE.png", dpi=300)

# plt.figure(figsize = (15, 10))
# plt.plot(alpha_val, psnr_val, color = 'black')
# plt.title("Tendency between PSNR and step size (at 1000 iteration)", fontsize = 20)
# plt.axhline(30, color='black')
# plt.axvline(init_alpha, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('MSE value', fontsize = 17)

# plt.savefig("Problem1_PSNR.png", dpi=300)

np.save("/home/Computer_Vision_PA1/Analysis/alpha_val.npy", alpha_val)
np.save("/home/Computer_Vision_PA1/Analysis/mse_val.npy", mse_val)
np.save("/home/Computer_Vision_PA1/Analysis/psnr_val.npy", psnr_val)
