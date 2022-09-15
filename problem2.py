# Computer vision programming assignment 1
# This code is written by
# 20175003 Sunghyun Kang

# Problem 2 (gradient descent with prior)
# import two dependencies
import numpy as np
import cv2

# this is for progress bar
from tqdm import tqdm
import time

#---------------------------------------------------------------------------------#

# MSE, PSNR definition
def MSE(image_gt, image_h, width, height):
    difference_square = 0
    for i in range(height):
        for j in range(width):
            temp = image_gt[i][j] - image_h[i][j]
            difference_square += np.square(temp)
    return np.divide(difference_square, height*width)

def PSNR(image_gt, mse):
    r = np.amax(image_gt)
    return 10*np.log10(np.divide(r**2, mse))

#---------------------------------------------------------------------------------#

# import upsampled image as greyscale (I_h0)
I_h = cv2.imread('/home/Computer_Vision_PA1/upsampled.png')
I_h = cv2.cvtColor(I_h, cv2.COLOR_BGR2GRAY)

# find the height and width of the image
height, width = I_h.shape

# import the ground_truth image
I_gt = cv2.imread('/home/Computer_Vision_PA1/HR.png')
I_gt = cv2.cvtColor(I_gt, cv2.COLOR_BGR2GRAY)

# define low resolution image input
# which refers to bilinear downsampling(I_h)
I_l = cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR)

# Display before image MSE, PSNR value
mse = MSE(I_gt, I_h, width, height)
psnr = PSNR(I_gt, mse)
print("Image import success! Here's upscaled image's MSE, PSNR value.")
print("MSE value: ", mse)
print("PSNR value: ", psnr)

#---------------------------------------------------------------------------------#

# Prior part, please be careful with normalization part
# 1. define hyperparameter gamma and get gradient of I_l
Gamma = 6
x_dir = cv2.Sobel(I_h, -1, dx = 1, dy = 0)
y_dir = cv2.Sobel(I_h, -1, dx = 0, dy = 1)
un_normalized_val = np.abs(x_dir) + abs(y_dir) + 1e-10
G_lu = cv2.normalize(un_normalized_val, None, 0, 1, cv2.NORM_MINMAX)  

# 2. get Laplacian of I_l
un_normalized_laplacian = cv2.Laplacian(I_h, -1)
laplacian = cv2.normalize(un_normalized_laplacian, None, 0, 1, cv2.NORM_MINMAX)

# 3. get the sharp edge
G_t = G_lu - laplacian
G_t = np.clip(G_t, a_min = 0.0, a_max = 1.0)

# 4. Now get the gradient of prior
grad_prior = Gamma*laplacian*np.divide(G_t, G_lu)

#---------------------------------------------------------------------------------#

# Iteration part (gradient descent)
# Define beta(hyperparameter) and iteration time
Alpha = 0.03
Beta = 0.001
MAX_ITER = 10000
counter = 0

for counter in tqdm(range(0, MAX_ITER)):
    counter += 1
    
    updated_laplacian = cv2.Laplacian(I_h, -1)
    
    I_dt = cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR)
    difference = I_dt - I_l

    # Get the prior and update gradient
    prior = Beta*(updated_laplacian - grad_prior)
    updated_grad = cv2.resize(difference, (height, width)) - prior
    
    # Update the value
    I_h = I_h - Alpha*updated_grad

#---------------------------------------------------------------------------------#

# check the MSE and PSNR

print("Iteration complete! Here's processed image's MSE, PSNR value.")
mse = MSE(I_gt, I_h, width, height)
psnr = PSNR(I_gt, mse)

print("MSE value: ", mse)
print("PSNR value: ", psnr)

# save the image
cv2.imwrite('/home/Computer_Vision_PA1/problem2.png', I_h)