# Computer vision programming assignment 1
# This code is written by
# 20175003 Sunghyun Kang

# Problem 1 (gradient descent without prior)
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

# start the gradient descent
# define the max iteration count
MAX_ITER = 10000
counter = 0
alpha = 0.03
for counter in tqdm(range(0, MAX_ITER)):
    counter += 1
    
    # gradient value
    difference = cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR) - I_l
    grad = cv2.resize(difference, (height, width))

    # Update the value
    I_h = I_h - alpha*grad

#---------------------------------------------------------------------------------#

# check the loss and error
e = np.square(I_l - cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR))

mse = MSE(I_gt, I_h, width, height)
psnr = PSNR(I_gt, mse)

print("Iteration complete! Here's processed image's MSE, PSNR value.")
print("MSE value: ", mse)
print("PSNR value: ", psnr)

# save the image
cv2.imwrite('/home/Computer_Vision_PA1/problem1.png', I_h)
