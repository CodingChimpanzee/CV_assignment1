# Computer vision programming assignment 1
# This code is written by
# 20175003 Sunghyun Kang

# Problem 1 (gradient descent without prior)
# import two dependencies
import numpy as np
import cv2

# this is for progress bar
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

# Display before image MSE, PSNR value
mse = MSE(I_gt, I_h, width, height)
psnr = PSNR(I_gt, mse)
print("Image import success! Here's upscaled image's MSE, PSNR value.")
print("MSE value: ", mse)
print("PSNR value: ", psnr)
print("Error: ", np.sum(np.square(cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR) - I_l)))

#---------------------------------------------------------------------------------#

# start the gradient descent
# define the max iteration count
MAX_ITER = 10000
counter = 0
alpha = 0.9
for counter in tqdm(range(0, MAX_ITER)):
    counter += 1
    
    # gradient value
    difference = cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR) - I_l
    grad = cv2.resize(difference, (height, width))

    # Update the value
    I_h = np.subtract(I_h, np.multiply(alpha, grad))

    # This is an error calculation part for the debugging process
    # e = np.square(difference)
    # print(np.sum(e))

#---------------------------------------------------------------------------------#


mse = MSE(I_gt, I_h, width, height)
psnr = PSNR(I_gt, mse)

print("Iteration complete! Here's processed image's MSE, PSNR value.")
print("MSE value: ", mse)
print("PSNR value: ", psnr)
print("Error: ", np.sum(np.square(cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR) - I_l)))

# save the image
cv2.imwrite('/home/Computer_Vision_PA1/problem1.png', I_h)
