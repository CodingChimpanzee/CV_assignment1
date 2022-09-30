# Computer vision programming assignment 1
# This code is written by
# 20175003 Sunghyun Kang

# This code is written for additional method
# Making blur kernel based on gaussian distribution.

# import dependencies
import numpy as np
import cv2
from scipy.signal import convolve2d, fftconvolve
from tqdm import tqdm

#---------------------------------------------------------------------------------#

# MSE, PSNR definition
def MSE(image_gt, image_h, width, height):
    difference_square = np.sum(np.square(image_gt-image_h))
    return np.divide(difference_square, height*width)

def PSNR(mse):
    r = 255.0
    return 10*np.log10(np.divide(r**2, mse))

#---------------------------------------------------------------------------------#

# import upsampled image as greyscale (I_h0)
I_h = cv2.imread('../Images/upsampled.png')
I_h = cv2.cvtColor(I_h, cv2.COLOR_BGR2GRAY)
# Change the form into float type
I_h = np.array(I_h, dtype = np.float32)
# Clip the maximum value as 255
I_h = np.clip(I_h, 0, 255)

# find the height and width of the image
height, width = I_h.shape

# import the ground_truth image
I_gt = cv2.imread('../Images/HR.png')
I_gt = cv2.cvtColor(I_gt, cv2.COLOR_BGR2GRAY)
# Change the form into float type
I_gt = np.array(I_gt, dtype = np.float32)

# Display before image MSE, PSNR value
mse = MSE(I_gt, I_h, width, height)
psnr = PSNR(mse)
print("Image import success! Here's upscaled image's MSE, PSNR value.")
print("MSE value: ", mse)
print("PSNR value: ", psnr)

#---------------------------------------------------------------------------------#

# Richardson-Lucy
# To obtain more faster result, you may use fftconvolve
# But it will not guarantee the clear image
def richardson_lucy_np(image, kernel, num_iters):
    # image = image in 2d numpy array
    # num_iters = number of iteration
    # kernel = For the PSF (Point Spread Function)

    reverse_kernel = np.flip(kernel)

    k_h, k_w = kernel.shape
    h, w = image.shape

    estimation = image
    temp_estimation = cv2.resize(estimation, (k_h, k_w))
    temp_image = cv2.resize(image, (k_h, k_w))
    for i in tqdm(range(0, num_iters)):
        convolution = convolve2d(temp_estimation, kernel, boundary='symm', mode='same')
        # convolution = fftconvolve(estimation, kernel, mode='same')
        blur_relative = np.divide(temp_image, convolution)
        estimate_error = convolve2d(blur_relative, reverse_kernel, boundary='symm', mode='same')
        # estimate_error = fftconvolve(blur_relative, reverse_kernel, mode='same')
        
        # Original equation that follows Richardson-Lucy deconvolution
        # estimation = np.multiply(estimation, estimate_error)
        
        # Equation that I used to invert the image
        estimation *= cv2.resize(np.multiply(temp_estimation, estimate_error), (h,w))
        temp_estimation = cv2.resize(estimation, (k_h, k_w))

    return cv2.resize(estimation, (h, w))

#---------------------------------------------------------------------------------#

# Load the kernel that previously generated
# This kernel is produced from the code
# https://github.com/avasile96/ip_project
kernel = np.loadtxt('./KERNEL.csv', delimiter=',')
result = richardson_lucy_np(I_h.copy(), kernel, 4)
cv2.imwrite('../Images/addtional_filter.png', result)

# Appropriately choose the value
result1 = result.copy()
for i in range(256):
    for j in range(256):
        if result1[i][j] < 1e+1:
            result1[i][j] = I_h[i][j]
        else:
            result1[i][j] = 0

# Sum up the two image
result2 = cv2.imread('../Images/problem2.png', cv2.COLOR_BGR2GRAY)
result = np.add(result1, result2)
result = np.clip(result, 0, 255)

# Pass the bilateral filter to obtain descent result
result = cv2.bilateralFilter(result, -1, 28, 4)

# Get MSE and PSNR
mse = MSE(I_gt, result, width, height)
psnr = PSNR(mse)

print("Iteration complete! Here's processed image's MSE, PSNR value.")
print("MSE value: ", mse)
print("PSNR value: ", psnr)

# save the image
cv2.imwrite('../Images/addtional.png', result)