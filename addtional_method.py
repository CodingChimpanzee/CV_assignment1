# Computer vision programming assignment 1
# This code is written by
# 20175003 Sunghyun Kang

# This code is written for additional method
# Making blur kernel based on gaussian distribution.

# import two dependencies
import numpy as np
import cv2

#---------------------------------------------------------------------------------#

# MSE, PSNR definition
def MSE(image_gt, image_h, width, height):
    difference_square = np.sum(np.square(image_gt-image_h))
    return np.divide(difference_square, height*width)

def PSNR(mse):
    r = 255
    return 10*np.log10(np.divide(r**2, mse))

#---------------------------------------------------------------------------------#

# import upsampled image as greyscale (I_h0)
I_h = cv2.imread('/home/Computer_Vision_PA1/upsampled.png')
I_h = cv2.cvtColor(I_h, cv2.COLOR_BGR2GRAY)
# Change the form into float type
I_h = np.array(I_h, dtype = np.float32)

# Clip the maximum value as 255
I_h = np.clip(I_h, 0, 255)

# find the height and width of the image
height, width = I_h.shape

# import the ground_truth image
I_gt = cv2.imread('/home/Computer_Vision_PA1/HR.png')
I_gt = cv2.cvtColor(I_gt, cv2.COLOR_BGR2GRAY)
# Change the form into float type
I_gt = np.array(I_gt, dtype = np.float32)

# define low resolution image input
# which refers to bilinear downsampling(I_h)
I_l = cv2.resize(I_gt, (height//4, width//4), interpolation = cv2.INTER_LINEAR)

#---------------------------------------------------------------------------------#
# Richardson-Lucy

from scipy.signal import convolve2d, fftconvolve
from tqdm import tqdm

def richardson_lucy_np(image, kernel, num_iters):
    # image = image in 2d numpy array
    # num_iters = number of iteration
    
    # How to get PSF? 
    # We will use blur kernel!: Gaussian filter
    # We have to make all of it as furier transform

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
        # estimation += np.multiply(estimation, estimate_error)
        estimation *= cv2.resize(np.multiply(temp_estimation, estimate_error), (h,w))
        temp_estimation = cv2.resize(estimation, (k_h, k_w))

    return cv2.resize(estimation, (h, w))

# kernel = np.abs(np.random.normal(0, 1, size = (width, height))) + 1e-10
# result_is = richardson_lucy_np(I_h, kernel, 100)
# mse = MSE(I_gt, result_is, width, height)
# cv2.imwrite('/home/Computer_Vision_PA1/addtional.png', result_is)
# print(mse)
# print(PSNR(mse))

# Now the trained kernel
# kernel_model = torch.load('/home/Computer_Vision_PA1/REDS_woVAE.pth', map_location=torch.device('cpu'))
# # Change the kernel to dataset
# print(kernel_model['recon_trunk.20.bias'].size())
# kernel = kernel_model['recon_trunk.20.bias'].numpy()
# for i in range(8):
#     kernel = np.append(kernel, kernel)

# kernel = np.reshape(kernel, (width, height))

# kernel = cv2.imread('/home/Computer_Vision_PA1/Assignments_kernel.png')
# kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY)
# kernel = cv2.resize(kernel, (width, height))
# kernel = np.clip(kernel, 0, 255)

kernel = np.loadtxt('/home/Computer_Vision_PA1/KERNEL.csv', delimiter=',')
result = richardson_lucy_np(I_h.copy(), kernel, 4)

cv2.imwrite('/home/Computer_Vision_PA1/addtional_filter.png', result)

result1 = result.copy()
for i in range(256):
    for j in range(256):
        if result1[i][j] < 1e+1:
            result1[i][j] = I_h[i][j]
        else:
            result1[i][j] = 0


# result2 = result.copy()
# for i in range(256):
#     for j in range(256):
#         if result1[i][j] < 1e+04:
#             result1[i][j] = I_h[i][j]
#         else:
#             result1[i][j] = 255

# cv2.imwrite('/home/Computer_Vision_PA1/addtional_filter_2.png', result2)
result2 = cv2.imread('Computer_Vision_PA1/problem2.png', cv2.COLOR_BGR2GRAY)

result = np.add(result1, result2)
result = np.clip(result, 0, 255)

result = cv2.bilateralFilter(result, -1, 28, 4)

# result_is = richardson_lucy_np(I_h, kernel, 6)
# result_is = np.array(result_is, dtype = np.float32)
# cv2.imwrite('/home/Computer_Vision_PA1/addtional_filter.png', result_is)
# result_is = cv2.imread('/home/Computer_Vision_PA1/addtional_filter.png', 0)
# result = cv2.inpaint(I_h, result_is, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

mse = MSE(I_gt, result, width, height)
cv2.imwrite('/home/Computer_Vision_PA1/addtional.png', result)
print(mse)
print(PSNR(mse))