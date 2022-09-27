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

# Kernel part
# Blurry image = I_h
B = I_h
# gamma value = 2.2 (It should be based on camera CCD value but I don't have info)
# Selected sub-window P as
P = I_h[80:200, 80:200]

# Inverse gamma correct P (default gamma = 2.2)
# Calculate gradient
x_dir = cv2.Sobel(P, -1, dx = 1, dy = 0)
y_dir = cv2.Sobel(P, -1, dx = 0, dy = 1)
un_normalized_val = np.abs(x_dir) + np.abs(y_dir)

# Our overall blurr direction is all around!
S = np.around(-2 * np.log2(3/0.5))

# for s in S:
#     P_s = cv2.resize(un_normalized_val, )



#---------------------------------------------------------------------------------#
# One more additional method: Richardson-Lucy

from numpy.fft import fftn, ifftn, fftshift 

def richardson_lucy_np(image, psf, num_iters):
    
    otf = fftn(fftshift(psf))
    otf_ = np.conjugate(otf)    #np.ones(image.shape)/image.sum()
    estimate = image

    for i in range(num_iters):
        #print(i)
        
        reblurred = ifftn(fftn(estimate) * otf)
        ratio = image / (reblurred + 1e-30)
        estimate = estimate * (ifftn(fftn(ratio) * otf_)).astype(float)
        

    return estimate

img = richardson_lucy_np(I_gt, I_h, 1000)
print(PSNR(I_gt, img))