# Computer vision programming assignment 1
# This code is written by
# 20175003 Sunghyun Kang

# Problem 2 (gradient descent with prior)
# import two dependencies
import numpy as np
import cv2

# import upsampled image as greyscale (I_h0)
I_h = cv2.imread('/home/Computer_Vision_PA1/upsampled.png', cv2.IMREAD_GRAYSCALE)

# grab the height and width of the image
height, width = I_h.shape

# import the ground_truth image
I_gt = cv2.imread('/home/Computer_Vision_PA1/HR.png', cv2.IMREAD_GRAYSCALE)

# define low resolution image input
# which refers to bilinear downsampling(I_h)
I_l = cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR)