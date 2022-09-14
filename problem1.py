# Computer vision programming assignment 1
# This code is written by
# 20175003 Sunghyun Kang

# Problem 1 (gradient descent without prior)
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

# start the gradient descent
# define the max iteration count
MAX_ITER = 1000
counter = 0
alpha = 0.3
while counter < 1000:
    counter += 1
    
    # gradient value
    difference = cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR) - I_l
    grad = cv2.resize(difference, (height, width))

    # Update the value
    I_h = I_h - alpha*grad

# check the loss
e = np.square(I_l - cv2.resize(I_h, (height//4, width//4), interpolation = cv2.INTER_LINEAR))

# check the MSE and PSNR
R = np.amax(I_gt)
mse = np.divide(np.square(I_gt - I_h), height*width)
psnr = 10*np.log10(np.divide(np.square(R), mse))

print("MSE value: ", mse)
print("PSNR value: ", psnr)

# save the image
cv2.imwrite('/home/Computer_Vision_PA1/problem1.png', I_h)
