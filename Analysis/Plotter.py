import numpy as np
import matplotlib.pyplot as plt

def nan_remove(array):
    return array[~np.isnan(array)]

alpha_val_1 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/alpha_val.npy'))
alpha_val_1_small = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/alpha_val_small.npy'))
alpha_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/alpha_val_big_2.npy'))
alpha_val_2_small = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/alpha_val_small_2.npy'))

mse_val_1 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/mse_val.npy'))
mse_val_1_small = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/mse_val_small.npy'))
mse_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/mse_val_big_2.npy'))
mse_val_2_small = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/mse_val_small_2.npy'))

psnr_val_1 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/psnr_val.npy'))
psnr_val_1_small = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/psnr_val_small.npy'))
psnr_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/psnr_val_big_2.npy'))
psnr_val_2_small = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/psnr_val_small_2.npy'))

# Plotting part
# Problem 1 (9500)
# plt.figure(figsize = (10, 7))
# plt.plot(alpha_val_1[:9504], mse_val_1[:9504], color = 'blue')
# plt.title("Tendency between MSE and step size (at 1000 iteration)", fontsize = 20)
# plt.axhline(500, color='black')
# plt.axvline(0, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('MSE value', fontsize = 17)

# plt.savefig("Problem1_MSE.png", dpi=300)

# plt.figure(figsize = (10, 7))
# plt.plot(alpha_val_1[:9504], psnr_val_1[:9504], color = 'red')
# plt.title("Tendency between PSNR and step size (at 1000 iteration)", fontsize = 20)
# plt.axhline(0, color='black')
# plt.axvline(0, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('PSNR value', fontsize = 17)

# plt.savefig("Problem1_PSNR.png", dpi=300)

# # Problem 2 (8300)
# plt.figure(figsize = (10, 7))
# plt.plot(alpha_val_2[:1925], mse_val_2[:1925], color = 'blue')
# plt.title("Tendency between MSE and step size using prior (at 1000 iteration)", fontsize = 20)
# plt.axhline(500, color='black')
# plt.axvline(0, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('MSE value', fontsize = 17)
# plt.axis([0, 2.4, 500, 1000])

# plt.savefig("Problem2_MSE.png", dpi=300)

# plt.figure(figsize = (10, 7))
# plt.plot(alpha_val_2[:1925], psnr_val_2[:1925], color = 'red')
# plt.title("Tendency between PSNR and step size using prior (at 1000 iteration)", fontsize = 20)
# plt.axhline(0, color='black')
# plt.axvline(0, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('PSNR value', fontsize = 17)
# plt.axis([0, 2.4, -1000, 100])

# plt.savefig("Problem2_PSNR.png", dpi=300)

# problem 1 small value
# plt.figure(figsize = (10, 7))
# plt.plot(alpha_val_1_small[:100], mse_val_1_small[:100], color = 'blue')
# plt.title("Tendency between MSE and step size (at 1000 iteration)", fontsize = 20)
# plt.axhline(500, color='black')
# plt.axvline(0, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('MSE value', fontsize = 17)

# plt.savefig("Problem1_MSE.png", dpi=300)

# plt.figure(figsize = (10, 7))
# plt.plot(alpha_val_1_small[:100], psnr_val_1_small[:100], color = 'red')
# plt.title("Tendency between PSNR and step size (at 1000 iteration)", fontsize = 20)
# plt.axhline(0, color='black')
# plt.axvline(0, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('PSNR value', fontsize = 17)

# plt.savefig("Problem1_PSNR.png", dpi=300)

# Problem 2 small value
# plt.figure(figsize = (10, 7))
# plt.plot(alpha_val_2_small[:1000], mse_val_2_small[:1000], color = 'blue')
# plt.title("Tendency between MSE and step size using prior (at 1000 iteration)", fontsize = 20)
# plt.axhline(500, color='black')
# plt.axvline(0, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('MSE value', fontsize = 17)

# plt.savefig("Problem2_MSE.png", dpi=300)

# plt.figure(figsize = (10, 7))
# plt.plot(alpha_val_2_small[:1000], psnr_val_2_small[:1000], color = 'red')
# plt.title("Tendency between PSNR and step size using prior (at 1000 iteration)", fontsize = 20)
# plt.axhline(0, color='black')
# plt.axvline(0, color='black')
# plt.xlabel('Step size value', fontsize = 17)
# plt.ylabel('PSNR value', fontsize = 17)

# plt.savefig("Problem2_PSNR.png", dpi=300)