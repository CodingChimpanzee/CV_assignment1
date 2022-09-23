import numpy as np
import matplotlib.pyplot as plt

def nan_remove(array):
    return array[~np.isnan(array)]

alpha_val_1 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/alpha_val.npy'))
alpha_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/alpha_val_2.npy'))

mse_val_1 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/mse_val.npy'))
mse_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/mse_val_2.npy'))

psnr_val_1 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/psnr_val.npy'))
psnr_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/psnr_val_2.npy'))

# Plotting part
# Problem 1
plt.figure(figsize = (15, 10))
plt.plot(alpha_val_1[:9500], mse_val_1[:9500], color = 'black')
plt.title("Tendency between MSE and step size (at 1000 iteration)", fontsize = 20)
plt.axhline(500, color='black')
plt.axvline(0, color='black')
plt.xlabel('Step size value', fontsize = 17)
plt.ylabel('MSE value', fontsize = 17)

plt.savefig("Problem1_MSE.png", dpi=300)

plt.figure(figsize = (15, 10))
plt.plot(alpha_val_1[:9500], psnr_val_1[:9500], color = 'black')
plt.title("Tendency between PSNR and step size (at 1000 iteration)", fontsize = 20)
plt.axhline(30, color='black')
plt.axvline(0, color='black')
plt.xlabel('Step size value', fontsize = 17)
plt.ylabel('PSNR value', fontsize = 17)

plt.savefig("Problem1_PSNR.png", dpi=300)

# Problem 2
plt.figure(figsize = (15, 10))
plt.plot(alpha_val_2[:8300], mse_val_2[:8300], color = 'black')
plt.title("Tendency between MSE and step size using prior (at 1000 iteration)", fontsize = 20)
plt.axhline(500, color='black')
plt.axvline(0, color='black')
plt.xlabel('Step size value', fontsize = 17)
plt.ylabel('MSE value', fontsize = 17)

plt.savefig("Problem2_MSE.png", dpi=300)

plt.figure(figsize = (15, 10))
plt.plot(alpha_val_2[:8300], psnr_val_2[:8300], color = 'black')
plt.title("Tendency between PSNR and step size using prior (at 1000 iteration)", fontsize = 20)
plt.axhline(30, color='black')
plt.axvline(0, color='black')
plt.xlabel('Step size value', fontsize = 17)
plt.ylabel('PSNR value', fontsize = 17)

plt.savefig("Problem2_PSNR.png", dpi=300)

