import numpy as np
import matplotlib.pyplot as plt

def nan_remove(array):
    return array[~np.isnan(array)]

iter_val_1 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/iter_val_1.npy'))
iter_mse_val_1 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/iter_mse_val_1.npy'))
iter_psnr_val_1 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/iter_psnr_val_1.npy'))

iter_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/iter_val_2.npy'))
iter_mse_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/iter_mse_val_2.npy'))
iter_psnr_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/iter_psnr_val_2.npy'))


# Problem 1 plot
plt.figure(figsize = (10, 7))
plt.plot(iter_val_1[:1000], iter_mse_val_1[:1000], color = 'blue')
plt.title("Tendency between MSE and step size using prior (at 1000 iteration)", fontsize = 20)
plt.axhline(500, color='black')
plt.axvline(0, color='black')
plt.xlabel('Step size value', fontsize = 17)
plt.ylabel('MSE value', fontsize = 17)

plt.savefig("Problem1_ITER_MSE.png", dpi=300)

plt.figure(figsize = (10, 7))
plt.plot(iter_val_1[:1000], iter_psnr_val_1[:1000], color = 'red')
plt.title("Tendency between PSNR and step size using prior (at 1000 iteration)", fontsize = 20)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('Step size value', fontsize = 17)
plt.ylabel('PSNR value', fontsize = 17)

plt.savefig("Problem1_ITER_PSNR.png", dpi=300)

# Problem 2 plot
plt.figure(figsize = (10, 7))
plt.plot(iter_val_2[:1000], iter_mse_val_2[:1000], color = 'blue')
plt.title("Tendency between MSE and step size using prior (at 1000 iteration)", fontsize = 20)
plt.axhline(500, color='black')
plt.axvline(0, color='black')
plt.xlabel('Step size value', fontsize = 17)
plt.ylabel('MSE value', fontsize = 17)

plt.savefig("Problem2_ITER_MSE.png", dpi=300)

plt.figure(figsize = (10, 7))
plt.plot(iter_val_2[:1000], iter_psnr_val_2[:1000], color = 'red')
plt.title("Tendency between PSNR and step size using prior (at 1000 iteration)", fontsize = 20)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('Step size value', fontsize = 17)
plt.ylabel('PSNR value', fontsize = 17)

plt.savefig("Problem2_ITER_PSNR.png", dpi=300)