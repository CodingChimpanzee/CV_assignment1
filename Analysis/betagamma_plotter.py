import numpy as np
import matplotlib.pyplot as plt

def nan_remove(array):
    return array[~np.isnan(array)]

beta_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/beta_val_2.npy'))
beta_mse_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/beta_mse_val_2.npy'))
beta_psnr_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/beta_psnr_val_2.npy'))

gamma_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/gamma_val_2.npy'))
gamma_mse_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/gamma_mse_val_2.npy'))
gamma_psnr_val_2 = nan_remove(np.load('/home/Computer_Vision_PA1/Analysis/gamma_psnr_val_2.npy'))

# Problem 2 beta
plt.figure(figsize = (12, 7))
plt.plot(beta_val_2[:1000], beta_mse_val_2[:1000], color = 'blue')
plt.title("Tendency between MSE and beta value (step size = 0.002, iteraion 1000 times)", fontsize = 20)
plt.axhline(500, color='black')
plt.axvline(0, color='black')
plt.xlabel('Gamma value', fontsize = 17)
plt.ylabel('MSE value', fontsize = 17)

plt.savefig("Problem2_beta_MSE.png", dpi=300)

plt.figure(figsize = (12, 7))
plt.plot(beta_val_2[:1000], beta_psnr_val_2[:1000], color = 'red')
plt.title("Tendency between PSNR and beta value(step size = 0.002, iteraion 1000 times)", fontsize = 20)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('Gamma value', fontsize = 17)
plt.ylabel('PSNR value', fontsize = 17)

plt.savefig("Problem2_beta_PSNR.png", dpi=300)

# Problem 2 gamma
plt.figure(figsize = (12, 7))
plt.plot(gamma_val_2[:1000], gamma_mse_val_2[:1000], color = 'blue')
plt.title("Tendency between MSE and gamma value (step size = 0.002, iteraion 1000 times)", fontsize = 20)
plt.axhline(500, color='black')
plt.axvline(0, color='black')
plt.xlabel('Gamma value', fontsize = 17)
plt.ylabel('MSE value', fontsize = 17)

plt.savefig("Problem2_gamma_MSE.png", dpi=300)

plt.figure(figsize = (12, 7))
plt.plot(gamma_val_2[:1000], gamma_psnr_val_2[:1000], color = 'red')
plt.title("Tendency between PSNR and gamma value(step size = 0.002, iteraion 1000 times)", fontsize = 20)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('Gamma value', fontsize = 17)
plt.ylabel('PSNR value', fontsize = 17)

plt.savefig("Problem2_gamma_PSNR.png", dpi=300)