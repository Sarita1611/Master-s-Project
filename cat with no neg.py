# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 00:16:54 2023

@author: sana kodakeri
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

f = np.zeros((500, 500))
f[245:255, :] = 1
f[:, 245:255] = 1
plt.figure(1)
plt.imshow(f, cmap='gray')

a=np.linspace(-500,500,500)
x,y=np.meshgrid(a,a)
g=np.sqrt(x**2+y**2<62500)
abs_g=np.abs(g)
plt.figure(2)
plt.imshow(g, cmap='gray')

f=f*g

plt.figure(3)
plt.imshow(f, cmap='gray')

fft1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))


num_iterations = 200

for i in range(num_iterations):
    print(f"Iteration {i + 1}")

    # Compute the Fourier transforms of the images
  
    fft2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g)))

    # Swap the magnitude components of the Fourier transforms
    swapped_fft = np.abs(fft1) * np.exp(1j * np.angle(fft2))

    # Inverse Fourier transform to get the swapped image
    swapped_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(swapped_fft)))

    swapped_image = np.real(swapped_image)
    
    s = np.zeros((500,500))
    s[swapped_image>0] = 1


    # Swap the images for the next iteration
    g = abs_g*swapped_image*s
    

    # Display the swapped image for this iteration
    plt.figure(4)
    plt.imshow(np.abs(swapped_image), cmap='gray')
    plt.title(f"Iteration {i + 1}")
    plt.axis('off')
    plt.show()