# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:40:23 2023

@author: Sarita
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('cat.jpg')
f= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(1)
plt.imshow(f, cmap='gray')

a=np.linspace(-500,500,500)
x,y=np.meshgrid(a,a)
g=np.sqrt(x**2+y**2<62500)
abs_g=np.abs(g)
plt.figure(2)
plt.imshow(g, cmap='gray')

f=f*g
g0 = g

plt.figure(3)
plt.imshow(f, cmap='gray')

fft1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))


num_iterations = 100

for i in range(num_iterations):
    print(f"Iteration {i + 1}")
    
    # Compute the Fourier transforms of the images
    prev_g = g
    
    fft2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g)))

    # Swap the magnitude components of the Fourier transforms
    swapped_fft = np.abs(fft1) * np.exp(1j * np.angle(fft2))

    # Inverse Fourier transform to get the swapped image
    swapped_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(swapped_fft)))

    swapped_image = np.real(swapped_image)      
    
    s = np.zeros((500,500))
    s[swapped_image>0] = 1
    s = s*g0


    # Swap the images for the next iteration
    g = swapped_image*s + (prev_g - 0.7*swapped_image) * (1-s)
    

    # Display the swapped image for this iteration
    plt.figure(4)
    plt.imshow(np.abs(swapped_image), cmap='gray')
    plt.title(f"Iteration {i + 1}")
    plt.axis('off')
    plt.show()