# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:53:34 2023

@author: Sarita
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('cat.jpg')
f= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

a=np.linspace(-500,500,500)
x,y=np.meshgrid(a,a)
g=np.sqrt(x**2+y**2<62500)
abs_g=np.abs(g)
plt.figure(2)
plt.imshow(g, cmap='gray')
f=f*g

fft1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))


num_iterations = 400


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
    g = abs_g*swapped_image*s + 0.07*swapped_image*(s-1)
    

    # Display the swapped image for this iteration
    plt.figure(3)
    plt.imshow(np.abs(swapped_image), cmap='gray')
    plt.title(f"Iteration {i + 1}")
    plt.show()
 









