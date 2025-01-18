# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 23:25:26 2023

@author: sana kodakeri
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
a=np.linspace(-500,500,500)
x, y = np.meshgrid(a,a)
R = np.exp(1j*2*np.pi*(x + y))

O = np.zeros((500,500))
O[(np.abs(x) <= 100)*(np.abs(y) <= 100)] = np.pi/2
O = np.exp(1j*O)
f=image*O

# Display or save the resulting phase-shifted image

plt.figure(2)
plt.title('Phase')
plt.imshow(np.angle(f), cmap='gray')

plt.figure(1)
plt.imshow(np.abs(f), cmap='gray')

g=np.sqrt(x**2+y**2<62500)
abs_g=np.abs(g)
plt.figure(3)
plt.imshow(g, cmap='gray')
f=f*g
g0 = g

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