# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:30:35 2023

@author: sana kodakeri
"""
import numpy as np
import matplotlib.pyplot as plt
a=np.linspace(-500,500,500)
x, y = np.meshgrid(a,a)

rho=np.sqrt(x**2+y**2)/62500
theta=np.arctan2(y,x)

Polar1=rho*np.cos(theta)
Polar2=rho*np.sin(theta)
Polar3=-1*rho**2
Polar4=rho**2*np.cos(2*theta)
Polar5=rho**2*np.sin(2*theta)
Polar6=rho*(-2+3*rho**2)*np.cos(theta)
Polar7=rho*(-2+3*rho**2)*np.sin(theta)
Polar8=1-6*rho**2+6*rho**4
Polar9=(rho**4)*np.cos(3*theta)
Polar10=(rho**4)*np.sin(3*theta)

z=(0.5*Polar1)+(0.5*Polar2)+(0.90*Polar3)+(0.25*Polar4)+(0.50*Polar5)+(0.45*Polar6)
z2=(0.9*Polar1)+(0.3*Polar2)+(0.60*Polar3)+(0.65*Polar4)+(0.15*Polar5)+(0.35*Polar6)

plt.figure(1)
plt.imshow(z, cmap='jet')

f1=np.exp(1j*z2)

g=np.sqrt(x**2+y**2<62500)

g0=g

plt.figure(2)
plt.imshow(np.abs(g), cmap='gray')

f=f1*g

plt.figure(3)
plt.imshow(np.angle(f), cmap='jet')
plt.colorbar()

fft1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))

g=g*np.exp(1j*0.01*np.random.rand(500,500))

plt.imshow(np.angle(g), cmap='jet')

num_iterations = 100

for i in range(num_iterations):
    prev_g=g
    
    fft2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g)))

    swapped_fft = np.abs(fft1) * np.exp(1j * np.angle(fft2))

    swapped_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(swapped_fft)))

    # Swap the images for the next iteration with fineup algorithm
    g = swapped_image*g0 + (prev_g-0.75*swapped_image)*(1-g)

 
plt.figure(4)
plt.imshow(np.angle(swapped_image)*g0, cmap='jet')
plt.colorbar()
plt.title(f"Iteration {i + 1}")
plt.axis('off')
plt.show()











