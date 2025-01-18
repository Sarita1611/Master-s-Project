# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 18:53:31 2023

@author: sana kodakeri
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as io
import autograd.numpy as anp
from autograd import grad
#%matplotlib widget
def TV(a):
    ux, uy = anp.gradient(a)
    g = anp.sqrt(anp.abs(ux)**2 + anp.abs(uy)**2 + 1e-10)
    s = anp.sum(g[:])
    return s
def data_error(guess, blurred, h):
    G = anp.fft.fftshift(anp.fft.fft2(anp.fft.ifftshift(guess)))
    H = anp.fft.fftshift(anp.fft.fft2(anp.fft.ifftshift(h)))
    err = blurred - anp.real(anp.fft.fftshift(anp.fft.ifft2(anp.fft.ifftshift(G*H))))
    err2 = err**2
    l2err = anp.sum(err2[:])
    return l2err   
# Read a test image

img = io.imread('boat512.tiff')
plt.figure(1)
plt.imshow(img, cmap='gray')
plt.show()
x0 = np.linspace(-255.5, 255.5, 512, endpoint=True)
x, y = np.meshgrid(x0,x0)
# Define a blur function on the grid

h=[(np.abs(x)<5)*(np.abs(y)<5)] # Square window psf
h = np.exp(-(x**2 + y**2)/10)  # Gaussian psf extending everywhere
h = np.sinc(x/5) * np.sinc(y/5)  # sinc blur function

h = h / np.sum(h[:])
plt.figure(2)
plt.imshow(h, cmap='gray')
plt.show()
# Blurred image

A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
H = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(h)))
b = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A*H))))

# Add noise
noise = np.random.rand(512,512) * 0.001 * np.mean(b[:])
b = b + noise

plt.figure(3)
plt.imshow(b, cmap='gray')
plt.show()
# Wiener filter recovery
NSR = 0.001
W = np.conj(H) / (np.abs(H)**2 + NSR)

# Deblur using Wiener filter
B = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(b)))
b1 = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(B*W))))

plt.figure(4)
plt.imshow(b1, cmap='gray')
plt.show()
# Iterative TV based deconvolution ... starting with Winer filter solution as first guess

gerr = grad(data_error, 0)
gtv = grad(TV, 0)

alpha = 1
guess = b1
t = 0.1

for iter in np.arange(1,11,1):
    
    C1 = data_error(guess, b, h)
    dC1 = gerr(guess, b, h)
    C2 = TV(guess)
    dC2 = gtv(guess)
    C = C1 + alpha*C2
    dC = dC1 + alpha*dC2
    dC = dC / np.linalg.norm(dC)
    
    donet = 0
    
    while(not donet):
        guess_try = guess - t* np.linalg.norm(guess) * dC
        C1try = data_error(guess_try, b, h)
        C2try = TV(guess_try)
        Ctry = C1try + alpha*C2try
        
        if(Ctry < C):
            donet = 1
            guess = guess_try
        else:
            t = t/2
            
    if(t < 1e-6):
        break
    else:
        t = 0.1


plt.figure(5)
plt.imshow(guess, cmap='gray')
plt.show()