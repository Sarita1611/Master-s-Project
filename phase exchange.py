# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:53:59 2023

@author: sana kodakeri
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
img=cv2.imread('sarita.jpg')
img2=cv2.imread('SANA TEST.jpg')
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
plt.figure(1)

plt.imshow(img2,cmap='gray')
plt.figure(2)
plt.imshow(img,cmap='gray')
fimg1=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gimg)))
fimg2=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gimg2)))
plt.figure(3)

plt.imshow(np.abs(fimg1),cmap='jet')
plt.figure(4)
plt.imshow(np.abs(fimg2),cmap='jet')
plt.figure(5)
plt.imshow(np.angle(fimg1),cmap='gray')
plt.figure(6)
plt.imshow(np.angle(fimg2),cmap='gray')       
b1=np.abs(fimg1)*np.exp(1j*np.angle(fimg2))
b2=np.abs(fimg2)*np.exp(1j*np.angle(fimg1))
c1=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(b1)))
c2=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(b2)))
plt.figure(7)

plt.imshow(np.abs(c1),cmap='gray')
plt.figure(8)
plt.imshow(np.abs(c2),cmap='gray')
