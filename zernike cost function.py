# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:54:55 2024

@author: sana kodakeri
"""

import numpy as np
import matplotlib.pyplot as plt 
import autograd.numpy as anp
from autograd import grad

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

z1=(0.5*Polar1)+(0.5*Polar2)+(0.90*Polar3)+(0.25*Polar4)+(0.50*Polar5)+(0.45*Polar6)
f1=np.exp(1j*z1)

plt.figure(1)
plt.imshow(z1, cmap='jet')

m = np.zeros((500,500))
m[150:250, 150:250] = 1

f =  f1*m 
plt.figure(2)
plt.imshow(np.angle(f)-np.amin(np.angle(f)), cmap='jet')
plt.colorbar()

k= np.angle(f)

fft1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))

g=m
g0=g

g=g*np.exp(1j*0.01*np.random.rand(500,500))

plt.figure(3)
plt.imshow(np.abs(g), cmap='gray')

num_iterations = 100

for i in range(num_iterations):
    
    prev_g=g
    
    fft2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g)))

    swapped_fft = np.abs(fft1) * np.exp(1j * np.angle(fft2))

    swapped_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(swapped_fft)))

    # Swap the images for the next iteration with fineup algorithm
    g = swapped_image*g0 + (prev_g-0.75*swapped_image)*(1-g)
   
    # Display the swapped image for this iteration
  
       
plt.figure(4)
plt.imshow(np.angle(swapped_image)*g0 -np.amin(np.angle(swapped_image)*g0), cmap='jet')
plt.colorbar()
plt.title(f"Iteration {i + 1}")
plt.axis('off')
plt.show() 

j=np.angle(swapped_image)*g0 



def data_error(guess, blurred, h):
    G = anp.fft.fftshift(anp.fft.fft2(anp.fft.ifftshift(guess)))
    H = anp.fft.fftshift(anp.fft.fft2(anp.fft.ifftshift(h)))
    err = blurred - anp.real(anp.fft.fftshift(anp.fft.ifft2(anp.fft.ifftshift(G*H))))
    err2 = err**2
    l2err = anp.sum(err2[:])
    return l2err   

gerr = anp.grad(data_error, 0)

guess = j
t = 0.001



"""


for iter in np.arange(1,11,1):
    
    C = data_error(guess, k, g0)
    dC = gerr(guess, k, g0)
    dC = dC / np.linalg.norm(dC)
    donet = 0
    
    while(not donet):
        guess_try = guess - t* np.linalg.norm(guess) * dC
        Ctry = data_error(guess_try, f, g0)
        
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

"""