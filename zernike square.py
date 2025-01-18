 # -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:57:13 2023

@author: Sarita
"""
import numpy as np
import matplotlib.pyplot as plt 
a=np.linspace(-500,500,500)
#a=np.linspace(-960,960,960)
#b=np.linspace(-540,540,540)
x, y = np.meshgrid(a,a)

'''
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
z2=(0.9*Polar1)+(0.3*Polar2)+(0.60*Polar3)+(0.65*Polar4)+(0.15*Polar5)+(0.35*Polar6)
z3=(0.25*Polar1)+(0.35*Polar2)+(0.790*Polar3)+(0.5*Polar4)+(0.450*Polar5)+(0.75*Polar6)
z4=(0.29*Polar1)+(0.63*Polar2)+(0.360*Polar3)+(0.95*Polar4)+(0.55*Polar5)+(0.85*Polar6)
z5=(0.01*Polar1)+(0.9*Polar2)+(0.5*Polar3)+(0.01*Polar4)+(0.5*Polar5)+(0.99*Polar6)
from PIL import Image
import numpy as np
def save_matrix_as_image(matrix, filename):
    # Normalize matrix values to range from 0 to 255
    normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 255
    # Convert the normalized matrix to uint8 data type
    image_data = normalized_matrix.astype(np.uint8)
    # Create an image object from the matrix data
    image = Image.fromarray(image_data, mode='L')
    # Save the image to file
    image.save(filename)
# Example matrix (replace this with your own matrix)
matrix = z5  # Random values between 0 and 255
# Example filename
filename = "z6.png"
# Save the matrix as a grayscale image
save_matrix_as_image(matrix, filename)
f1=np.exp(1j*z1)
f2=np.exp(1j*z2)
f3=np.exp(1j*z3)
f4=np.exp(1j*z4)
m = np.zeros((500,500))
n = np.zeros((500,500))
o = np.zeros((500,500))
p = np.zeros((500,500))
m[150:250, 150:250] = 1
n[150:250, 250:350] = 1
o[250:350, 150:250] = 1
p[250:350, 250:350] = 1
'''
'''
f11 =  f1*m 
f12 =  f2*n 
f13 =  f3*o 
f14 =  f4*p 
'''
from PIL import Image
import numpy as np

img = Image.open('hex.png')
img_gray = img.convert('L')
img_array = np.array(img_gray)
 
g=np.sqrt(x**2+y**2<15625)
g0=g

#fft1 = img_array *g
#fft22 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fft1)))

plt.figure(1)
plt.imshow(np.abs(img_array), cmap='jet')
plt.axis('off')
'''
g=g*np.exp(1j*0.01*np.random.rand(500,500))

plt.figure(2)
plt.imshow(np.abs(fft22)/200, cmap='jet')
plt.colorbar()

plt.figure(3)
plt.imshow(np.angle(g), cmap='jet')

num_iterations = 100
for i in range(num_iterations):
    
    prev_g=g
    
    fft2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g)))

    swapped_fft = fft1 * np.exp(1j * np.angle(fft1))

    swapped_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(swapped_fft)))

    # Swap the images for the next iteration with fineup algorithm
    g = swapped_image*g0 + (prev_g-0.75*swapped_image)*(1-g)
  
       
plt.figure(3)
plt.imshow(np.angle(swapped_image)*g0 -np.amin(np.angle(swapped_image)*g0), cmap='jet')
plt.colorbar()
plt.title(f"Iteration {i + 1}")
plt.show() 

plt.figure(4)
plt.imshow((np.abs(swapped_image)*1000), cmap='jet')
plt.colorbar()
plt.title(f"Iteration {i + 1}")
plt.show() 
'''