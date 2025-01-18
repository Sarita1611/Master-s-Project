# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:13:38 2024

@author: Sarita
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

z1=(0.5*Polar1)+(0.5*Polar2)+(0.90*Polar3)+(0.25*Polar4)+(0.50*Polar5)+(0.45*Polar6)
z2=(0.9*Polar1)+(0.3*Polar2)+(0.60*Polar3)+(0.65*Polar4)+(0.15*Polar5)+(0.35*Polar6)
z3=(0.25*Polar1)+(0.35*Polar2)+(0.790*Polar3)+(0.5*Polar4)+(0.450*Polar5)+(0.75*Polar6)
z4=(0.29*Polar1)+(0.63*Polar2)+(0.360*Polar3)+(0.95*Polar4)+(0.55*Polar5)+(0.85*Polar6)

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

g= m + n + o +p

g0=g

f11 =  f1*m 
f12 =  f2*n 
f13 =  f3*o 
f14 =  f4*p 

f = f11 + f12 +f13 +f14
plt.figure(2)
plt.imshow(np.angle(f)-np.amin(np.angle(f)), cmap='jet')
plt.colorbar()
'''
from PIL import Image

def collage_grayscale_images(image_paths, output_size):
    # Create a blank canvas for the collage
    collage = Image.new('L', output_size)

    # Calculate the width and height of each individual image in the collage
    width_per_image = output_size[0] // 2
    height_per_image = output_size[1] // 2

    # Paste each image onto the collage
    for i, image_path in enumerate(image_paths):
        # Open the image
        image = Image.open(image_path)

        # Resize the image to fit into the collage
        image = image.resize((width_per_image, height_per_image))

        # Calculate the position to paste the image
        row = i // 2
        col = i % 2
        position = (col * width_per_image, row * height_per_image)

        # Paste the resized image onto the collage at the appropriate position
        collage.paste(image, position)

    return collage

# Example usage
image_paths = ['z1 image.png', 'z2 image.png', 'z3 image.png', 'z4 image.png']  # Replace with your image paths
output_size = (1920,1080)  # Specify the size of the output collage

# Create the collage
collage = collage_grayscale_images(image_paths, output_size)

# Save or display the collage
collage.show()  # Display the collage
# collage.save('collage.jpg')  # Save the collage to a file

'''