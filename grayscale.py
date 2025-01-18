# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:22:37 2024

@author: sana kodakeri
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_grayscale_image(width, height):
   
    img = Image.new('L', (width, height))
 
    img_array = np.array(img)
    
    for y in range(height):
        for x in range(width):
            img_array[y][x] = int(65)
    
    img = Image.fromarray(img_array)
    return img

width = 1080
height = 1920
gray_img = create_grayscale_image(width, height)
gray_img.show()
plt.imshow(gray_img)
#plt.axis('off') # Remove axes
#plt.savefig('grayscale_image.png', bbox_inches='tight', pad_inches=8)

cv2.imwrite('array_image.png', gray_img)