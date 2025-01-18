# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:48:10 2023

@author: Sarita
"""

import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

def mse(image1, image2):
    err = np.sum((image1 - image2) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

# Load the pixelated images
image1 = cv2.imread('combined input.png')
image2 = cv2.imread('combined result.png')

# Convert images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute Structural Similarity Index (SSI)
ssi_index, _ = ssim(gray_image1, gray_image2, full=True)

# Compute Mean Squared Error (MSE)
mse_value = mse(gray_image1, gray_image2)

print(f"Structural Similarity Index: {ssi_index}")
print(f"Mean Squared Error: {mse_value}")
