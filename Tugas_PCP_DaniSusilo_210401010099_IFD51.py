import numpy as np
import imageio.v2 as img
import matplotlib.pyplot as plt
from skimage import exposure, filters

# Load gambar
image = img.imread("d:/UNSIA/Semester 7/02-Pengolahan Citra/Sesi-01/Coding PC/source.jpg")

# normalisasi gambar jika diperlukan
if image.dtype == np.uint8:
    image = image.astype(np.float32) / 255.0

# ektrak setiap channel warna
red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]

# terapkan pemerataan histogram pada setiap channel
red_eq = exposure.equalize_hist(red)
green_eq = exposure.equalize_hist(green)
blue_eq = exposure.equalize_hist(blue)

# terapkan koreksi gamma untuk menonjolkan detail
red_eq = exposure.adjust_gamma(red_eq, gamma=0.8)
green_eq = exposure.adjust_gamma(green_eq, gamma=0.8)
blue_eq = exposure.adjust_gamma(blue_eq, gamma=0.8)

# Terapkan filter deteksi tepi untuk setiap saluran 
red_edges = filters.sobel(red_eq)
green_edges = filters.sobel(green_eq)
blue_edges = filters.sobel(blue_eq)

# Menormalkan hasil deteksi
red_edges = (red_edges - np.min(red_edges)) / (np.max(red_edges) - np.min(red_edges))
green_edges = (green_edges - np.min(green_edges)) / (np.max(green_edges) - np.min(green_edges))
blue_edges = (blue_edges - np.min(blue_edges)) / (np.max(blue_edges) - np.min(blue_edges))

# Plotting
plt.figure(figsize=(10, 12))

# Plot original image
plt.subplot(4, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Plot red channel 
plt.subplot(4, 3, 4)
plt.imshow(np.stack([red_eq, np.zeros_like(red_eq), np.zeros_like(red_eq)], axis=-1))  # Only red channel
plt.title('Red Channel (Enhanced Contrast)')
plt.axis('off')

plt.subplot(4, 3, 5)
plt.imshow(np.stack([red_edges, np.zeros_like(red_edges), np.zeros_like(red_edges)], axis=-1))  # Only red edges
plt.title('Red Channel (Edge Detection)')
plt.axis('off')

# Plot green channel 
plt.subplot(4, 3, 7)
plt.imshow(np.stack([np.zeros_like(green_eq), green_eq, np.zeros_like(green_eq)], axis=-1))  # Only green channel
plt.title('Green Channel (Enhanced Contrast)')
plt.axis('off')

plt.subplot(4, 3, 8)
plt.imshow(np.stack([np.zeros_like(green_edges), green_edges, np.zeros_like(green_edges)], axis=-1))  # Only green edges
plt.title('Green Channel (Edge Detection)')
plt.axis('off')

# Plot blue channel
plt.subplot(4, 3, 10)
plt.imshow(np.stack([np.zeros_like(blue_eq), np.zeros_like(blue_eq), blue_eq], axis=-1))  # Only blue channel
plt.title('Blue Channel (Enhanced Contrast)')
plt.axis('off')

plt.subplot(4, 3, 11)
plt.imshow(np.stack([np.zeros_like(blue_edges), np.zeros_like(blue_edges), blue_edges], axis=-1))  # Only blue edges
plt.title('Blue Channel (Edge Detection)')
plt.axis('off')

plt.tight_layout()
plt.show()
