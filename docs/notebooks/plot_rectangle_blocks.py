import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fft2
plt.ion()
plt.close('all')
plt.style.use("ggplot")


# Parameters
image_size = 512  # Size of the generated image
num_rectangles = 5  # Number of rectangles
rectangle_width = 30  # Width of each rectangle
spacing = 60  # Spacing between rectangles

# Create the image
image = np.zeros((image_size, image_size))

# Generate rectangle pattern
for i in range(num_rectangles):
    x = image_size // 2  # Center of the image
    y = image_size // 2
    left = x - rectangle_width // 2
    right = x + rectangle_width // 2
    top = y - rectangle_width // 2
    bottom = y + rectangle_width // 2
    image[top:bottom, left:right] = 1

    x += spacing
    left = x - rectangle_width // 2
    right = x + rectangle_width // 2
    image[top:bottom, left:right] = 1

    x -= 2 * spacing
    left = x - rectangle_width // 2
    right = x + rectangle_width // 2
    image[top:bottom, left:right] = 1

# Compute Fourier Transform
fourier_image = fftshift(fft2(image))

# Plot the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot the Fourier Transform
plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(fourier_image) + 1), cmap='gray')
plt.title('Fourier Domain')
plt.axis('off')


EXPERIMENT='symmetrical_rectangle_blocks'
plt.tight_layout()
plt.savefig(f'./figures/{EXPERIMENT}_exploration.png')
plt.close()
