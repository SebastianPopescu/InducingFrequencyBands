import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift, ifft2

# Parameters
image_size = 512  # Size of the generated image
num_rectangles = 5  # Number of rectangles
rectangle_width = 30  # Width of each rectangle
spacing = 60  # Spacing between rectangles

# Define the power spectral density (PSD)
def psd(fx, fy):
    fx_center = image_size // 2
    fy_center = image_size // 2

    psd_value = 0
    for _ in range(num_rectangles):
        psd_value += np.sinc((fx - fx_center) / rectangle_width) * np.sinc((fy - fy_center) / rectangle_width)
        fx_center += spacing

    return psd_value

# Generate the frequency domain representation (power spectral density)
freq_domain = np.fromfunction(np.vectorize(psd), (image_size, image_size))

# Compute the inverse Fourier Transform to get the time domain signal
time_domain = np.real(ifft2(fftshift(freq_domain)))

# Plot the power spectral density (PSD)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(freq_domain, cmap='gray')
plt.title('Power Spectral Density')
plt.axis('off')

# Plot the corresponding time domain signal
plt.subplot(1, 2, 2)
plt.plot(time_domain[image_size // 2, :])
plt.title('Time Domain')
plt.xlabel('Time')
plt.ylabel('Amplitude')

EXPERIMENT='symmetrical_rectangle_blocks_v2'
plt.tight_layout()
plt.savefig(f'./figures/{EXPERIMENT}_exploration.png')
plt.close()

