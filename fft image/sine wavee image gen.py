import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image dimensions
width, height = 512, 512

# Create a blank canvas for the original image
original_image = np.zeros((height, width), dtype=np.uint8)

# Define sine wave parameters for the original image
frequency_x = 7  # Adjust the frequency in the x-direction
frequency_y = 5  # Adjust the frequency in the y-direction
amplitude = 127  # Adjust the amplitude of the sine wave

# Generate the original image with vertical sine waves
for x in range(width):
    for y in range(height):
        # Calculate the pixel value using a sine wave
        pixel_value = amplitude * np.cos(2 * np.pi * frequency_x * x / width) + amplitude * np.cos(2 * np.pi * frequency_y * y / height)

        # Set the pixel value in the original image
        original_image[y, x] = int(pixel_value)

# Create a function to apply filters
def apply_filter(filter_type, image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    if filter_type == 'low-pass':
        # Apply a low-pass filter
        crow, ccol = height // 2, width // 2
        d = 30  # Adjust the cut-off distance
        f_transform_shifted[crow - d:crow + d, ccol - d:ccol + d] = 0
    elif filter_type == 'high-pass':
        # Apply a high-pass filter
        crow, ccol = height // 2, width // 2
        d = 30  # Adjust the cut-off distance
        f_transform_shifted[crow - d:crow + d, ccol - d:ccol + d] = f_transform_shifted[crow - d:crow + d, ccol - d:ccol + d]
    elif filter_type == 'band-pass':
        # Apply a band-pass filter
        crow, ccol = height // 2, width // 2
        d = 30  # Adjust the cut-off distance
        f_transform_shifted[:crow - d, :] = 0
        f_transform_shifted[crow + d:, :] = 0

    # Inverse FFT to get the filtered image
    filtered_image = np.fft.ifftshift(f_transform_shifted)
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.abs(filtered_image)

    return filtered_image

# Allow the user to select a filter type
print("Select a filter type:")
print("1. Low-pass")
print("2. High-pass")
print("3. Band-pass")
filter_type_choice = input("Enter the number (1/2/3): ")

if filter_type_choice == '1':
    filter_type = 'low-pass'
elif filter_type_choice == '2':
    filter_type = 'high-pass'
elif filter_type_choice == '3':
    filter_type = 'band-pass'
else:
    print("Invalid choice. Using low-pass filter by default.")
    filter_type = 'low-pass'

# Apply the selected filter
filtered_image = apply_filter(filter_type, original_image)

# Perform FFT on the original image
f_transform_original = np.fft.fft2(original_image)
f_transform_shifted_original = np.fft.fftshift(f_transform_original)

# Calculate the magnitude spectrum of the original image
magnitude_spectrum_original = np.log(np.abs(f_transform_shifted_original) + 1)

# Display the original image's FFT magnitude spectrum and the filtered image
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(original_image, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(132), plt.imshow(magnitude_spectrum_original, cmap='gray')
plt.title('FFT Magnitude Spectrum (Original)'), plt.axis('off')
plt.subplot(133), plt.imshow(filtered_image, cmap='gray')
plt.title(f'{filter_type.capitalize()} Filtered Image'), plt.axis('off')
plt.show()
