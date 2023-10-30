import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (replace 'your_image.jpg' with the path to your image)
image = cv2.imread('cube.jpg', cv2.IMREAD_GRAYSCALE)

# Create a function to apply filters
def apply_filter(filter_type, image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    if filter_type == 'low-pass':
        # Apply a low-pass filter
        crow, ccol = image.shape[0] // 2, image.shape[1] // 2
        d = 30  # Adjust the cut-off distance 
        f_transform_shifted[crow - d:crow + d, ccol - d:ccol + d] = 0
    elif filter_type == 'high-pass':
        # Apply a high-pass filter
        crow, ccol = image.shape[0] // 2, image.shape[1] // 2
        d = 1  # Adjust the cut-off distance
        f_transform_shifted[crow - d:crow + d, ccol - d:ccol + d] = f_transform_shifted[crow - d:crow + d, ccol - d:ccol + d]
    elif filter_type == 'band-pass':
        # Apply a band-pass filter
        crow, ccol = image.shape[0] // 2, image.shape[1] // 2
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
filtered_image = apply_filter(filter_type, image)

# Perform FFT on the original image
f_transform_original = np.fft.fft2(image)
f_transform_shifted_original = np.fft.fftshift(f_transform_original)

# Calculate the magnitude spectrum of the original image
magnitude_spectrum_original = np.log(np.abs(f_transform_shifted_original) + 1)

# Display the original image, its FFT magnitude spectrum, and the filtered image
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(132), plt.imshow(magnitude_spectrum_original, cmap='gray')
plt.title('FFT Magnitude Spectrum (Original)'), plt.axis('off')
plt.subplot(133), plt.imshow(filtered_image, cmap='gray')
plt.title(f'{filter_type.capitalize()} Filtered Image'), plt.axis('off')
plt.tight_layout()
plt.show()
