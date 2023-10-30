import cv2
import matplotlib.pyplot as plt

# Load the noisy image
noisy_image = cv2.imread('enbryos2.jpg')

# Apply Gaussian filter for denoising
denoised_image_gaussian = cv2.GaussianBlur(noisy_image, (7, 7), 1)

# Apply Median filter for denoising
denoised_image_median = cv2.medianBlur(noisy_image, 7)

# Apply Bilateral filter for denoising
denoised_image_bilateral = cv2.bilateralFilter(noisy_image, 10, 15, 15)

# Apply Non-Local Means denoising
denoised_image_nlm = cv2.fastNlMeansDenoisingColored(noisy_image, None, 15, 15, 10, 27)

# Create a figure with custom size
fig = plt.figure(figsize=(16, 8))

# Add the original noisy image
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_title('Noisy Image')
ax1.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
ax1.axis('off')

# Add denoised images with titles
filters = ['Gaussian', 'Median', 'Bilateral', 'NLM']
images = [denoised_image_gaussian, denoised_image_median, denoised_image_bilateral, denoised_image_nlm]

for i in range(4):
    ax = fig.add_subplot(2, 3, i + 2)
    ax.set_title(f'Denoised ({filters[i]})')
    ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    ax.axis('off')

# Add a common title for the entire figure
fig.suptitle('Comparison of Denoising Filters', fontsize=16)

# Save the figure with additional details
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
