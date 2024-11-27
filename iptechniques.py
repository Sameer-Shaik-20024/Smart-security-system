#Importing necessary libraries
import cv2
import numpy as np


# Function to apply a median filter to reduce noise in the image
# `kernel_size` determines the size of the filter
def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

#Function to apply a high-pass filter to enhance edges and details
# The kernel emphasizes edges by subtracting surrounding pixel values
def apply_high_pass_filter(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel) # Apply kernel using filter2D

# Function to apply histogram equalization to enhance contrast
# If the image is in color, only the luminance (Y) channel is equalized
def apply_histogram_equalization(image):
    if len(image.shape) == 3:
	# Convert image from BGR to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb) # Split into Y, Cr, and Cb channels
        cv2.equalizeHist(channels[0], channels[0]) # Equalize the Y (luminance) channel
        cv2.merge(channels, ycrcb) # Merge channels back
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR) # Convert back to BGR
    else: # Grayscale image
        return cv2.equalizeHist(image)

# Function to apply edge detection using the Canny algorithm
# Detects edges by identifying areas of high intensity gradient
def apply_edge_detection(image):
    return cv2.Canny(image, 100, 200) # Use thresholds of 100 and 200

# Function to apply the Sobel filter for edge detection
# Computes gradients in the x and y directions and combines them
def apply_sobel_filter(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) # Gradient in x-direction
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) # Gradient in y-direction
    return cv2.magnitude(sobelx, sobely) # Magnitude of gradients for edge strength

# Function to apply unsharp masking to sharpen the image
# Enhances the image by subtracting a blurred version from the original
def apply_unsharp_masking(image):
    gaussian = cv2.GaussianBlur(image, (5,5), 0) # Create a blurred version of the image
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0) # Combine original and blurred images

# Function to apply a specified filter to an image based on the filter type
def apply_filter(image_path, filter_type):
    # Read the image from the provided file path
    image = cv2.imread(image_path)
    if image is None: # Check if the image was loaded successfully
        return None

    # Apply the selected filter based on the filter_type argument    
    if filter_type == 'median':
        return apply_median_filter(image)
    elif filter_type == 'highpass':
        return apply_high_pass_filter(image)
    elif filter_type == 'histogram':
        return apply_histogram_equalization(image)
    elif filter_type == 'edge':
        return apply_edge_detection(image)
    elif filter_type == 'sobel':
        return apply_sobel_filter(image)
    elif filter_type == 'unsharp':
        return apply_unsharp_masking(image)
    else:
        return image # Return the original image if no filter type matches

# Function to process an image by applying all filters and saving the results
def process(image_path):
    # Apply all filters and save the results
    filters = ['median', 'highpass', 'histogram', 'edge', 'sobel', 'unsharp']
    for filter_type in filters:
	# Apply each filter to the image
        processed_image = apply_filter(image_path, filter_type)
        if processed_image is not None: # Check if the filter was applied successfully
	    # Save the processed image with a corresponding filter name
            cv2.imwrite(f'processed_images/{filter_type}_filtered.jpg', processed_image)
