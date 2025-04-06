import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1 = cv2.imread('Tests/first_image.jpg')  # Beginning of week
image2 = cv2.imread('Tests/second_image.jpg')  # End of week

# Convert images to HSV color space for better color segmentation
hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

# Define green color range
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create masks for green areas
mask1 = cv2.inRange(hsv1, lower_green, upper_green)
mask2 = cv2.inRange(hsv2, lower_green, upper_green)

# Calculate green pixel area
green_area1 = cv2.countNonZero(mask1)
green_area2 = cv2.countNonZero(mask2)

# Calculate growth percentage - measures increase in green pixels
growth_percent = ((green_area2 - green_area1) / max(1, green_area1)) * 100

# Calculate health percentage based on green intensity and growth
# Get average intensity of green pixels in the plant
green_intensity = np.mean(hsv2[mask2 > 0, 1]) if np.sum(mask2) > 0 else 0
max_intensity = 255  # Maximum possible intensity

# Calculate intensity percentage
intensity_percent = (green_intensity / max_intensity) * 100

# Calculate health percentage (60% based on growth, 40% based on color intensity)
capped_growth = min(max(growth_percent, 0), 100)  # Limit growth to 0-100 range
health_percent = (0.6 * capped_growth) + (0.4 * intensity_percent)

# Ensure health is between 0-100%
health_percent = min(max(health_percent, 0), 100)

# Determine health status
if health_percent > 50:
    health_status = "Healthy"
    color = (0, 255, 0)  # Green text
else:
    health_status = "Unhealthy"
    color = (0, 0, 255)  # Red text

# Create a copy of the second image to add annotations
measured_image = image2.copy()

# Add health and growth information
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(measured_image, f'Health: {health_status} ({health_percent:.1f}%)', (10, 70), font, 2.5, color, 4)
# cv2.putText(measured_image, f'Growth: {growth_percent:.1f}%', (10, 100), font, 2.0, color, 4)

# Function to resize images for display
def resize_for_display(image, scale_percent=20):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Create visualization of green areas detected
green_detection = cv2.bitwise_and(image2, image2, mask=mask2)

# Resize all images before display
display_image = resize_for_display(image2)
display_measured = resize_for_display(measured_image)
display_green = resize_for_display(green_detection)

# Display the resized images
cv2.imshow('Original Plant Image', display_image)
cv2.imshow('Plant Health Measurement', display_measured)
cv2.imshow('Green Area Detection', display_green)

cv2.waitKey(0)
cv2.destroyAllWindows()
