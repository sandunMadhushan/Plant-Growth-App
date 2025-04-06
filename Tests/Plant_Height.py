import cv2
import numpy as np

# Load the image
image = cv2.imread('Tests/plant_image.jpg')

# Create a copy for display
result_image = image.copy()

# Convert to HSV color space for better green detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for green color to detect only the plant
lower_green = np.array([25, 30, 30])
upper_green = np.array([90, 255, 255])
plant_mask = cv2.inRange(hsv, lower_green, upper_green)

# Clean the mask
kernel = np.ones((3, 3), np.uint8)
plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)

# Find plant contours
plant_contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(plant_contours) > 0:
    # Find the largest plant contour
    largest_contour = max(plant_contours, key=cv2.contourArea)

    # Get the extreme points of the plant
    contour_points = largest_contour.reshape(-1, 2)
    top_y = np.min(contour_points[:, 1])  # Topmost point of plant

    # Find where the stem meets the soil
    # Convert to grayscale and threshold to separate soil
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, soil_thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    # Get the x-coordinate at the middle of the plant contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
    else:
        cx = int(np.mean(contour_points[:, 0]))

    # Create a vertical line from the top of the plant downward
    height, width = image.shape[:2]
    bottom_y = None

    # Scan downward from the top to find where the stem meets the soil
    # Use a small window around the center of the plant
    window_width = 20
    for y in range(top_y, height):
        # Sample pixels in a horizontal window at this y-coordinate
        x_start = max(0, cx - window_width // 2)
        x_end = min(width, cx + window_width // 2)
        window = plant_mask[y, x_start:x_end]

        # If we've reached a point where there's no more plant (below the stem)
        if np.sum(window) == 0:
            # Look at the original image to determine if we've reached soil
            color = image[y, cx]
            # Check for brownish color (soil)
            if color[0] < 100 and color[1] < 100 and color[2] < 100:
                bottom_y = y
                break

    # If we couldn't detect the soil automatically, use the bottom of the plant contour
    if bottom_y is None:
        bottom_y = np.max(contour_points[:, 1])

    # Calculate plant height in pixels
    plant_height = bottom_y - top_y

    # Draw the plant height visualization
    cv2.line(result_image, (cx, top_y), (cx, bottom_y), (0, 0, 255), 2)
    cv2.circle(result_image, (cx, top_y), 5, (255, 0, 0), -1)
    cv2.circle(result_image, (cx, bottom_y), 5, (255, 0, 0), -1)

    # Add text with plant height
    cv2.putText(result_image, f"Height: {plant_height} pixels",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 3)

    # Resize for display
    display_scale = 0.2
    display_image = cv2.resize(result_image, None, fx=display_scale, fy=display_scale,
                               interpolation=cv2.INTER_AREA)

    # Show the result
    cv2.imshow("Plant Height", display_image)
    print(f"Plant height in pixels: {plant_height}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No plant detected!")
