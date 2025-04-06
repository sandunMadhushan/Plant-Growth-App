import cv2
import numpy as np


def measure_plant_height(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    result_image = image.copy()
    height, width = image.shape[:2]

    # STEP 1: ENHANCED PLANT DETECTION
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create broader masks for different plant colors
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_yellow = np.array([15, 15, 15])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks
    plant_mask = cv2.bitwise_or(green_mask, yellow_mask)

    # Clean the mask (smaller kernel for thin seedlings)
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel_small)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel_large)

    # Find plant contours
    contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No plant detected!")
        return None

    # Find the plant contour
    plant_contour = max(contours, key=cv2.contourArea)

    # Get plant bounding rectangle
    x, y, w, h = cv2.boundingRect(plant_contour)

    # Check if this is a small seedling (for different processing)
    is_small_plant = h < height * 0.2

    # Get plant top (y-coordinate)
    top_y = y

    # Find plant center
    M = cv2.moments(plant_contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
    else:
        center_x = x + w // 2

    # STEP 2: SPECIALIZED SOIL DETECTION
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # For small seedlings in glass pots, specifically target the soil line
    if is_small_plant:
        # Look specifically at the top edge of soil (right where pot begins)
        soil_line_found = False

        # Use the region right below plant to search for soil
        search_start = y + h  # Bottom of plant
        search_end = min(search_start + 30, height)  # Only look 30px below plant

        # Method 1: Look for dark soil band
        for sy in range(search_start, search_end, 1):
            if sy >= height:
                continue

            # Check for soil color across a horizontal band
            soil_pixels = 0
            for sx in range(max(0, center_x - 30), min(width, center_x + 30)):
                if sy < height and sx < width:
                    pixel = image[sy, sx]
                    # Brownish soil detection (red > blue, moderate brightness)
                    if pixel[2] > pixel[0] and np.sum(pixel) < 450 and np.sum(pixel) > 50:
                        soil_pixels += 1

            # If enough soil pixels found in row
            if soil_pixels > 15:
                soil_y = sy
                soil_line_found = True
                break

        # Method 2: Look for horizontal edge (soil surface or pot rim)
        if not soil_line_found:
            max_edge_strength = 0
            strongest_edge_y = None

            for sy in range(search_start, search_end, 1):
                edge_row = edges[sy, max(0, center_x - 40):min(width, center_x + 40)]
                edge_strength = np.sum(edge_row)

                if edge_strength > max_edge_strength:
                    max_edge_strength = edge_strength
                    strongest_edge_y = sy

            if max_edge_strength > 300 and strongest_edge_y is not None:
                soil_y = strongest_edge_y
                soil_line_found = True

        # Method 3: Look for brightness transition
        if not soil_line_found:
            for sy in range(search_start, search_end, 1):
                if sy + 5 < height and sy - 5 >= 0:
                    above = np.mean(gray[sy - 5:sy, max(0, center_x - 20):min(width, center_x + 20)])
                    below = np.mean(gray[sy:sy + 5, max(0, center_x - 20):min(width, center_x + 20)])

                    if abs(above - below) > 20:
                        soil_y = sy
                        soil_line_found = True
                        break

        # Fallback: Use bottom of plant + small offset
        if not soil_line_found:
            soil_y = y + h + 5
    else:
        # For taller plants, use different soil detection approach
        soil_y = None

        # Start from midpoint of plant height
        search_start = y + h // 2
        search_end = min(y + h + 50, height - 10)

        # Look for soil using color and texture
        for sy in range(search_start, search_end, 2):
            if sy >= height - 5:
                continue

            # Sample window around center line
            window = image[sy:sy + 5, max(0, center_x - 20):min(width, center_x + 20)]

            # Calculate color properties
            if window.size > 0:
                avg_color = np.mean(window, axis=(0, 1))

                # Check for soil properties
                if avg_color[2] > avg_color[0] and np.sum(avg_color) < 500:
                    # Verify with texture check
                    texture_above = np.std(gray[max(0, sy - 10):sy, max(0, center_x - 20):min(width, center_x + 20)])
                    texture_below = np.std(
                        gray[sy:min(height, sy + 10), max(0, center_x - 20):min(width, center_x + 20)])

                    if abs(texture_above - texture_below) > 5:
                        soil_y = sy
                        break

        # Fallback for tall plants
        if soil_y is None:
            soil_y = y + h * 2 // 3

    # Ensure plant height is reasonable (with constraints)
    plant_height = soil_y - top_y

    # Sanity check - if height is unreasonable, use bounding box with offset
    min_reasonable = 10
    max_reasonable = height * 0.7

    if plant_height < min_reasonable or plant_height > max_reasonable:
        if is_small_plant:
            soil_y = y + h + 5  # Small offset for small plants
        else:
            soil_y = y + h * 2 // 3  # Proportional for tall plants
        plant_height = soil_y - top_y

    # Draw the measurement visualization
    cv2.line(result_image, (center_x, top_y), (center_x, soil_y), (0, 0, 255), 2)
    cv2.circle(result_image, (center_x, top_y), 5, (255, 0, 0), -1)
    cv2.circle(result_image, (center_x, soil_y), 5, (255, 0, 0), -1)

    # Add text with height information
    cv2.putText(result_image, f"Plant Height: {plant_height} pixels",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Resize for display
    display_scale = min(1, 800 / max(width, height))
    resized_image = cv2.resize(result_image, None,
                               fx=display_scale,
                               fy=display_scale,
                               interpolation=cv2.INTER_AREA)

    # Display result
    cv2.imshow("Plant Height", resized_image)
    print(f"Plant height: {plant_height} pixels")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return plant_height

# Example usage
measure_plant_height('Asset/Images/Selected images/height/2024_12_25_06PM.JPG')
