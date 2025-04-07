import os
import cv2
import numpy as np
import datetime
import re
import matplotlib.pyplot as plt

def measure_plant_height(image_path, display=True):
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

    # Clean the mask
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel_small)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel_large)

    # Find plant contours
    contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No plant detected in {image_path}!")
        return None

    # Find the plant contour
    plant_contour = max(contours, key=cv2.contourArea)

    # Get plant bounding rectangle
    x, y, w, h = cv2.boundingRect(plant_contour)

    # Get plant top and bottom points more precisely
    # This helps get a more accurate height measurement
    hull = cv2.convexHull(plant_contour)
    top_y = min(point[0][1] for point in hull)
    bottom_y = max(point[0][1] for point in hull)

    # Calculate actual plant height (either using bounding box or hull points)
    plant_height = bottom_y - top_y

    # If hull measurement seems incorrect, fall back to bounding box
    if plant_height < h * 0.7:
        plant_height = h

    # Output the height without classification
    print(f"Plant height: {plant_height} pixels")

    if display:
        # Draw the measurement visualization
        center_x = x + w // 2
        cv2.line(result_image, (center_x, top_y), (center_x, bottom_y), (0, 0, 255), 2)
        cv2.circle(result_image, (center_x, top_y), 5, (255, 0, 0), -1)
        cv2.circle(result_image, (center_x, bottom_y), 5, (255, 0, 0), -1)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add text with height information
        cv2.putText(result_image, f"Height: {plant_height}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save annotated image for website display
        output_dir = os.path.join(os.path.dirname(image_path), "annotated")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, result_image)

        # Resize for display
        display_scale = min(1, 800 / max(width, height))
        resized_image = cv2.resize(result_image, None,
                                   fx=display_scale,
                                   fy=display_scale,
                                   interpolation=cv2.INTER_AREA)
        # Save resized image
        resized_dir = os.path.join(os.path.dirname(image_path), "resized")
        os.makedirs(resized_dir, exist_ok=True)
        resized_path = os.path.join(resized_dir, f"resized_{os.path.basename(image_path)}")
        cv2.imwrite(resized_path, resized_image)

        # Display result
        cv2.imshow("Plant Height", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return plant_height


def extract_date_from_filename(filename):
    """Extract date from filename with format YYYY_MM_DD_XXAM/PM.JPG"""
    pattern = r'(\d{4})_(\d{2})_(\d{2})_(\d{2})([AP]M)'
    match = re.search(pattern, filename)

    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(4))
        am_pm = match.group(5)

        # Adjust hour for PM
        if am_pm == 'PM' and hour < 12:
            hour += 12
        elif am_pm == 'AM' and hour == 12:
            hour = 0

        # Create datetime object
        date_obj = datetime.datetime(year, month, day, hour)
        return date_obj, f"{month}/{day} {hour:02d}:00"

    return None, None


# Main execution
if __name__ == "__main__":
    # Path to a single selected image
    image_path = "Asset/Images/Selected images/height/2024_12_26_09AM.JPG"  # Replace with your selected image path
    
    # Extract date from filename
    date_obj, date_label = extract_date_from_filename(os.path.basename(image_path))
    if date_obj is None:
        print(f"Could not extract date from {image_path}, skipping")
    else:
        # Measure plant height for the selected image
        height = measure_plant_height(image_path, display=True)

        if height is not None:
            print(f"Height: {height} pixels, Date: {date_label}")
        else:
            print("Could not measure plant height")
