import cv2
import numpy as np
import os


def analyze_and_display_leaf_health(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    # Create masks
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Calculate health score
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    total = green_pixels + yellow_pixels

    if total == 0:
        health_score = 0.0
    else:
        health_score = green_pixels / total

    # Create annotated image
    annotated_image = image.copy()

    # Add health information text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Health: {health_score * 100:.1f}%"
    color = (0, 255, 0) if health_score > 0.5 else (0, 0, 255)

    cv2.putText(annotated_image, text, (20, 50), font, 1, color, 2)

    # Resize for better display
    annotated_image = cv2.resize(annotated_image, (800, 600))

    return annotated_image


# Create output directory if it doesn't exist
folder_path = "Asset/Images/Selected images/health"
output_folder = os.path.join(folder_path, 'measured_images')
os.makedirs(output_folder, exist_ok=True)

# Process images in folder
for filename in sorted(os.listdir(folder_path)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder_path, filename)
        result_image = analyze_and_display_leaf_health(image_path)

        if result_image is not None:
            # Save processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result_image)

            # Display image
            cv2.imshow("Plant Health Analysis", result_image)
            print(f"Processed and saved: {filename} - Press 'n' to continue or 'q' to quit")

            # Wait for key press
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

cv2.destroyAllWindows()
