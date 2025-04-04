import cv2
import numpy as np


def resize_image(image, width=500):
    height, w, _ = image.shape
    ratio = width / w
    new_height = int(height * ratio)
    return cv2.resize(image, (width, new_height))


def count_and_show_leaves(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image '{image_path}' not found")

        img = resize_image(img)

        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Further refined green detection range
        lower_green = np.array([30, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological operations to remove noise and enhance leaf regions
        kernel = np.ones((5, 5), np.uint8)
        processed_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=3)

        # Find contours on the processed mask
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours (noise)
        min_leaf_area = 800  # Increased to further prevent false positives
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_leaf_area]
        leaf_count = len(valid_contours)

        # Draw contours on the original image
        result = img.copy()
        cv2.drawContours(result, valid_contours, -1, (0, 0, 255), 2)

        # Label detected leaves
        for i, contour in enumerate(valid_contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, str(i + 1), (cX - 10, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display results
        cv2.imshow("Original", img)
        cv2.imshow("Green Mask", green_mask)
        cv2.imshow("Processed Mask", processed_mask)
        cv2.imshow("Final Detection", result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return leaf_count

    except Exception as e:
        print(f"Error: {e}")
        return 0


# Test with your image
if __name__ == "__main__":
    image_filename = "static/images/test/plantleafcount.jpg"  # Change to your image
    leaf_count = count_and_show_leaves(image_filename)
    print(f"Detected {leaf_count} individual leaves")