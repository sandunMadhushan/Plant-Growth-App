import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64


def resize_image(image, width=500):
    height, w, _ = image.shape
    ratio = width / w
    new_height = int(height * ratio)
    return cv2.resize(image, (width, new_height))


def count_and_show_leaves(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return 0

        img = resize_image(img)

        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Green detection range - adjusted for your bright green seedlings
        lower_green = np.array([25, 100, 100])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological operations - use smaller kernel for better separation
        kernel = np.ones((2, 2), np.uint8)
        processed_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Apply more aggressive erosion to separate connected leaves
        eroded_mask = cv2.erode(processed_mask, kernel, iterations=1)

        # Distance transform with lower threshold to detect more seed points
        dist_transform = cv2.distanceTransform(eroded_mask, cv2.DIST_L2, 3)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(dist_transform, 0.3, 1.0, cv2.THRESH_BINARY)
        markers = np.uint8(thresh * 255)

        # Find marker contours
        marker_contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_marker_area = 5
        valid_marker_contours = [c for c in marker_contours if cv2.contourArea(c) > min_marker_area]

        # Watershed markers
        markers_watershed = np.zeros(processed_mask.shape, dtype=np.int32)
        markers_watershed[processed_mask == 0] = 1
        for i, contour in enumerate(valid_marker_contours):
            cv2.drawContours(markers_watershed, [contour], -1, i + 2, -1)

        # Watershed
        cv2.watershed(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), markers_watershed)

        # Extract leaf regions
        leaf_regions = np.zeros_like(processed_mask)
        for i in range(2, len(valid_marker_contours) + 2):
            leaf_regions[markers_watershed == i] = 255

        # Find leaf contours
        leaf_contours, _ = cv2.findContours(leaf_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_leaf_area = 30
        max_leaf_area = 3000
        valid_leaf_contours = [c for c in leaf_contours if min_leaf_area < cv2.contourArea(c) < max_leaf_area]

        # If we still have merged leaves, try to separate them using convexity defects
        if len(valid_leaf_contours) < 3:
            new_contours = []
            for contour in valid_leaf_contours:
                area = cv2.contourArea(contour)
                # If contour is large enough, it might be multiple leaves
                if area > 200:
                    # Find convex hull and defects
                    hull = cv2.convexHull(contour, returnPoints=False)
                    if len(hull) > 3:  # Need at least 4 points for defects
                        defects = cv2.convexityDefects(contour, hull)
                        if defects is not None:
                            # If significant defects exist, split the contour
                            significant_defects = [defect for defect in defects if defect[0][3] > 1000]
                            if len(significant_defects) >= 1:
                                # Use defects to create a mask that splits the contour
                                mask = np.zeros_like(processed_mask)
                                cv2.drawContours(mask, [contour], -1, 255, -1)

                                # Create a line through the deepest defect point
                                for defect in significant_defects:
                                    s, e, f, d = defect[0]
                                    start = tuple(contour[s][0])
                                    end = tuple(contour[e][0])
                                    far = tuple(contour[f][0])

                                    # Draw a line to split the contour
                                    cv2.line(mask, far, (
                                    far[0] * 2 - start[0] // 2 - end[0] // 2, far[1] * 2 - start[1] // 2 - end[1] // 2),
                                             0, 2)

                                # Find contours in the split mask
                                split_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for split_contour in split_contours:
                                    if cv2.contourArea(split_contour) > min_leaf_area:
                                        new_contours.append(split_contour)
                            else:
                                new_contours.append(contour)
                        else:
                            new_contours.append(contour)
                    else:
                        new_contours.append(contour)
                else:
                    new_contours.append(contour)

            if len(new_contours) > len(valid_leaf_contours):
                valid_leaf_contours = new_contours

        # For your seedlings, limit to 3 leaves maximum
        if len(valid_leaf_contours) > 3:
            valid_leaf_contours = sorted(valid_leaf_contours, key=cv2.contourArea, reverse=True)[:3]

        leaf_count = len(valid_leaf_contours)

        # Create result image
        result = img.copy()
        cv2.drawContours(result, valid_leaf_contours, -1, (0, 0, 255), 2)

        # Label leaves
        for i, contour in enumerate(valid_leaf_contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, str(i + 1), (cX - 10, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Save debug image
        debug_path = os.path.join(os.path.dirname(image_path), f"debug_{os.path.basename(image_path)}")
        cv2.imwrite(debug_path, result)

        return leaf_count

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0


def analyze_images_and_plot(directory):
    leaf_counts = []
    days = []

    # Loop through images from day 1 to day 10
    for day in range(1, 11):
        # Try different possible file extensions
        found = False
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            image_path = os.path.join(directory, f"day_{day}{ext}")
            if os.path.exists(image_path):
                print(f"Found image for day {day}: {image_path}")
                leaf_count = count_and_show_leaves(image_path)
                print(f"Day {day}: {leaf_count} leaves detected")
                leaf_counts.append(leaf_count)
                days.append(day)
                found = True
                break

        if not found:
            print(f"Image for day {day} not found with any supported extension.")

    # Plot the results if we have data
    if leaf_counts:
        plt.figure(figsize=(10, 6))
        plt.plot(days, leaf_counts, marker='o', color='green', linewidth=2)
        plt.title("Leaf Count Changes Over 10 Days")
        plt.xlabel("Day")
        plt.ylabel("Leaf Count")
        plt.grid(True)
        plt.xticks(range(1, 11))
        plt.yticks(range(0, max(leaf_counts) + 2))

        # Save the plot
        plt.savefig("leaf_count_changes.png", dpi=300)
        plt.show()

        # Also save the data to a CSV file
        with open("leaf_count_data.csv", "w") as f:
            f.write("Day,LeafCount\n")
            for day, count in zip(days, leaf_counts):
                f.write(f"{day},{count}\n")

        print(f"Data saved to leaf_count_data.csv")
    else:
        print("No images were found or processed. Nothing to plot.")


# Directory containing the images
image_directory = "Tests/images/"
analyze_images_and_plot(image_directory)
