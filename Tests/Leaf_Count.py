import cv2
import numpy as np


def resize_image(image, width=500):
    height, w, _ = image.shape
    ratio = width / w
    new_height = int(height * ratio)
    return cv2.resize(image, (width, new_height))


def count_and_show_leaves(image_path):
    try:
        # Read and resize image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image '{image_path}' not found")

        img = resize_image(img)
        original = img.copy()

        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        height, width = closing.shape
        left_mask = np.zeros((height, width), np.uint8)
        left_mask[:, int(width * 0.3):] = 255
        processed_mask = cv2.bitwise_and(closing, left_mask)

        # Find external contours
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and aspect ratio
        min_leaf_area = 30
        max_leaf_area = 5000
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_leaf_area < area < max_leaf_area:
                # Check aspect ratio for leaf-like shapes
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / max(h, 1)  # Avoid division by zero

                # Typical seedling leaves have aspect ratios between 0.2 and 3.0
                if 0.2 < aspect_ratio < 3.0:
                    valid_contours.append(contour)

        # If we still can't find leaves properly, try another approach
        if len(valid_contours) < 2:
            # Try LAB color space for better leaf segmentation
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Use 'a' channel which represents green-red
            # Threshold to isolate green areas
            _, a_thresh = cv2.threshold(a, 120, 255, cv2.THRESH_BINARY_INV)

            # Apply the same left-side mask
            a_thresh = cv2.bitwise_and(a_thresh, left_mask)

            a_processed = cv2.morphologyEx(a_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            a_processed = cv2.morphologyEx(a_processed, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Find contours again with LAB approach
            contours, _ = cv2.findContours(a_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_leaf_area < area < max_leaf_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / max(h, 1)
                    if 0.2 < aspect_ratio < 3.0:
                        valid_contours.append(contour)

            # If we still don't have enough contours, try watershed segmentation
            if len(valid_contours) < 2:
                # Start with our processed mask
                sure_bg = cv2.dilate(processed_mask, kernel, iterations=3)

                # Find sure foreground area using distance transform
                dist_transform = cv2.distanceTransform(processed_mask, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)

                # Find unknown region
                unknown = cv2.subtract(sure_bg, sure_fg)

                # Marker labeling
                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0

                # Apply watershed
                markers = cv2.watershed(img, markers)

                # Create a mask for each detected leaf region
                leaf_mask = np.zeros_like(processed_mask)
                for i in range(2, markers.max() + 1):
                    # Create a mask for this marker
                    curr_mask = np.zeros_like(processed_mask, dtype=np.uint8)
                    curr_mask[markers == i] = 255

                    # Find contours in this mask
                    curr_contours, _ = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)

                    # Add valid contours
                    for contour in curr_contours:
                        area = cv2.contourArea(contour)
                        if min_leaf_area < area < max_leaf_area:
                            valid_contours.append(contour)

        # Create result image showing the detected leaves
        result = original.copy()

        # Draw and number the detected leaves
        for i, contour in enumerate(valid_contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, str(i + 1), (cX - 10, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)

    
        cv2.imshow("Original", original)
        cv2.imshow("Green Mask", green_mask)
        cv2.imshow("Processed Mask", processed_mask)
        cv2.imshow("Final Detection", result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return len(valid_contours)

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    image_filename = "Tests/images/day_10.JPG"
    leaf_count = count_and_show_leaves(image_filename)
    print(f"Detected {leaf_count} individual leaves")

# def count_and_show_leaves(image_path):
#     try:
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"Image '{image_path}' not found")
#
#         img = resize_image(img)
#
#         # Convert to HSV color space
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#         # Green detection range
#         lower_green = np.array([25, 100, 100])
#         upper_green = np.array([95, 255, 255])
#         green_mask = cv2.inRange(hsv, lower_green, upper_green)
#
#         # Morphological operations
#         kernel = np.ones((3, 3), np.uint8)
#         processed_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#         processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#
#         # Distance transform
#         dist_transform = cv2.distanceTransform(processed_mask, cv2.DIST_L2, 5)
#         cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
#         _, thresh = cv2.threshold(dist_transform, 0.4, 1.0, cv2.THRESH_BINARY)
#         markers = np.uint8(thresh * 255)
#         markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
#
#         # Find marker contours
#         marker_contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         min_marker_area = 5
#         valid_marker_contours = [c for c in marker_contours if cv2.contourArea(c) > min_marker_area]
#
#         # Watershed markers
#         markers_watershed = np.zeros(processed_mask.shape, dtype=np.int32)
#         markers_watershed[processed_mask == 0] = 1
#         for i, contour in enumerate(valid_marker_contours):
#             cv2.drawContours(markers_watershed, [contour], -1, i + 2, -1)
#
#         # Watershed
#         cv2.watershed(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), markers_watershed)
#
#         # Extract leaf regions
#         leaf_regions = np.zeros_like(processed_mask)
#         for i in range(2, len(valid_marker_contours) + 2):
#             leaf_regions[markers_watershed == i] = 255
#
#         # Find leaf contours
#         leaf_contours, _ = cv2.findContours(leaf_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         min_leaf_area = 30
#         max_leaf_area = 3000
#         valid_leaf_contours = [c for c in leaf_contours if min_leaf_area < cv2.contourArea(c) < max_leaf_area]
#
#         # Alternative approach if needed
#         if len(valid_leaf_contours) != 3:
#             contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             valid_contours = []
#
#             for contour in contours:
#                 area = cv2.contourArea(contour)
#                 if min_leaf_area < area < max_leaf_area:
#                     if area < 500:
#                         valid_contours.append(contour)
#                     else:
#                         hull = cv2.convexHull(contour, returnPoints=False)
#                         if len(hull) > 3:
#                             defects = cv2.convexityDefects(contour, hull)
#                             if defects is not None:
#                                 mask = np.zeros_like(processed_mask)
#                                 cv2.drawContours(mask, [contour], -1, 255, -1)
#
#                                 for i in range(defects.shape[0]):
#                                     s, e, f, d = defects[i, 0]
#                                     if d / 256.0 > 10:
#                                         start_pt = tuple(contour[s][0])
#                                         end_pt = tuple(contour[e][0])
#                                         far_pt = tuple(contour[f][0])
#                                         cv2.line(mask, far_pt,
#                                                  (2 * far_pt[0] - start_pt[0], 2 * far_pt[1] - start_pt[1]), 0, 2)
#
#                                 split_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                                 for c in split_contours:
#                                     if cv2.contourArea(c) > min_leaf_area:
#                                         valid_contours.append(c)
#
#             if len(valid_contours) != 3:
#                 valid_contours = valid_leaf_contours if len(valid_leaf_contours) > 0 else valid_marker_contours
#                 if len(valid_contours) > 3:
#                     valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]
#         else:
#             valid_contours = valid_leaf_contours
#
#         leaf_count = len(valid_contours)
#
#         # Create result image
#         result = img.copy()
#         cv2.drawContours(result, valid_contours, -1, (0, 0, 255), 2)
#
#         # Label leaves
#         for i, contour in enumerate(valid_contours):
#             M = cv2.moments(contour)
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
#                 cv2.putText(result, str(i + 1), (cX - 10, cY),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#
#         # Encode images
#         _, img_encoded = cv2.imencode('.png', img)
#         _, result_encoded = cv2.imencode('.png', result)
#
#         img_base64 = base64.b64encode(img_encoded).decode('utf-8')
#         result_base64 = base64.b64encode(result_encoded).decode('utf-8')
#
#         return {
#             'leaf_count': leaf_count,
#             'original_image': img_base64,
#             'processed_image': result_base64
#         }
#
#     except Exception as e:
#         print(f"Error: {e}")
#         return {'leaf_count': 0, 'original_image': None, 'processed_image': None}


# if __name__ == "__main__":
#     image_filename = "App/Static/Images/2024_12_27_12AM_u.JPG"
#     leaf_count = count_and_show_leaves(image_filename)
#     print(f"Detected {leaf_count} individual leaves")
