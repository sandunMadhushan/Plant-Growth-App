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
        
        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Highly specific green detection range for these bright green leaves
        lower_green = np.array([25, 100, 100])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        processed_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Apply distance transform to help separate connected leaves
        dist_transform = cv2.distanceTransform(processed_mask, cv2.DIST_L2, 5)
        
        # Normalize the distance image
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # Threshold to obtain the peaks - foreground markers
        # Lower threshold to catch more potential leaf centers
        _, thresh = cv2.threshold(dist_transform, 0.4, 1.0, cv2.THRESH_BINARY)
        
        # Convert back to uint8 for findContours
        markers = np.uint8(thresh * 255)
        
        # Apply morphological operations to separate markers if needed
        markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        
        # Find contours of the markers
        marker_contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter marker contours by area
        min_marker_area = 5
        valid_marker_contours = [c for c in marker_contours if cv2.contourArea(c) > min_marker_area]
        
        # Create watershed markers
        markers_watershed = np.zeros(processed_mask.shape, dtype=np.int32)
        
        # Background markers
        markers_watershed[processed_mask == 0] = 1
        
        # Foreground markers from valid_marker_contours
        for i, contour in enumerate(valid_marker_contours):
            cv2.drawContours(markers_watershed, [contour], -1, i+2, -1)
        
        # Apply watershed
        cv2.watershed(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), markers_watershed)
        
        # Extract leaf regions from watershed result
        leaf_regions = np.zeros_like(processed_mask)
        for i in range(2, len(valid_marker_contours) + 2):
            leaf_regions[markers_watershed == i] = 255
        
        # Find contours on the leaf regions
        leaf_contours, _ = cv2.findContours(leaf_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter leaf contours by area
        min_leaf_area = 30
        max_leaf_area = 3000
        valid_leaf_contours = [c for c in leaf_contours if min_leaf_area < cv2.contourArea(c) < max_leaf_area]
        
        # If we don't have exactly 3 contours, try alternative approach
        if len(valid_leaf_contours) != 3:
            # Use convexity defects to split connected leaves
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_leaf_area < area < max_leaf_area:
                    # Small contours are likely individual leaves
                    if area < 500:
                        valid_contours.append(contour)
                    else:
                        # Larger contours might be multiple connected leaves
                        # Try to split using convexity defects
                        hull = cv2.convexHull(contour, returnPoints=False)
                        if len(hull) > 3:  # Need at least 4 points for convexity defects
                            defects = cv2.convexityDefects(contour, hull)
                            if defects is not None:
                                # Use the defects to split the contour
                                mask = np.zeros_like(processed_mask)
                                cv2.drawContours(mask, [contour], -1, 255, -1)
                                
                                for i in range(defects.shape[0]):
                                    s, e, f, d = defects[i, 0]
                                    if d/256.0 > 10:  # Significant defect
                                        start_pt = tuple(contour[s][0])
                                        end_pt = tuple(contour[e][0])
                                        far_pt = tuple(contour[f][0])
                                        
                                        # Draw a line to split the contour
                                        cv2.line(mask, far_pt, (2*far_pt[0]-start_pt[0], 2*far_pt[1]-start_pt[1]), 0, 2)
                                
                                # Find the split contours
                                split_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for c in split_contours:
                                    if cv2.contourArea(c) > min_leaf_area:
                                        valid_contours.append(c)
            
            # If we still don't have 3 contours, use the marker approach
            if len(valid_contours) != 3:
                valid_contours = valid_leaf_contours if len(valid_leaf_contours) > 0 else valid_marker_contours
                
                # Keep only the 3 largest contours if we have more
                if len(valid_contours) > 3:
                    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]
        else:
            valid_contours = valid_leaf_contours
        
        leaf_count = len(valid_contours)
        
        # Create result image
        result = img.copy()
        cv2.drawContours(result, valid_contours, -1, (0, 0, 255), 2)
        
        # Label each leaf
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
        cv2.imshow("Distance Transform", dist_transform)
        cv2.imshow("Markers", markers)
        cv2.imshow("Final Detection", result)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return leaf_count
    
    except Exception as e:
        print(f"Error: {e}")
        return 0

# Test with your image
if __name__ == "__main__":
    image_filename = "Asset/Images/2024_12_27_12AM_u.JPG"  # Path to the uploaded image
    leaf_count = count_and_show_leaves(image_filename)
    print(f"Detected {leaf_count} individual leaves")
