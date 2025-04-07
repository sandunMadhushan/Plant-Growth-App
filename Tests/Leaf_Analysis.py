import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def resize_image(image, width=500):
    height, w, _ = image.shape
    ratio = width / w
    new_height = int(height * ratio)
    return cv2.resize(image, (width, new_height))


def count_and_show_leaves(image_path, show_debug=False):
    try:
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

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        height, width = closing.shape
        left_mask = np.zeros((height, width), np.uint8)
        left_mask[:, int(width * 0.3):] = 255
        processed_mask = cv2.bitwise_and(closing, left_mask)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_leaf_area = 30
        max_leaf_area = 5000
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_leaf_area < area < max_leaf_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / max(h, 1)
                if 0.2 < aspect_ratio < 3.0:
                    valid_contours.append(contour)

        if len(valid_contours) < 2:
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            _, a, _ = cv2.split(lab)
            _, a_thresh = cv2.threshold(a, 120, 255, cv2.THRESH_BINARY_INV)
            a_thresh = cv2.bitwise_and(a_thresh, left_mask)
            a_processed = cv2.morphologyEx(a_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            a_processed = cv2.morphologyEx(a_processed, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(a_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_leaf_area < area < max_leaf_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / max(h, 1)
                    if 0.2 < aspect_ratio < 3.0:
                        valid_contours.append(contour)

            if len(valid_contours) < 2:
                sure_bg = cv2.dilate(processed_mask, kernel, iterations=3)
                dist_transform = cv2.distanceTransform(processed_mask, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                markers = cv2.watershed(img, markers)

                for i in range(2, markers.max() + 1):
                    curr_mask = np.zeros_like(processed_mask, dtype=np.uint8)
                    curr_mask[markers == i] = 255
                    curr_contours, _ = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in curr_contours:
                        area = cv2.contourArea(contour)
                        if min_leaf_area < area < max_leaf_area:
                            valid_contours.append(contour)

        result = original.copy()
        for i, contour in enumerate(valid_contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, str(i + 1), (cX - 10, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)

        # Save debug image in the same folder as input
        debug_path = os.path.join(os.path.dirname(image_path), f"debug_{os.path.basename(image_path)}")
        cv2.imwrite(debug_path, result)

        if show_debug:
            cv2.imshow("Original", original)
            cv2.imshow("Green Mask", green_mask)
            cv2.imshow("Processed Mask", processed_mask)
            cv2.imshow("Final Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return len(valid_contours)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def analyze_folder(folder_path):
    results = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            count = count_and_show_leaves(image_path)
            results[filename] = count
            print(f"{filename}: {count} leaves detected")
    return results


def plot_leaf_counts(results_dict):
    sorted_items = sorted(results_dict.items())
    image_names = [name for name, _ in sorted_items]
    leaf_counts = [count if count is not None else 0 for _, count in sorted_items]

    plt.figure(figsize=(10, 6))
    plt.plot(image_names, leaf_counts, marker='o', linestyle='-', color='green')
    plt.xlabel("Image")
    plt.ylabel("Leaf Count")
    plt.title("Leaf Count Over Time")
    plt.xticks(rotation=45, ha='right')

    for i, count in enumerate(leaf_counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("leaf_counts_line_graph.png")
    plt.show()


if __name__ == "__main__":
    image_folder = "Tests/images"  # 🖼️ Set your image folder path here
    results = analyze_folder(image_folder)
    plot_leaf_counts(results)