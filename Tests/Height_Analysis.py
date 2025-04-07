import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime
from matplotlib import dates as mdates


def measure_plant_height(image_path, display=False):
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
    # In most cases, the convex hull gives a more precise measurement
    plant_height = bottom_y - top_y

    # If hull measurement seems incorrect, fall back to bounding box
    if plant_height < h * 0.7:
        plant_height = h

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


def analyze_plant_growth(directory):
    """Analyze all plant images in the directory and plot height over time"""
    # Lists to store data
    heights = []
    dates = []
    date_labels = []
    filenames = []

    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = [f for f in os.listdir(directory) if any(f.endswith(ext) for ext in valid_extensions)]

    if not image_files:
        print(f"No image files found in {directory}")
        return

    print(f"Found {len(image_files)} images to process")

    # Create output directories
    output_dir = os.path.join(directory, "results")
    os.makedirs(output_dir, exist_ok=True)

    annotated_dir = os.path.join(directory, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    # Process each image
    for i, filename in enumerate(sorted(image_files)):
        image_path = os.path.join(directory, filename)
        print(f"Processing {i + 1}/{len(image_files)}: {filename}...")

        # Extract date from filename
        date_obj, date_label = extract_date_from_filename(filename)
        if date_obj is None:
            print(f"Could not extract date from {filename}, skipping")
            continue

        # Measure plant height and save annotated image
        height = measure_plant_height(image_path, display=True)

        if height is not None:
            heights.append(height)
            dates.append(date_obj)
            date_labels.append(date_label)
            filenames.append(filename)
            print(f"Height: {height} pixels, Date: {date_label}")

            # Save annotated image for web display
            result_image = cv2.imread(image_path)
            x, y, w, h = 10, 10, 200, 30  # Position for text
            cv2.putText(result_image, f"Height: {height}px",
                        (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            annotated_path = os.path.join(annotated_dir, filename)
            cv2.imwrite(annotated_path, result_image)

    if not heights:
        print("No valid measurements obtained")
        return

    # Sort data by date
    sorted_data = sorted(zip(dates, date_labels, heights, filenames))
    dates = [item[0] for item in sorted_data]
    date_labels = [item[1] for item in sorted_data]
    heights = [item[2] for item in sorted_data]
    filenames = [item[3] for item in sorted_data]

    # Create multiple graph styles for website display

    # 1. Standard line plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(heights)), heights, marker='o', color='green', linewidth=2, markersize=8)
    plt.title("Plant Height Changes Over Time", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Plant Height (pixels)", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Set x-axis labels to dates
    plt.xticks(range(len(heights)), date_labels, rotation=45)

    # Improve plot layout
    plt.tight_layout()

    # Save the standard plot
    plt.savefig(os.path.join(output_dir, "plant_height_growth.png"), dpi=300, bbox_inches='tight')
    print(f"Standard plot saved to {os.path.join(output_dir, 'plant_height_growth.png')}")

    # 2. Web-optimized version (transparent background, thicker lines)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(heights)), heights, marker='o', color='#00A86B', linewidth=3, markersize=10)

    # Add growth rate annotations
    if len(heights) > 1:
        for i in range(1, len(heights)):
            growth_rate = heights[i] - heights[i - 1]
            if growth_rate > 0:
                plt.annotate(f"+{growth_rate}",
                             xy=(i, heights[i]),
                             xytext=(0, 10),
                             textcoords="offset points",
                             ha='center',
                             color='green')

    plt.title("Plant Growth Monitoring", fontsize=18, fontweight='bold')
    plt.xlabel("Measurement Date", fontsize=14)
    plt.ylabel("Plant Height (pixels)", fontsize=14)
    plt.grid(True, alpha=0.2)

    # Set x-axis labels to dates
    plt.xticks(range(len(heights)), date_labels, rotation=45)

    # Add light background with transparency for better web display
    plt.gca().set_facecolor('whitesmoke')

    # Add light grid for better readability
    plt.grid(color='gainsboro', linestyle='-', linewidth=0.5)

    # Tight layout and save with transparency
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plant_height_web.png"), dpi=300,
                transparent=True, bbox_inches='tight')
    print(f"Web-optimized plot saved to {os.path.join(output_dir, 'plant_height_web.png')}")

    # 3. Create a bar chart version for alternative visualization
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(heights)), heights, color='#00A86B', alpha=0.8)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.title("Plant Height Measurements", fontsize=18, fontweight='bold')
    plt.xlabel("Measurement Date", fontsize=14)
    plt.ylabel("Plant Height (pixels)", fontsize=14)
    plt.xticks(range(len(heights)), date_labels, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plant_height_bars.png"), dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to {os.path.join(output_dir, 'plant_height_bars.png')}")

    # Save data to CSV
    csv_path = os.path.join(output_dir, "plant_height_data.csv")
    with open(csv_path, "w") as f:
        f.write("Date,Filename,Height\n")
        for date, filename, height in zip(date_labels, filenames, heights):
            f.write(f"{date},{filename},{height}\n")

    print(f"Data saved to {csv_path}")

    # Generate HTML snippet for website integration
    html_snippet = f"""
    <div class="plant-growth-analysis">
        <h2>Plant Growth Monitoring Results</h2>
        <div class="growth-chart">
            <img src="results/plant_height_web.png" alt="Plant Height Growth Chart">
        </div>
        <div class="growth-table">
            <table>
                <tr>
                    <th>Date</th>
                    <th>Height (px)</th>
                </tr>
                {"".join(f'<tr><td>{date}</td><td>{height}</td></tr>' for date, height in zip(date_labels, heights))}
            </table>
        </div>
    </div>
    """

    # Save HTML snippet
    html_path = os.path.join(output_dir, "website_integration.html")
    with open(html_path, "w") as f:
        f.write(html_snippet)

    print(f"HTML snippet for website integration saved to {html_path}")

    # Show the plot (can be commented out when running on server)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Directory containing the images
    image_directory = "Asset/Images/Selected images/height"
    analyze_plant_growth(image_directory)
