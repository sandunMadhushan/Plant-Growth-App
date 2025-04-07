import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

health_scores = []
dates = []

def analyze_leaf_health(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    total = green_pixels + yellow_pixels

    return green_pixels / total if total != 0 else 0

# Process images
folder_path = "Asset/Images/Selected images/health"
for filename in sorted(os.listdir(folder_path)):
    if filename.lower().endswith(".jpg"):
        score = analyze_leaf_health(os.path.join(folder_path, filename))
        health_scores.append(score)
        dates.append(filename.split(".")[0])  # Extract date from filename

# Generate and save plot
plt.figure(figsize=(10, 5))
plt.plot(dates, health_scores, marker='o', color='green')
plt.title("Plant Health Over Time")
plt.xlabel("Date")
plt.ylabel("Health Score (0 to 1)")
plt.ylim(0, 1)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save plot to health folder
plot_path = os.path.join(folder_path, "plant_health_plot.png")
plt.savefig(plot_path, bbox_inches='tight')
plt.close()

print(f"Plot saved successfully at: {plot_path}")
