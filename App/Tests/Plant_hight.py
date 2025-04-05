import cv2
import numpy as np

def measure_plant_height(image_path, pot_height_mm=None):
    """
    Measure the height of a plant in an image.
    
    Args:
        image_path: Path to the image file
        pot_height_mm: Actual height of the pot in millimeters (for calibration)
                      If provided, will calculate actual plant height
    
    Returns:
        plant_height_pixels: Height of the plant in pixels
        plant_height_mm: Height of the plant in millimeters (if pot_height_mm is provided)
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image '{image_path}' not found")
        
        # Create a copy for visualization
        result_img = img.copy()
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        # Green for plant
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([95, 255, 255])
        
        # Yellow/brown for stem
        lower_yellow = np.array([15, 30, 40])
        upper_yellow = np.array([30, 255, 255])
        
        # Create masks
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks to get the entire plant
        plant_mask = cv2.bitwise_or(green_mask, yellow_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No plant detected in the image")
        
        # Find the largest contour (should be the plant)
        plant_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(plant_contour)
        
        # Draw bounding box
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Find the pot in the image
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # Find contours
        pot_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and position (pot should be at the bottom)
        pot_contour = None
        max_area = 0
        for contour in pot_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                # Get the bottom-most point
                bottom_most = tuple(contour[contour[:, :, 1].argmax()][0])
                if bottom_most[1] > img.shape[0] * 0.5:  # Bottom half of the image
                    pot_contour = contour
                    max_area = area
        
        # If pot is found, calculate the top of the pot
        pot_top_y = None
        if pot_contour is not None:
            # Get the bounding rectangle of the pot
            pot_x, pot_y, pot_w, pot_h = cv2.boundingRect(pot_contour)
            pot_top_y = pot_y
            
            # Draw pot bounding box
            cv2.rectangle(result_img, (pot_x, pot_y), (pot_x + pot_w, pot_y + pot_h), (255, 0, 0), 2)
        else:
            # If pot not found, use the bottom of the image as reference
            pot_top_y = img.shape[0]
        
        # Calculate plant height in pixels
        plant_height_pixels = pot_top_y - y
        
        # Draw height line
        cv2.line(result_img, (x + w//2, y), (x + w//2, pot_top_y), (0, 0, 255), 2)
        cv2.putText(result_img, f"Height: {plant_height_pixels} px", 
                   (x + w + 10, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate actual height if pot_height_mm is provided
        plant_height_mm = None
        if pot_height_mm is not None and pot_contour is not None:
            # Get pot height in pixels
            _, _, _, pot_height_pixels = cv2.boundingRect(pot_contour)
            
            # Calculate mm per pixel ratio
            mm_per_pixel = pot_height_mm / pot_height_pixels
            
            # Calculate plant height in mm
            plant_height_mm = plant_height_pixels * mm_per_pixel
            
            cv2.putText(result_img, f"Height: {plant_height_mm:.1f} mm", 
                       (x + w + 10, y + h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display results
        cv2.imshow("Original", img)
        cv2.imshow("Plant Mask", plant_mask)
        cv2.imshow("Height Measurement", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return plant_height_pixels, plant_height_mm
    
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    image_path = "Asset/Images/2024_12_27_12AM_u.JPG"
    
    # If you know the actual height of the pot in mm, you can provide it for calibration
    pot_height_mm = 50  # Example value, replace with actual pot height if known
    
    height_pixels, height_mm = measure_plant_height(image_path, pot_height_mm)
    
    if height_pixels:
        print(f"Plant height: {height_pixels} pixels")
        if height_mm:
            print(f"Plant height: {height_mm:.1f} mm")
