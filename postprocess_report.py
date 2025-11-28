# src/postprocess_report.py
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Define colors for visualization
CLASS_COLORS = {
    0: (0, 0, 0),       # Background (Black)
    1: (0, 0, 255),     # Class 1: Pothole (Blue)
    2: (0, 255, 0)      # Class 2: Crack (Green)
}
CLASS_NAMES = {
    1: "Pothole",
    2: "Crack"
}

def create_segmentation_mask(prediction_tensor, target_size):
    """
    Converts model output tensor to a NumPy segmentation mask.
    """
    # Get the class with the highest probability for each pixel
    # prediction_tensor is (1, C, H, W). We take argmax along C (dim=1)
    mask_tensor = torch.argmax(prediction_tensor.squeeze(0), dim=0).cpu().numpy()
    
    # Resize mask back to the original target resolution for accurate calculation
    mask_resized = cv2.resize(mask_tensor.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
    
    return mask_resized

def quantify_and_report(mask_resized, image_name, report_path="./data/Defect_Report.csv"):
    """
    Module 3: Calculates defect area and logs the report.
    """
    results = []
    total_pixels = mask_resized.size
    
    print("\nINFO: Quantifying defects...")
    
    for class_id, class_name in CLASS_NAMES.items():
        # Count pixels belonging to the current class ID
        defect_pixels = np.sum(mask_resized == class_id)
        
        # Calculate area percentage
        area_percentage = (defect_pixels / total_pixels) * 100
        
        # Simple severity assignment based on area
        severity = "LOW"
        if area_percentage > 0.5:
            severity = "MEDIUM"
        if area_percentage > 2.0:
            severity = "HIGH"
            
        if defect_pixels > 0:
            results.append({
                'timestamp': datetime.now().isoformat(),
                'image_id': image_name,
                'defect_type': class_name,
                'area_sq_pixels': int(defect_pixels),
                'area_percent': round(area_percentage, 4),
                'severity': severity
            })
            print(f"  - Found {class_name}: {defect_pixels} pixels ({area_percentage:.4f}%) - Severity: {severity}")
            
    # Log data to CSV (NFR: Data integrity/Logging)
    if results:
        df = pd.DataFrame(results)
        
        if not os.path.exists(report_path):
            df.to_csv(report_path, index=False)
        else:
            df.to_csv(report_path, mode='a', header=False, index=False)
        
        print(f"INFO: Defect report logged successfully to {report_path}")

    return results


def visualize_mask(original_img, mask_resized, output_path):
    """
    Overlay the segmented mask onto the original image for visualization.
    """
    # Create an empty overlay image
    overlay = np.zeros_like(original_img, dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        # Apply color to the pixels of the corresponding class
        overlay[mask_resized == class_id] = color
        
    # Blend the original image with the mask (alpha blending)
    # NFR: Usability (clear visualization)
    alpha = 0.6 
    segmented_img = cv2.addWeighted(original_img, 1 - alpha, overlay, alpha, 0)
    
    # Save the output
    cv2.imwrite(output_path, segmented_img)
    print(f"INFO: Segmented image saved to {output_path}")