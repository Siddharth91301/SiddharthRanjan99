# main.py
import argparse
import os
import torch
import numpy as np
import cv2

# Import the three functional modules
from models.model_architecture import load_model
from src.preprocess import load_and_preprocess
from src.postprocess_report import create_segmentation_mask, quantify_and_report, visualize_mask

# --- Mock Data Simulation ---
def mock_prediction(input_tensor, num_classes=2):
    """
    Mocks the output of the U-Net model for demonstration. 
    In a real project, this would be model(input_tensor).
    """
    # Simulate a prediction where some areas are Pothole (1) or Crack (2)
    B, C, H, W = input_tensor.shape
    
    # Create random noise (0 or 1)
    mock_raw_output = torch.rand(B, num_classes, H, W)
    
    # Inject a simple simulated defect in the center-right area for Class 1 (Pothole)
    # This simulates a successful localization
    center_h, center_w = H // 2, W // 2
    mock_raw_output[0, 0, center_h-10:center_h+10, center_w+30:center_w+50] += 5.0 
    
    # Inject a simulated defect in the bottom-left area for Class 2 (Crack)
    mock_raw_output[0, 1, center_h+40:center_h+50, center_w-50:center_w-10] += 5.0 
    
    return mock_raw_output

# --- Main Pipeline ---
def run_road_d_loc(image_path, model_weights_path, output_dir="output"):
    """
    The main execution flow, demonstrating the logical workflow.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Module 1: Image Acquisition and Preprocessing
    try:
        original_img, input_tensor = load_and_preprocess(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Determine original size for resizing the mask back
    original_h, original_w = original_img.shape[:2]

    # 2. Module 2: Deep Segmentation and Localization
    # In a real project: model = load_model(model_weights_path)
    # prediction_tensor = model(input_tensor)
    
    print("INFO: Performing Segmentation Inference...")
    # Using mock prediction for demonstration
    prediction_tensor = mock_prediction(input_tensor) 

    # 3. Post-processing: Convert tensor to mask
    mask_resized = create_segmentation_mask(prediction_tensor, (original_w, original_h))

    # 4. Module 3: Defect Quantification and Reporting
    image_name = os.path.basename(image_path)
    quantify_and_report(mask_resized, image_name)

    # 5. Visualization (part of Module 3 output)
    output_filename = image_name.replace('.', '_segmented.')
    output_path = os.path.join(output_dir, output_filename)
    visualize_mask(original_img, mask_resized, output_path)

    print("\n--- Processing Complete ---")
    print(f"Check the output folder: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Road Surface Defect Localization System (Road-D-Loc)")
    parser.add_argument("image_path", type=str, help="Path to the input road image.")
    parser.add_argument("--weights", type=str, default="./models/checkpoints/best_unet_weights.pth", 
                        help="Path to the pre-trained U-Net weights file.")
    
    args = parser.parse_args()

    # --- Setup Mock Environment ---
    # Create placeholder directories and a dummy image for testing the script structure
    os.makedirs("./models/checkpoints", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    # If the user doesn't provide an image, create a dummy one.
    if not os.path.exists(args.image_path):
        dummy_img = np.zeros((500, 800, 3), dtype=np.uint8) + 150 # Gray image
        cv2.putText(dummy_img, "DUMMY ROAD IMAGE", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite("test_road.jpg", dummy_img)
        args.image_path = "test_road.jpg"
        print(f"NOTE: Created a dummy image '{args.image_path}' to demonstrate the pipeline.")
    
    run_road_d_loc(args.image_path, args.weights)