# src/preprocess.py
import cv2
import numpy as np
import torch
import torchvision.transforms as T

def load_and_preprocess(image_path, target_size=(256, 256)):
    """
    Module 1: Loads the image, resizes it, and converts it to a normalized PyTorch tensor.
    
    Input: Path to the image file.
    Output: Original image (for visualization) and preprocessed tensor (for model).
    """
    # 1. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Error: Could not open or find the image at {image_path}")
        
    # Convert BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Preprocessing (Resize and Normalization)
    
    # Save the original image for later visualization/overlay (NFR: Usability)
    original_img = img_bgr.copy() 
    
    # Resize the image for model input
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Define transformations: Convert to Tensor and normalize (using ImageNet stats as a common baseline)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert numpy array (H, W, C) to PyTorch tensor (C, H, W) and normalize
    input_tensor = transform(img_resized).unsqueeze(0) # Add batch dimension (B, C, H, W)
    
    print(f"INFO: Image loaded and preprocessed to tensor shape: {input_tensor.shape}")
    
    return original_img, input_tensor