import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import time

# --- 📁 1. CONFIGURE YOUR PATHS AND PROMPT HERE ---

# --- Folders ---
INPUT_FOLDER = r"C:\Users\User\Documents\Bioinformatics_Year3_Sem2\Internship\Fly Project\Image\Female - Synthesiomyia nudiseta\Cropped" # Folder with your iopaint-cleaned images
OUTPUT_FOLDER = r"C:\Users\User\Documents\Bioinformatics_Year3_Sem2\Internship\Fly Project\Image\Female - Synthesiomyia nudiseta\SAM" # Folder where the black and white masks will be saved

# --- SAM Model ---
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"

# --- 🎯 Prompt (Use the relative coordinates you calculated) ---
# Each point is [relative_x, relative_y]
RELATIVE_PROMPT_POINTS = np.array([
    [0.216, 0.274], # Positive Point 1
    [0.728, 0.233], # Positive Point 2
    [0.285, 0.788]  # Negative Point
])
PROMPT_LABELS = np.array([1, 1, 0])

# --- -------------------------------------------- ---

def process_images():
    # --- 2. SETUP ---
    print("--- Starting Batch Segmentation ---")
    
    # Check if output folder exists, create if not
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    # Set up SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Get list of images to process
    try:
        image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if not image_files:
            print(f"Error: No images found in '{INPUT_FOLDER}'. Please check the path.")
            return
        print(f"Found {len(image_files)} images to process.")
    except FileNotFoundError:
        print(f"Error: Input folder '{INPUT_FOLDER}' not found. Please create it and add your images.")
        return

    # --- 3. PROCESSING LOOP ---
    for filename in image_files:
        start_time = time.time()
        print(f"\nProcessing: {filename}...")
        
        # Load the image
        image_path = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  - Warning: Could not read image, skipping.")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set the image in the predictor
        predictor.set_image(image_rgb)
        
        # Calculate absolute pixel coordinates from relative points
        h, w, _ = image_rgb.shape
        absolute_points = (RELATIVE_PROMPT_POINTS * [w, h]).astype(int)

        # Predict the mask
        masks, scores, logits = predictor.predict(
            point_coords=absolute_points,
            point_labels=PROMPT_LABELS,
            multimask_output=False,
        )
        
        # Save the binary mask
        if len(masks) > 0:
            # Create a binary mask (0 for background, 255 for foreground)
            binary_mask = np.where(masks[0] > 0, 255, 0).astype(np.uint8)
            
            # Construct output path
            output_filename = os.path.splitext(filename)[0] + "_mask.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Save the file
            cv2.imwrite(output_path, binary_mask)
            end_time = time.time()
            print(f"  - ✅ Success! Mask saved to {output_path} (Score: {scores[0]:.2f}, Time: {end_time - start_time:.2f}s)")
        else:
            print(f"  - ⚠️ Warning: No mask was generated for this image.")
            
    print("\n--- Batch processing complete! ---")

if __name__ == "__main__":
    process_images()