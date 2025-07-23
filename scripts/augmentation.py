"""Script for image data augmentation with automatic cropping and resizing."""

import os
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image
import torch
from PIL import Image

# === PARAMETERS ===
INPUT_ROOT = './Chunk2'
OUTPUT_ROOT = './augmented_images'
AUGMENTATIONS_PER_IMAGE = 7
CROP_PADDING = 10

# === DATA AUGMENTATION ===
class CustomAugment:
    """Custom image augmentation class that applies various transformations."""
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            transforms.RandomApply([transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)], p=0.5),
            transforms.RandomAdjustSharpness(2, p=0.3),
        ])

    def __call__(self, img):
        return self.augment(img)

augmenter = CustomAugment()

# === FUNCTIONS ===
def auto_crop_and_resize(image_path, target_size=256, scale=1):
    """Automatically crop and resize an image to focus on the main object.
    
    Args:
        image_path (str): Path to the image file
        target_size (int, optional): Target size for the output image. Defaults to 256.
        scale (float, optional): Scale factor for padding around the object. Defaults to 1.
        
    Returns:
        PIL.Image: Cropped and resized image
    """
    # Try first with cv2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # If cv2 fails, use PIL and always convert to numpy array
    if img is None:
        try:
            pil_fallback = Image.open(image_path).convert("L")
            img = np.array(pil_fallback)
        except Exception as e:
            raise ValueError(f"Error reading {image_path}: {e}") from e
    
    # Make sure img is always a numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        raise ValueError("No white object found in the image")

    x, y, w, h = cv2.boundingRect(coords)

    # Proportional box centered on the object
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Use the effective dimension of the object
    obj_size = max(w, h)
    # Add controlled padding around the object
    padding = int(obj_size * scale)  # scale now controls the padding
    half_size = (obj_size + padding) // 2
    
    x1 = max(center_x - half_size, 0)
    y1 = max(center_y - half_size, 0)
    x2 = min(center_x + half_size, img.shape[1])
    y2 = min(center_y + half_size, img.shape[0])
    
    # Make sure the crop is square
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w != crop_h:
        # Take the smaller dimension to keep everything in the image
        min_size = min(crop_w, crop_h)
        x2 = x1 + min_size
        y2 = y1 + min_size

    cropped = img[y1:y2, x1:x2]
    pil_img = Image.fromarray(cropped)
    pil_img = F.resize(img=pil_img, size=(target_size, target_size))
    return pil_img

def process_folder(folder_path, output_path):
    """Process a folder of images for augmentation.

    Args:
        folder_path (str): Path to the folder containing images to process
        output_path (str): Path where augmented images will be saved
    """
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        image_path = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0]

        try:
            # Use smaller scale for smaller object in the center
            cropped_resized_img = auto_crop_and_resize(image_path, scale=0.5)

            # Save original image (already a PIL Image from the function)
            cropped_resized_img.save(os.path.join(output_path, f"{base_name}_orig.png"))

            # Generate augmented images
            for i in range(AUGMENTATIONS_PER_IMAGE):
                try:
                    # Apply augmentation
                    augmented = augmenter(cropped_resized_img)
                    
                    # Handle tensor/PIL conversion
                    if isinstance(augmented, torch.Tensor):
                        augmented_pil = to_pil_image(augmented)
                    else:
                        augmented_pil = augmented
                    
                    # Save augmented image
                    augmented_pil.save(os.path.join(output_path, f"{base_name}_aug_{i}.png"))
                    
                except Exception as aug_error:
                    print(f"Error in augmentation {i} for {image_path}: {aug_error}")
                    continue

        except Exception as e:
            print(f"Error on {image_path}: {e}")

def process_all_folders(input_root, output_root):
    """Process all folders recursively from input root to output root.
    
    Args:
        input_root (str): Root directory containing folders with images
        output_root (str): Root directory where processed images will be saved
    """
    # Check input root exists
    if not os.path.exists(input_root):
        print(f"Input root {input_root} does not exist.")
        return
    
    for root, dirs, _ in os.walk(input_root):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            relative_path = os.path.relpath(folder_path, input_root)
            output_path = os.path.join(output_root, relative_path)
            print(f"Processing {relative_path}...")
            process_folder(folder_path, output_path)

# === MAIN ===
if __name__ == "__main__":
    process_all_folders(INPUT_ROOT, OUTPUT_ROOT)