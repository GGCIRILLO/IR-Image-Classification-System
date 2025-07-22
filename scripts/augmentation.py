"""Script per data augmentation di immagini con crop automatico e ridimensionamento."""

import os
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image
import torch
from PIL import Image

# === PARAMETRI ===
INPUT_ROOT = './Chunk2'
OUTPUT_ROOT = './immagini_augmentate'
AUGMENTATIONS_PER_IMAGE = 7
CROP_PADDING = 10

# === DATA AUGMENTATION ===
class CustomAugment:
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

# === FUNZIONI ===
def auto_crop_and_resize(image_path, target_size=256, scale=1):
    # Prova prima con cv2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Se cv2 fallisce, usa PIL e converti sempre in numpy array
    if img is None:
        try:
            pil_fallback = Image.open(image_path).convert("L")
            img = np.array(pil_fallback)
        except Exception as e:
            raise ValueError(f"Errore nel leggere {image_path}: {e}") from e
    
    # Assicurati che img sia sempre un numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        raise ValueError("Nessun oggetto bianco trovato nell'immagine")

    x, y, w, h = cv2.boundingRect(coords)

    # Box proporzionale centrato sull'oggetto
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Usa la dimensione dell'oggetto effettivo, non il max
    obj_size = max(w, h)
    # Aggiungi padding controllato intorno all'oggetto
    padding = int(obj_size * scale)  # scale ora controlla il padding
    half_size = (obj_size + padding) // 2
    
    x1 = max(center_x - half_size, 0)
    y1 = max(center_y - half_size, 0)
    x2 = min(center_x + half_size, img.shape[1])
    y2 = min(center_y + half_size, img.shape[0])
    
    # Assicurati che il crop sia quadrato
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w != crop_h:
        # Prendi la dimensione minore per mantenere tutto nell'immagine
        min_size = min(crop_w, crop_h)
        x2 = x1 + min_size
        y2 = y1 + min_size

    cropped = img[y1:y2, x1:x2]
    pil_img = Image.fromarray(cropped)
    pil_img = F.resize(img=pil_img, size=(target_size, target_size))
    return pil_img

def process_folder(folder_path, output_path):
    """_summary_

    Args:
        folder_path (_type_): _description_
        output_path (_type_): _description_
    """
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        image_path = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0]

        try:
            # Usa scale più piccolo per oggetto più piccolo al centro
            cropped_resized_img = auto_crop_and_resize(image_path, scale=0.5)

            # Salva immagine originale (già PIL Image dalla funzione)
            cropped_resized_img.save(os.path.join(output_path, f"{base_name}_orig.png"))

            # Genera immagini augmentate
            for i in range(AUGMENTATIONS_PER_IMAGE):
                try:
                    # Applica augmentation
                    augmented = augmenter(cropped_resized_img)
                    
                    # Gestisci conversione tensor/PIL
                    if isinstance(augmented, torch.Tensor):
                        augmented_pil = to_pil_image(augmented)
                    else:
                        augmented_pil = augmented
                    
                    # Salva immagine augmentata
                    augmented_pil.save(os.path.join(output_path, f"{base_name}_aug_{i}.png"))
                    
                except Exception as aug_error:
                    print(f"Errore nell'augmentation {i} per {image_path}: {aug_error}")
                    continue

        except Exception as e:
            print(f"Errore su {image_path}: {e}")

def process_all_folders(input_root, output_root):
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