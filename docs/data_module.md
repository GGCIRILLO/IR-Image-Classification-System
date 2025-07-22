# Data Module Documentation

The `data` module is responsible for handling IR (Infrared) image processing operations. It provides functionality for loading, validating, preprocessing, and enhancing IR images before they are used for embedding extraction and similarity search.

## Files and Classes

### `ir_processor.py`

This file contains the `IRImageProcessor` class which is the core component for processing infrared images.

#### `IRImageProcessor`

A specialized image processor for infrared imagery that handles the unique characteristics of IR images.

**Key Methods:**

- `__init__(self, target_size: Tuple[int, int] = (224, 224), preserve_aspect_ratio: bool = False)`: Initializes the processor with target size and aspect ratio preservation settings.
- `load_and_validate_image(self, file_path: str)`: Loads an image from a file path and validates it as a proper IR image.
- `validate_ir_format(self, image: np.ndarray)`: Validates that an image meets the requirements for IR format.
- `preprocess_ir_image(self, image: np.ndarray)`: Applies a series of preprocessing steps to enhance IR image quality.
- `normalize_image(self, image: np.ndarray)`: Normalizes pixel values to a standard range.
- `enhance_contrast(self, image: np.ndarray)`: Enhances the contrast of IR images to make features more distinguishable.
- `reduce_noise(self, image: np.ndarray)`: Applies noise reduction techniques specific to IR imagery.
- `resize_to_standard(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224))`: Resizes images to a standard size for model input.
- `get_processing_stats(self, image: np.ndarray)`: Calculates and returns statistics about the processed image.

**Analysis Methods:**

- `_calculate_background_darkness(self, image: np.ndarray)`: Measures the darkness level of the image background.
- `_calculate_bright_pixel_ratio(self, image: np.ndarray)`: Calculates the ratio of bright pixels in the image.
- `_calculate_contrast_ratio(self, image: np.ndarray)`: Measures the contrast ratio in the image.
- `_estimate_noise_level(self, image: np.ndarray)`: Estimates the level of noise present in the image.

The `IRImageProcessor` is designed to handle the specific challenges of infrared imagery, including low contrast, noise, and unique thermal signatures. It ensures that images are properly prepared for the embedding extraction process, which is critical for accurate similarity search results.
