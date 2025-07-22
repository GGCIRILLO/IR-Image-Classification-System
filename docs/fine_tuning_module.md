# Fine-Tuning Module Documentation

The `fine_tuning` module provides functionality for optimizing pre-trained models to improve their performance on IR image classification tasks. It includes tools for dataset creation, model fine-tuning, and performance evaluation.

## Files and Classes

### `model_optimizer.py`

This file contains classes for optimizing models through fine-tuning and parameter adjustment.

#### `IRImageDataset`

A PyTorch dataset class for IR images used during fine-tuning.

**Key Methods:**

- `__init__(self, images: List[IRImage], transform=None)`: Initializes the dataset with IR images and optional transformations.
- `__len__(self)`: Returns the number of images in the dataset.
- `__getitem__(self, idx)`: Returns a specific image and its label by index.

#### `ModelOptimizer`

The main class responsible for optimizing models through fine-tuning and parameter adjustment.

**Key Methods:**

- `__init__(self, extractor: EmbeddingExtractor, searcher: SimilaritySearcher, config: Optional[Dict[str, Any]] = None)`: Initializes the optimizer with an embedding extractor, similarity searcher, and optional configuration.
- `optimize_similarity_scores(self, training_data: List[IRImage], validation_data: List[IRImage])`: Optimizes the similarity scoring mechanism using training and validation data.
- `_contrastive_fine_tuning(self, training_data: List[IRImage], validation_data: List[IRImage])`: Performs contrastive fine-tuning on the model.
- `_optimize_similarity_metric(self, validation_data: List[IRImage])`: Optimizes the similarity metric used for comparing embeddings.
- `_evaluate_similarity_metric(self, validation_data: List[IRImage], metric_config: Dict[str, Any])`: Evaluates a similarity metric configuration.
- `_calibrate_confidence_scores(self, validation_data: List[IRImage])`: Calibrates confidence scores for similarity results.
- `_fit_calibration_curve(self, similarities: List[float], accuracies: List[float])`: Fits a calibration curve for mapping similarity scores to confidence values.
- `_evaluate_current_performance(self, validation_data: List[IRImage])`: Evaluates the current performance of the model.
- `_update_confidence_calculation(self, calibration_params: Dict[str, float])`: Updates the confidence calculation based on calibration parameters.
- `_build_class_mapping(self)`: Builds a mapping between class names and indices.
- `_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor)`: Calculates contrastive loss for training.
- `_save_checkpoint(self, model: nn.Module, epoch: int, loss: float)`: Saves a checkpoint during training.
- `_default_config(self)`: Returns the default configuration for the optimizer.
- `apply_quick_fixes(self)`: Applies quick fixes to improve model performance.
- `generate_optimization_report(self)`: Generates a report on the optimization process.

**Static Methods:**

- `create_optimizer(database_path: str, model_path: Optional[str] = None, collection_name: str = "ir_embeddings")`: Creates an optimizer instance with the specified parameters.
- `improved_confidence_calculation(similarity_score: float, rank: int, total_results: int)`: An improved method for calculating confidence scores.
- `improved_similarity_conversion(chroma_results)`: An improved method for converting ChromaDB results.
- `normalized_extract_embedding(image: np.ndarray)`: A normalized method for extracting embeddings.

The fine-tuning module is essential for adapting pre-trained models to the specific characteristics of IR imagery and improving their performance on military target identification tasks. It provides tools for both model parameter optimization and post-processing adjustments to enhance classification accuracy and confidence estimation.
