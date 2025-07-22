# Training Module Documentation

The `training` module provides functionality for training, fine-tuning, and validating deep learning models for IR image classification. It includes model adapters, training pipelines, and validation tools.

## Files and Classes

### `model_adapters.py`

This file contains adapter classes for different deep learning models, providing a consistent interface for embedding extraction.

#### `BaseModelAdapter`

Base class for model adapters with common functionality.

**Key Methods:**

- `__init__(self, model_name: str, embedding_dim: int = 512, device: Optional[str] = None)`: Initializes the adapter with model name, embedding dimension, and device.
- `load_model(self, model_path: Optional[str] = None, **kwargs)`: Loads a pre-trained model.
- `extract_embedding(self, image: np.ndarray)`: Extracts an embedding from an image.
- `batch_extract(self, images: List[np.ndarray], batch_size: int = 32)`: Extracts embeddings from multiple images.
- `preprocess_image(self, image: np.ndarray)`: Preprocesses an image for the model.
- `validate_embedding_quality(self, embedding: np.ndarray)`: Validates the quality of an embedding.
- `get_model_info(self)`: Returns information about the model.
- `save_model(self, save_path: str)`: Saves the model to a file.
- `_validate_image_input(self, image: np.ndarray)`: Validates an image input.

#### `IRResNet50Adapter`

Adapter for ResNet50 models specialized for IR images.

**Key Methods:**

- `__init__(self, embedding_dim: int = 512, device: Optional[str] = None, pretrained: bool = True)`: Initializes the adapter.
- `_setup_transforms(self)`: Sets up image transformations.
- `fine_tune_setup(self, learning_rate: float = 1e-4, freeze_backbone: bool = False)`: Sets up the model for fine-tuning.

#### `QwenVLMAdapter`

Adapter for Qwen Vision-Language Models.

**Key Methods:**

- `__init__(self, model_name: str = "Qwen/Qwen-VL-Chat", embedding_dim: int = 768, device: Optional[str] = None)`: Initializes the adapter.
- `fine_tune_setup(self, learning_rate: float = 5e-5, freeze_vision_encoder: bool = True)`: Sets up the model for fine-tuning.

#### `L2Norm`

PyTorch module for L2 normalization.

**Key Methods:**

- `__init__(self, dim: int = 1)`: Initializes the normalization module.
- `forward(self, x: torch.Tensor)`: Applies L2 normalization to a tensor.

#### `ModelAdapterFactory`

Factory class for creating model adapters.

**Key Methods:**

- `create_adapter(model_type: str, **kwargs)`: Creates a model adapter of the specified type.
- `get_available_adapters()`: Returns a list of available adapter types.

### `model_config.py`

This file contains configuration classes for models and training.

#### `ModelConfig`

Base configuration class for models.

**Key Methods:**

- `__post_init__(self)`: Performs post-initialization validation.
- `validate(self)`: Validates the configuration.
- `to_dict(self)`: Converts the configuration to a dictionary.
- `from_dict(cls, config_dict: Dict[str, Any])`: Creates a configuration from a dictionary.

#### `ResNet50Config`

Configuration specific to ResNet50 models.

#### `QwenVLMConfig`

Configuration specific to Qwen Vision-Language Models.

#### `TrainingConfig`

Configuration for model training.

#### `DeploymentConfig`

Configuration for model deployment.

#### `ConfigManager`

Manages model and training configurations.

**Key Methods:**

- `__init__(self, config_dir: str = './config')`: Initializes the manager with a configuration directory.
- `load_config(self, filename: str, config_type: str)`: Loads a configuration from a file.
- `save_config(self, config: Union[ModelConfig, TrainingConfig, DeploymentConfig], filename: str)`: Saves a configuration to a file.
- `create_default_configs(self)`: Creates default configurations.
- `get_config_summary(self, config: Union[ModelConfig, TrainingConfig, DeploymentConfig])`: Returns a summary of a configuration.
- `get_device_config()`: Returns the device configuration.
- `validate_config_compatibility(self, model_config: ModelConfig, training_config: TrainingConfig)`: Validates compatibility between configurations.

**Static Methods:**

- `estimate_memory_requirements(config: ModelConfig)`: Estimates memory requirements for a model configuration.

### `trainer.py`

This file contains classes for training and fine-tuning models.

#### `IRImageDataset`

PyTorch dataset for IR images.

**Key Methods:**

- `__init__(self, images: List[IRImage], transform: Optional[Callable] = None)`: Initializes the dataset with images and optional transformations.
- `__len__(self)`: Returns the number of images in the dataset.
- `__getitem__(self, idx: int)`: Returns a specific image and its label by index.
- `_build_class_indices(self)`: Builds indices for classes.
- `get_triplet(self, anchor_idx: int)`: Gets a triplet (anchor, positive, negative) for training.

#### `TripletLoss`

Loss function for triplet-based training.

**Key Methods:**

- `__init__(self, margin: float = 0.3, hard_negative_mining: bool = True)`: Initializes the loss function.
- `forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor)`: Calculates the triplet loss.
- `_select_hard_negatives(self, anchor: torch.Tensor, negatives: torch.Tensor)`: Selects hard negative examples.

#### `TrainingMetrics`

Class for tracking and reporting training metrics.

#### `ModelTrainer`

Trains and fine-tunes models for IR image classification.

**Key Methods:**

- `__init__(self, model_adapter: BaseModelAdapter, config: TrainingConfig)`: Initializes the trainer with a model adapter and configuration.
- `__post_init__(self)`: Performs post-initialization setup.
- `train(self, train_images: List[IRImage], val_images: List[IRImage])`: Trains the model on the provided images.
- `evaluate_model(self, test_images: List[IRImage])`: Evaluates the model on test images.
- `load_checkpoint(self, checkpoint_path: str)`: Loads a training checkpoint.
- `setup_optimizer_and_scheduler(self)`: Sets up the optimizer and learning rate scheduler.
- `setup_logging(self)`: Sets up logging for training.
- `get_training_summary(self)`: Returns a summary of the training process.
- `plot_training_history(self, save_path: Optional[str] = None)`: Plots the training history.
- `meets_target_accuracy(self, target: float = 0.95)`: Checks if the model meets a target accuracy.
- `to_dict(self)`: Converts the trainer state to a dictionary.
- `validate(self)`: Validates the trainer configuration.

**Internal Methods:**

- `_train_epoch(self, train_loader: DataLoader, epoch: int)`: Trains for one epoch.
- `_validate_epoch(self, val_loader: DataLoader, epoch: int)`: Validates for one epoch.
- `_calculate_validation_metrics(self, embeddings: List[np.ndarray], classes: List[str])`: Calculates validation metrics.
- `_save_checkpoint(self, epoch: int, metrics: TrainingMetrics, is_best: bool = False)`: Saves a training checkpoint.
- `_log_epoch_progress(self, metrics: TrainingMetrics)`: Logs progress for an epoch.

### `validation_pipeline.py`

This file contains classes for validating trained models.

#### `ValidationResult`

Class representing the results of model validation.

#### `ValidationPipeline`

Pipeline for validating trained models.

**Key Methods:**

- `__init__(self, model_adapter: BaseModelAdapter, target_accuracy: float = 0.95)`: Initializes the pipeline with a model adapter and target accuracy.
- `validate_model(self, test_images: List[IRImage], class_labels: Optional[List[str]] = None)`: Validates the model on test images.
- `benchmark_performance(self, test_images: List[IRImage], target_time_per_image: float = 2.0)`: Benchmarks the model's performance.
- `validate_embedding_quality(self, embeddings: List[np.ndarray])`: Validates the quality of embeddings.
- `plot_validation_results(self, result: ValidationResult, save_path: Optional[str] = None)`: Plots validation results.
- `save_validation_report(self, result: ValidationResult, report_path: str)`: Saves a validation report.
- `to_dict(self)`: Converts the pipeline state to a dictionary.

**Internal Methods:**

- `_extract_embeddings_with_timing(self, images: List[IRImage])`: Extracts embeddings with timing information.
- `_classify_by_similarity(self, embeddings: List[np.ndarray], images: List[IRImage], class_labels: Optional[List[str]] = None)`: Classifies images by similarity.
- `_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray)`: Calculates cosine similarity between vectors.
- `_generate_recommendations(self, result: ValidationResult)`: Generates recommendations based on validation results.
- `_log_validation_results(self, result: ValidationResult)`: Logs validation results.

The training module provides comprehensive tools for training, fine-tuning, and validating deep learning models for IR image classification. It supports different model architectures and training strategies, with a focus on achieving high accuracy and performance for military applications.
