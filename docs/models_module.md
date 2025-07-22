# Models Module Documentation

The `models` module defines the core data structures, interfaces, and object classification taxonomy used throughout the IR image classification system. It provides a foundation for representing and working with IR images, embeddings, and classification results.

## Files and Classes

### `data_models.py`

This file contains the core data classes used throughout the system.

#### `IRImage`

Represents an infrared image with its associated metadata.

**Key Methods:**

- `__post_init__(self)`: Performs post-initialization validation and processing.
- `from_file_path(cls, file_path: str, image_id: str, object_class: str = "")`: Creates an IRImage instance from a file path.
- `to_pil_image(self)`: Converts the image data to a PIL Image.
- `get_object_class_info(self)`: Returns information about the object class.
- `get_object_category(self)`: Returns the category of the object.
- `get_threat_level(self)`: Returns the threat level associated with the object.
- `is_military_asset(self)`: Checks if the object is a military asset.
- `is_critical_asset(self)`: Checks if the object is a critical asset.
- `_extract_main_object_type(self, object_class: str)`: Extracts the main object type from a class name.

#### `Embedding`

Represents a vector embedding of an image with associated metadata.

**Key Methods:**

- `__post_init__(self)`: Performs post-initialization validation.
- `validate(self)`: Validates the embedding vector.
- `normalize(self)`: Normalizes the embedding vector.
- `cosine_similarity(self, other: 'Embedding')`: Calculates cosine similarity with another embedding.

#### `SimilarityResult`

Represents a single result from a similarity search.

#### `QueryResult`

Represents the complete results of a query, including multiple similarity results.

**Key Methods:**

- `get_top_k(self, k: int = 5)`: Returns the top k results.
- `get_high_confidence_results(self, min_confidence: float = 0.8)`: Returns results with confidence above a threshold.

### `interfaces.py`

This file defines interfaces and base classes for the system's components.

#### Interfaces:

- `IImageProcessor`: Interface for image processing components.
- `IEmbeddingExtractor`: Interface for embedding extraction components.
- `IVectorStore`: Interface for vector storage components.
- `IDataAugmentation`: Interface for data augmentation components.

#### Base Classes:

- `BaseImageProcessor`: Base implementation of the image processor interface.
- `BaseEmbeddingExtractor`: Base implementation of the embedding extractor interface.
- `BaseVectorStore`: Base implementation of the vector store interface.
- `BaseDataAugmentation`: Base implementation of the data augmentation interface.

Each base class provides a partial implementation of its respective interface, with common functionality and validation methods.

### `object_classes.py`

This file defines the taxonomy of object classes for IR image classification.

#### `ObjectCategory`

An enumeration of object categories (e.g., MILITARY, CIVILIAN, INFRASTRUCTURE).

#### `ObjectClass`

Represents a specific object class with its properties.

**Key Methods:**

- `__post_init__(self)`: Performs post-initialization validation.

#### `ObjectClassRegistry`

A registry of all object classes in the system.

**Key Methods:**

- `__init__(self)`: Initializes the registry.
- `_initialize_classes(self)`: Initializes the predefined object classes.
- `get_class_by_name(self, name: str)`: Gets an object class by its name.
- `get_class_by_folder(self, folder_name: str)`: Gets an object class by its folder name.
- `get_classes_by_category(self, category: ObjectCategory)`: Gets all classes in a category.
- `get_all_classes(self)`: Gets all registered object classes.
- `get_class_names(self)`: Gets all class names.
- `get_folder_names(self)`: Gets all folder names.
- `get_category_distribution(self)`: Gets the distribution of classes by category.
- `__len__(self)`: Returns the number of registered classes.
- `__iter__(self)`: Provides an iterator over the registered classes.

**Static Methods:**

- `get_object_classes()`: Returns all predefined object classes.
- `get_class_id_mapping()`: Returns a mapping from class IDs to class names.
- `get_id_class_mapping()`: Returns a mapping from class names to class IDs.

The models module provides the foundational data structures and interfaces that enable the system to represent and process IR images, embeddings, and classification results in a consistent and type-safe manner. It also defines the taxonomy of object classes that the system can recognize and classify.
