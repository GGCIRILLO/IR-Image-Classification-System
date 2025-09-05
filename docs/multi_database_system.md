# Multi-Database System Documentation

## Overview

The IR Image Classification system now supports multiple vector databases for different model types (ResNet18, ResNet50). This enhancement allows for:

- Separate databases for different model architectures
- Dynamic switching between databases
- Backward compatibility with existing code
- Centralized configuration management

## Architecture

### Database Configuration

The system uses a centralized configuration file at `config/multi_database_config.json` that defines:

```json
{
  "databases": {
    "resnet50": {
      "database_path": "./data/vector_db",
      "collection_name": "ir_embeddings",
      "embedding_dimension": 512,
      "description": "ResNet50 model database (existing/legacy)"
    },
    "resnet18": {
      "database_path": "./data/vector_db_resnet18",
      "collection_name": "ir_embeddings_resnet18",
      "embedding_dimension": 512,
      "description": "ResNet18 model database (new)"
    }
  }
}
```

### Enhanced Components

#### DatabaseManager

- **Multi-database support**: Manages multiple databases for different model types
- **Dynamic switching**: Switch between databases at runtime
- **Batch operations**: Initialize, backup, and manage all databases
- **Status monitoring**: Check status of all configured databases

#### SimilaritySearcher

- **Model-aware**: Automatically uses correct database for specified model type
- **Dynamic switching**: Change databases without recreating instances
- **Legacy compatibility**: Maintains existing API for backward compatibility

## Usage Examples

### Using DatabaseManager

```python
from src.database.db_manager import DatabaseManager, ModelType

# Create model-specific managers
resnet18_manager = DatabaseManager(model_type=ModelType.RESNET18)
resnet50_manager = DatabaseManager(model_type=ModelType.RESNET50)

# Initialize databases for all model types
manager = DatabaseManager()
results = manager.initialize_all_databases()

# Check status of all databases
manager.print_multi_database_status()

# Switch between databases
manager.switch_database(ModelType.RESNET18)
```

### Using SimilaritySearcher

```python
from src.database.similarity_searcher import SimilaritySearcher, ModelType

# Create model-specific searchers
resnet18_searcher = SimilaritySearcher(model_type=ModelType.RESNET18)
resnet50_searcher = SimilaritySearcher(model_type=ModelType.RESNET50)

# Switch databases dynamically
searcher = SimilaritySearcher(model_type=ModelType.RESNET50)
searcher.switch_database(ModelType.RESNET18)

# Legacy compatibility (still works)
legacy_searcher = SimilaritySearcher(
    database_path="./data/vector_db",
    collection_name="ir_embeddings"
)
```

### Factory Functions

```python
from src.database.db_manager import create_resnet18_manager, create_resnet50_manager
from src.database.similarity_searcher import create_resnet18_searcher, create_resnet50_searcher

# Easy creation of model-specific instances
resnet18_manager = create_resnet18_manager()
resnet18_searcher = create_resnet18_searcher()
```

## Database Paths

| Model Type | Database Path               | Collection Name          |
| ---------- | --------------------------- | ------------------------ |
| ResNet50   | `./data/vector_db`          | `ir_embeddings`          |
| ResNet18   | `./data/vector_db_resnet18` | `ir_embeddings_resnet18` |

## Migration Guide

### For Existing Code

Existing code continues to work without changes:

```python
# This still works exactly as before
searcher = SimilaritySearcher("./data/vector_db", "ir_embeddings")
manager = DatabaseManager("./data/vector_db", "ir_embeddings")
```

### For New Code

New code can take advantage of model-aware features:

```python
# New model-aware approach
searcher = SimilaritySearcher(model_type=ModelType.RESNET18)
manager = DatabaseManager(model_type=ModelType.RESNET18)
```

## Configuration Management

### Automatic Configuration

The system automatically creates the multi-database configuration file on first use with sensible defaults.

### Manual Configuration

You can manually edit `config/multi_database_config.json` to:

- Add new model types
- Change database paths
- Modify collection names
- Update embedding dimensions

### Validation

The system validates configurations and provides helpful error messages for invalid setups.

## Benefits

1. **Isolation**: Each model type has its own database, preventing conflicts
2. **Performance**: Optimized database configurations per model type
3. **Scalability**: Easy to add new model types
4. **Compatibility**: Existing code continues to work unchanged
5. **Flexibility**: Dynamic switching between databases
6. **Management**: Centralized configuration and batch operations

## Testing

Run the multi-database test to verify functionality:

```bash
python scripts/test_multi_database.py
```

This test verifies:

- Database manager functionality
- Similarity searcher enhancements
- Configuration file creation
- Dynamic switching capabilities
- Legacy compatibility

## Requirements Satisfied

✅ **2.1**: Multiple database support with distinct paths  
✅ **2.2**: Model type-based database selection  
✅ **2.3**: Existing ResNet50 database remains unchanged and functional
