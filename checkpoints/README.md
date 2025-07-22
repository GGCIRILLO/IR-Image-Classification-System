# Checkpoints Directory

This directory stores model checkpoints and fine-tuning results for the IR Image Classification System.

## Structure

### `fine_tuning/`

Contains fine-tuned model weights and configurations:

- `resnet50_ir_v1.pth`: Base ResNet50 model fine-tuned on IR images
- `resnet50_ir_v2.pth`: Enhanced ResNet50 model with improved accuracy
- `qwen_vlm_ir.pth`: Qwen Vision Language Model fine-tuned for IR classification
- `model_configs/`: Configuration files for each fine-tuned model

### `logs/`

Training and fine-tuning logs:

- `training_logs/`: Detailed training progress logs
- `tensorboard/`: TensorBoard event files for visualization
- `metrics/`: Performance metrics for each training run
- `hyperparameter_search/`: Results from hyperparameter optimization

## Model Versions

### ResNet50 IR v1

- Base ResNet50 architecture fine-tuned on IR images
- 512-dimensional embeddings
- 85% accuracy on validation set
- Optimized for general IR object classification

### ResNet50 IR v2

- Enhanced ResNet50 with custom IR-specific layers
- 512-dimensional embeddings with improved normalization
- 92% accuracy on validation set
- Optimized for military object classification

### Qwen VLM IR

- Qwen Vision Language Model adapted for IR images
- 768-dimensional embeddings
- 94% accuracy on validation set
- Better performance on complex scenes and ambiguous objects

## Usage

To use a checkpoint in the IR Image Classification System:

```bash
# Run mission with specific model
python scripts/run_mission.py --image query.png \
  --database data/chroma_db_final \
  --model checkpoints/fine_tuning/resnet50_ir_v2.pth
```

To continue fine-tuning from a checkpoint:

```bash
python scripts/enhanced_fine_tuning.py \
  --train-data data/processed \
  --database data/chroma_db_final \
  --model-type resnet50 \
  --checkpoint checkpoints/fine_tuning/resnet50_ir_v1.pth \
  --epochs 50
```

## Model Selection Guidelines

- **ResNet50 IR v1**: Use for general-purpose IR classification with lower computational requirements
- **ResNet50 IR v2**: Use for military applications requiring higher accuracy
- **Qwen VLM IR**: Use for complex scenes or when highest accuracy is required (needs more GPU memory)

## Adding New Checkpoints

When adding new model checkpoints:

1. Save the model with a descriptive name including architecture and version
2. Include a configuration file with hyperparameters and training details
3. Document the model's performance metrics and intended use cases
4. Update this README with information about the new model
