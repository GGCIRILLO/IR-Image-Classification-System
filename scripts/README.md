# Scripts Directory

This directory contains command-line utilities for the IR Image Classification System.

## Available Scripts

### `run_mission.py` - Mission Runner

A comprehensive command-line interface for executing IR image classification missions end-to-end.

#### Features

- **Multiple Configuration Presets**: Military, development, production, and testing configurations
- **Flexible Ranking Strategies**: similarity_only, confidence_weighted, hybrid_score, military_priority
- **Advanced Confidence Scoring**: similarity_based, statistical, ensemble, military_calibrated
- **Multiple Output Formats**: table, json, detailed, military report
- **Performance Monitoring**: Built-in validation and performance metrics
- **Military-Grade Operations**: Classification levels, operator tracking, mission IDs

#### Basic Usage

```bash
# Simple query
python scripts/run_mission.py --image query.png --database data/chroma_db_final

# Military deployment
python scripts/run_mission.py --image target.png --database data/chroma_db_final \
  --preset military --strategy military_priority \
  --confidence-strategy military_calibrated --format military

# Development testing
python scripts/run_mission.py --image test.png --database data/chroma_db_final \
  --preset development --debug --output results.json

# High-precision reconnaissance
python scripts/run_mission.py --image recon.png --database data/chroma_db_final \
  --confidence-threshold 0.9 --similarity-threshold 0.8 \
  --validation-mode strict --max-results 10
```

#### Command-Line Options

**Required Arguments:**

- `--image, -i`: Path to query image file (PNG, JPEG, TIFF, BMP)
- `--database, -d`: Path to vector database directory

**Model Configuration:**

- `--model, -m`: Path to fine-tuned model weights (.pth file)
- `--collection`: Vector database collection name (default: ir_embeddings)

**Configuration Presets:**

- `--preset, -p`: Use predefined configuration (military, development, production, testing)

**Ranking & Confidence:**

- `--strategy`: Ranking strategy (similarity_only, confidence_weighted, hybrid_score, military_priority)
- `--confidence-strategy`: Confidence calculation (similarity_based, statistical, ensemble, military_calibrated)

**Thresholds:**

- `--confidence-threshold`: Minimum confidence (0.0-1.0, default: 0.7)
- `--similarity-threshold`: Minimum similarity (0.0-1.0, default: 0.5)
- `--max-results`: Maximum results to return (default: 5)

**Processing Options:**

- `--max-query-time`: Maximum processing time in seconds (default: 2.0)
- `--validation-mode`: Image validation (strict, relaxed, disabled)
- `--disable-gpu`: Use CPU only
- `--disable-cache`: Disable result caching
- `--enable-diversity`: Enable diversity filtering

**Output Options:**

- `--output, -o`: Output file path (JSON format)
- `--format`: Output format (table, json, detailed, military)
- `--quiet, -q`: Suppress detailed output
- `--debug`: Enable debug output
- `--save-metadata`: Include detailed metadata

**Mission Parameters:**

- `--mission-id`: Custom mission identifier
- `--operator`: Operator/analyst name
- `--classification`: Mission classification level (UNCLASSIFIED, RESTRICTED, CONFIDENTIAL, SECRET, TOP_SECRET)

#### Output Formats

**Table Format (default):**
Simple tabular output with rank, object class, similarity, and confidence.

**JSON Format:**
Machine-readable JSON with full result metadata.

**Detailed Format:**
Comprehensive output with all available information and metadata.

**Military Format:**
Official military intelligence report style with threat assessment.

#### Examples

**1. Basic Object Identification:**

```bash
python scripts/run_mission.py \
  --image sample_tank.png \
  --database data/chroma_db_final
```

**2. Military Intelligence Operation:**

```bash
python scripts/run_mission.py \
  --image surveillance_image.png \
  --database data/chroma_db_final \
  --preset military \
  --strategy military_priority \
  --confidence-strategy military_calibrated \
  --format military \
  --operator "Analyst_Alpha" \
  --classification SECRET \
  --mission-id "OP_EAGLE_EYE_001"
```

**3. Development Testing:**

```bash
python scripts/run_mission.py \
  --image test_image.png \
  --database data/chroma_db_final \
  --preset development \
  --debug \
  --output test_results.json \
  --save-metadata
```

**4. High-Precision Analysis:**

```bash
python scripts/run_mission.py \
  --image critical_target.png \
  --database data/chroma_db_final \
  --confidence-threshold 0.9 \
  --similarity-threshold 0.8 \
  --validation-mode strict \
  --max-results 3 \
  --format detailed
```

#### Performance Monitoring

The script includes built-in performance validation that checks:

- Query processing time requirements
- System response times
- Resource utilization
- Model accuracy metrics

Use `--debug` to see detailed performance statistics.

#### Error Handling

The script provides comprehensive error handling and validation:

- Input file validation (format, existence)
- Parameter range validation
- Database connectivity checks
- Model loading verification
- Performance requirement validation

#### Testing

Run the test suite to verify functionality:

```bash
python tests/test_run_mission_script.py
```

This will test help output, input validation, and basic functionality with sample data.

## Integration with Other Components

The `run_mission.py` script integrates with:

- **QueryProcessor**: Core processing engine
- **ResultRanker**: Advanced ranking algorithms
- **ConfidenceCalculator**: Confidence scoring strategies
- **Vector Database**: Chroma DB for similarity search
- **Deep Learning Models**: PyTorch-based IR classification models

## Security Considerations

- **Classification Levels**: Built-in support for military classification standards
- **Operator Tracking**: Audit trail with operator identification
- **Mission IDs**: Unique identification for operational tracking
- **Secure Output**: Configurable output security based on classification level

## Requirements

- Python 3.8+
- All project dependencies (see requirements.txt)
- Initialized vector database
- Optional: Fine-tuned model weights

## Troubleshooting

**Common Issues:**

1. **Database not found**: Ensure the database path exists and is properly initialized
2. **Model loading errors**: Check model file path and compatibility
3. **Performance issues**: Consider using GPU acceleration and caching
4. **Memory errors**: Reduce batch size or max results for large queries

**Debug Mode:**
Use `--debug` flag for detailed error information and performance metrics.
