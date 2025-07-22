# Examples Directory

This directory contains example scripts demonstrating how to use the IR Image Classification System's core components.

## Files

### `query_processor_examples.py`

Demonstrates how to use the QueryProcessor class for IR image classification:

- Loading and initializing the query processor
- Processing single and batch queries
- Configuring ranking strategies and confidence thresholds
- Handling query results and extracting information

Example usage:

```python
from src.query import QueryProcessor

# Initialize processor
processor = QueryProcessor(
    database_path="data/vector_db",
    model_path="checkpoints/fine_tuning/resnet50_ir_v2.pth"
)
processor.initialize()

# Process a query
result = processor.process_query("examples/tank.png")

# Display results
for i, match in enumerate(result.results, 1):
    print(f"{i}. {match.object_class}: {match.confidence:.3f}")
```

### `ranking_confidence_examples.py`

Shows how to work with the ranking and confidence calculation components:

- Using different ranking strategies (similarity, confidence, hybrid, military)
- Configuring confidence calculation methods
- Customizing thresholds and filters
- Analyzing ranking and confidence metrics

Example usage:

```python
from src.query.ranker import ResultRanker, RankingStrategy
from src.query.confidence import ConfidenceCalculator, ConfidenceStrategy

# Create ranker with military priority strategy
ranker = ResultRanker(strategy=RankingStrategy.MILITARY_PRIORITY)

# Create confidence calculator with ensemble strategy
confidence_calc = ConfidenceCalculator(strategy=ConfidenceStrategy.ENSEMBLE)

# Rank and filter results
ranked_results = ranker.rank_results(similarity_results)

# Calculate confidence scores
confidence_scores = confidence_calc.calculate_batch_confidence(similarity_results)
```

## Running Examples

To run these examples:

```bash
# Run query processor examples
python examples/query_processor_examples.py

# Run ranking and confidence examples
python examples/ranking_confidence_examples.py
```

## Creating New Examples

When adding new examples:

1. Create a new Python file with a descriptive name
2. Include detailed comments explaining the purpose and functionality
3. Add proper imports and error handling
4. Provide sample output or expected results
5. Update this README with information about the new example
