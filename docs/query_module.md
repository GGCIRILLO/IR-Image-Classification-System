# Query Module Documentation

The `query` module provides functionality for processing IR image queries, ranking results, and calculating confidence scores. It serves as the main interface for performing image similarity searches and interpreting the results.

## Files and Classes

### `processor.py`

This file contains the `QueryProcessor` class which orchestrates the entire query processing pipeline.

#### `QueryProcessor`

The main class responsible for processing IR image queries and returning ranked results.

**Key Methods:**

- `__init__(self, database_path: str, model_path: Optional[str] = None, collection_name: str = "ir_embeddings", config: Optional[Dict[str, Any]] = None)`: Initializes the query processor.
- `initialize(self)`: Initializes the components needed for query processing.
- `process_query(self, image_input: Union[str, np.ndarray, "PILImage", IRImage], query_id: Optional[str] = None, options: Optional[Dict[str, Any]] = None)`: Processes a query image and returns similarity results.
- `get_performance_stats(self)`: Returns performance statistics for query processing.
- `get_query_history(self, limit: Optional[int] = None)`: Returns the history of processed queries.
- `clear_cache(self)`: Clears the query cache.
- `update_confidence_config(self, new_config: ConfidenceConfig)`: Updates the confidence calculation configuration.
- `update_ranking_config(self, new_config: RankingConfig)`: Updates the result ranking configuration.
- `update_confidence_accuracy(self, predicted_confidence: float, actual_accuracy: float, object_class: str)`: Updates the confidence calculation based on feedback.
- `get_confidence_calibration_metrics(self)`: Returns metrics about confidence calibration.
- `get_ranking_statistics(self)`: Returns statistics about result ranking.
- `validate_system_performance(self)`: Validates the overall system performance.

**Internal Methods:**

- `_validate_and_preprocess_query(self, image_input: Union[str, np.ndarray, "PILImage", IRImage], options: Dict[str, Any])`: Validates and preprocesses the query image.
- `_extract_query_embedding(self, processed_image: np.ndarray)`: Extracts an embedding from the processed query image.
- `_perform_similarity_search(self, query_embedding: np.ndarray, options: Dict[str, Any])`: Performs a similarity search using the query embedding.
- `_rank_and_filter_results(self, results: List[SimilarityResult], options: Dict[str, Any], query_embedding_vector: Optional[np.ndarray] = None)`: Ranks and filters the similarity results.
- `_categorize_confidence(self, confidence: float)`: Categorizes a confidence score into a confidence level.
- `_generate_cache_key(self, image: np.ndarray, options: Dict[str, Any])`: Generates a cache key for the query.
- `_cache_query_result(self, cache_key: str, result: QueryResult)`: Caches a query result.
- `_get_model_version(self)`: Gets the version of the embedding model.
- `_update_performance_metrics(self, total_time: float, preprocessing_time: float, embedding_time: float, search_time: float)`: Updates performance metrics.

#### `QueryProcessingError`

Exception raised when an error occurs during query processing.

#### `QueryValidationError`

Exception raised when a query fails validation.

### `ranker.py`

This file contains classes for ranking and filtering similarity search results.

#### `RankingConfig`

Configuration class for result ranking settings.

#### `RankingStrategy`

Enumeration of different ranking strategies.

#### `ConfidenceLevel`

Enumeration of confidence levels (e.g., HIGH, MEDIUM, LOW).

#### `RankingMetrics`

Class for tracking and reporting ranking performance metrics.

#### `ResultRanker`

Ranks and filters similarity search results based on various criteria.

**Key Methods:**

- `__init__(self, config: Optional[RankingConfig] = None)`: Initializes the ranker with optional configuration.
- `rank_results(self, results: List[SimilarityResult], query_context: Optional[Dict[str, Any]] = None)`: Ranks and filters results.
- `get_ranking_statistics(self)`: Returns statistics about ranking performance.
- `reset_metrics(self)`: Resets the ranking metrics.
- `update_config(self, new_config: RankingConfig)`: Updates the ranking configuration.
- `validate(self)`: Validates the ranker configuration.

**Ranking Methods:**

- `_rank_by_similarity(self, results: List[SimilarityResult], context: Dict[str, Any])`: Ranks results by similarity score.
- `_rank_by_confidence_weighted(self, results: List[SimilarityResult], context: Dict[str, Any])`: Ranks results by confidence-weighted score.
- `_rank_by_hybrid_score(self, results: List[SimilarityResult], context: Dict[str, Any])`: Ranks results by a hybrid score.
- `_rank_by_military_priority(self, results: List[SimilarityResult], context: Dict[str, Any])`: Ranks results with priority for military objects.

**Static Methods:**

- `weighted_score(result: SimilarityResult)`: Calculates a weighted score for a result.
- `hybrid_score(result: SimilarityResult)`: Calculates a hybrid score for a result.
- `military_priority_score(result: SimilarityResult)`: Calculates a military priority score for a result.

**Internal Methods:**

- `_apply_initial_filters(self, results: List[SimilarityResult])`: Applies initial filters to results.
- `_apply_diversity_filtering(self, results: List[SimilarityResult])`: Applies diversity filtering to results.
- `_add_ranking_metadata(self, results: List[SimilarityResult])`: Adds ranking metadata to results.
- `_calculate_confidence_boost(self, confidence: float)`: Calculates a confidence boost factor.
- `_categorize_confidence(self, confidence: float)`: Categorizes a confidence score.
- `_generate_confidence_explanation(self, result: SimilarityResult)`: Generates an explanation for a confidence score.
- `_calculate_ranking_metrics(self, total_candidates: int, filtered_results: int, final_results: int, ranking_time: float, results: List[SimilarityResult])`: Calculates ranking metrics.

### `confidence.py`

This file contains classes for calculating and analyzing confidence scores for similarity results.

#### `ConfidenceConfig`

Configuration class for confidence calculation settings.

#### `ConfidenceStrategy`

Enumeration of different confidence calculation strategies.

#### `ConfidenceFactors`

Class representing the factors that contribute to a confidence score.

#### `ConfidenceMetrics`

Class for tracking and reporting confidence calculation metrics.

#### `ConfidenceCalculator`

Calculates confidence scores for similarity results.

**Key Methods:**

- `__init__(self, config: Optional[ConfidenceConfig] = None)`: Initializes the calculator with optional configuration.
- `calculate_confidence(self, result: SimilarityResult, all_results: Optional[List[SimilarityResult]] = None, query_embedding: Optional[Embedding] = None, context: Optional[Dict[str, Any]] = None)`: Calculates a confidence score for a result.
- `calculate_batch_confidence(self, results: List[SimilarityResult], query_embedding: Optional[Embedding] = None, context: Optional[Dict[str, Any]] = None)`: Calculates confidence scores for multiple results.
- `update_config(self, new_config: ConfidenceConfig)`: Updates the confidence configuration.
- `update_historical_accuracy(self, predicted_confidence: float, actual_accuracy: float, object_class: str)`: Updates historical accuracy data.
- `reset_historical_data(self)`: Resets historical accuracy data.
- `get_calibration_metrics(self)`: Returns metrics about confidence calibration.
- `validate(self)`: Validates the calculator configuration.

**Confidence Calculation Methods:**

- `_calculate_similarity_confidence(self, result: SimilarityResult, all_results: Optional[List[SimilarityResult]], query_embedding: Optional[Embedding], context: Dict[str, Any])`: Calculates confidence based on similarity.
- `_calculate_statistical_confidence(self, result: SimilarityResult, all_results: Optional[List[SimilarityResult]], query_embedding: Optional[Embedding], context: Dict[str, Any])`: Calculates confidence based on statistics.
- `_calculate_ensemble_confidence(self, result: SimilarityResult, all_results: Optional[List[SimilarityResult]], query_embedding: Optional[Embedding], context: Dict[str, Any])`: Calculates confidence using an ensemble approach.
- `_calculate_military_confidence(self, result: SimilarityResult, all_results: Optional[List[SimilarityResult]], query_embedding: Optional[Embedding], context: Dict[str, Any])`: Calculates confidence with military-specific factors.

**Factor Methods:**

- `_get_similarity_factor(self, similarity: float)`: Gets a confidence factor based on similarity.
- `_get_distribution_factor(self, result: SimilarityResult, all_results: Optional[List[SimilarityResult]])`: Gets a confidence factor based on result distribution.
- `_get_quality_factor(self, query_embedding: Optional[Embedding], context: Dict[str, Any])`: Gets a confidence factor based on embedding quality.
- `_get_historical_factor(self, object_class: str)`: Gets a confidence factor based on historical accuracy.

**Internal Methods:**

- `_apply_confidence_bounds(self, confidence: float)`: Applies bounds to a confidence score.
- `_calculate_ensemble_uncertainty(self, factors: Dict[str, float], weights: Dict[str, float])`: Calculates uncertainty in an ensemble confidence score.
- `_is_critical_object(self, object_class: str)`: Checks if an object class is critical.

#### `ConfidenceAnalysis`

Class for analyzing confidence scores and their relationship to accuracy.

### `config.py`

This file contains configuration classes for the query module.

#### `QueryProcessorConfig`

Configuration class for the query processor.

#### `ValidationMode`

Enumeration of validation modes.

#### `CachePolicy`

Enumeration of cache policies.

#### `MilitaryQueryConfig`

Specialized configuration for military queries.

#### `DevelopmentQueryConfig`

Specialized configuration for development queries.

**Common Methods:**

- `validate(self)`: Validates the configuration.
- `to_dict(self)`: Converts the configuration to a dictionary.
- `from_dict(cls, config_dict: Dict[str, Any])`: Creates a configuration from a dictionary.
- `get_config_for_environment(environment: str)`: Gets a configuration for a specific environment.

The query module is the main interface for performing IR image similarity searches and interpreting the results. It orchestrates the entire query processing pipeline, from image preprocessing to result ranking and confidence calculation.
