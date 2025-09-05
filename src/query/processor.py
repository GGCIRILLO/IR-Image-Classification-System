"""
Query processing engine for IR Image Classification System.

This module provides the QueryProcessor class that handles end-to-end query
processing from image input to similarity search results. Includes query
validation, preprocessing, embedding extraction, and result ranking.
"""
import os
import hashlib
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

from ..models.data_models import IRImage, Embedding, SimilarityResult, QueryResult
from ..models.interfaces import IImageProcessor, IEmbeddingExtractor, IVectorStore
from ..data.ir_processor import IRImageProcessor
from ..embedding.extractor import EmbeddingExtractor
from ..database.similarity_searcher import SimilaritySearcher, SearchConfig, SearchMode
from .ranker import ResultRanker, RankingConfig, RankingStrategy
from .confidence import ConfidenceCalculator, ConfidenceConfig, ConfidenceStrategy


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryValidationError(Exception):
    """Raised when query validation fails."""
    pass


class QueryProcessingError(Exception):
    """Raised when query processing fails."""
    pass


class QueryProcessor:
    """
    Main query processing engine for IR image similarity search.
    
    Handles the complete pipeline from image input to ranked similarity results:
    1. Query validation and preprocessing
    2. Image format validation and enhancement
    3. Embedding extraction
    4. Vector similarity search
    5. Result ranking and confidence scoring
    """
    
    def __init__(self, 
                 database_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 model_type: Optional[str] = None):
        """
        Initialize query processor with required components.
        
        Args:
            database_path: Path to the vector database (optional, auto-determined from model_type)
            model_path: Path to the fine-tuned embedding model
            collection_name: Name of the vector database collection (optional, auto-determined from model_type)
            config: Optional configuration parameters
            model_type: Model type for automatic database selection ('resnet18', 'resnet50')
        """
        self.config = config or {}
        
        # Determine model type from various sources
        self.model_type = (
            model_type or 
            self.config.get('model_type') or 
            'resnet50'  # Default fallback
        )
        
        # Auto-configure database path and collection name based on model type
        self.database_path, self.collection_name = self._get_database_config(
            database_path, collection_name, self.model_type
        )
        
        self.model_path = model_path
        
        # Initialize components
        self.image_processor: Optional[IImageProcessor] = None
        self.embedding_extractor: Optional[IEmbeddingExtractor] = None
        self.similarity_searcher: Optional[SimilaritySearcher] = None
        
        # Initialize ranking and confidence components
        ranking_config = RankingConfig(
            strategy=RankingStrategy(self.config.get('ranking_strategy', 'hybrid_score')),
            min_confidence=self.config.get('min_confidence_threshold', 0.7),
            max_results=self.config.get('top_k_results', 5),
            enable_diversity_filtering=self.config.get('enable_diversity_filtering', False)
        )
        self.result_ranker = ResultRanker(ranking_config)
        
        confidence_config = ConfidenceConfig(
            strategy=ConfidenceStrategy(self.config.get('confidence_strategy', 'ensemble')),
            min_confidence=self.config.get('min_confidence_threshold', 0.7),
            enable_adaptive_calibration=self.config.get('enable_confidence_calibration', True)
        )
        self.confidence_calculator = ConfidenceCalculator(confidence_config)
        
        # Query tracking
        self.query_history: List[QueryResult] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'processing_time': [],
            'preprocessing_time': [],
            'embedding_time': [],
            'search_time': []
        }
        
        # Configuration parameters
        self.max_query_time = self.config.get('max_query_time', 2.0)  # 2 seconds max
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.25)  # Lowered for IR images
        self.top_k_results = self.config.get('top_k_results', 5)
        self.enable_result_reranking = self.config.get('enable_result_reranking', True)
        self.cache_queries = self.config.get('cache_queries', True)
        
        # Query cache for performance
        self.query_cache: Dict[str, QueryResult] = {}
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        
        self.is_initialized = False
        logger.info(f"QueryProcessor created with model: {self.model_type}, database: {self.database_path}")
    
    def _get_database_config(self, 
                           database_path: Optional[str], 
                           collection_name: Optional[str], 
                           model_type: str) -> Tuple[str, str]:
        """
        Get database path and collection name based on model type.
        
        Args:
            database_path: Explicit database path (takes precedence)
            collection_name: Explicit collection name (takes precedence)
            model_type: Model type for auto-configuration
            
        Returns:
            Tuple[str, str]: (database_path, collection_name)
        """
        # Import here to avoid circular imports
        from ..database.db_manager import DatabaseManager, ModelType
        
        try:
            # Convert string to ModelType enum
            if model_type.lower() == 'resnet18':
                model_enum = ModelType.RESNET18
            elif model_type.lower() == 'resnet50':
                model_enum = ModelType.RESNET50
            else:
                logger.warning(f"Unknown model type '{model_type}', defaulting to ResNet50")
                model_enum = ModelType.RESNET50
            
            # Use DatabaseManager to get model-specific configuration
            db_manager = DatabaseManager(model_type=model_enum)
            
            # Use explicit values if provided, otherwise use auto-configured values
            final_database_path = database_path or db_manager.database_path
            final_collection_name = collection_name or db_manager.collection_name
            
            logger.info(f"Database config for {model_type}: {final_database_path} / {final_collection_name}")
            return final_database_path, final_collection_name
            
        except Exception as e:
            logger.warning(f"Failed to auto-configure database for {model_type}: {e}")
            # Fallback to defaults
            fallback_db_path = database_path or "./data/vector_db"
            fallback_collection = collection_name or "ir_embeddings"
            return fallback_db_path, fallback_collection
    
    def initialize(self) -> bool:
        """
        Initialize all query processor components.
        
        Returns:
            bool: True if initialization successful
            
        Raises:
            QueryProcessingError: If initialization fails
        """
        try:
            logger.info("Initializing QueryProcessor components...")
            
            # Initialize image processor
            self.image_processor = IRImageProcessor(
                target_size=(224, 224),
                preserve_aspect_ratio=False
            )
            logger.info("Image processor initialized")
            
            # Initialize embedding extractor with correct model type
            self.embedding_extractor = EmbeddingExtractor(model_type=self.model_type)
            if self.model_path:
                # Load custom model
                self.embedding_extractor.load_model(self.model_path)
                logger.info(f"Embedding extractor initialized with custom {self.model_type} model: {self.model_path}")
            else:
                # Load default pretrained model (without fine-tuning)
                try:
                    self.embedding_extractor.load_model(None)  # This will use the specified model type
                    logger.info(f"Embedding extractor initialized with default {self.model_type} model")
                except Exception as e:
                    logger.warning(f"Failed to load default {self.model_type} model: {e}")
                    logger.info("Embedding extractor will use basic feature extraction")
            
            # Initialize similarity searcher
            search_config = SearchConfig(
                mode=SearchMode.APPROXIMATE,
                k=self.top_k_results,
                confidence_threshold=self.min_confidence_threshold,
                max_search_time_ms=self.max_query_time * 1000,
                enable_reranking=self.enable_result_reranking,
                cache_queries=self.cache_queries
            )
            
            self.similarity_searcher = SimilaritySearcher(
                database_path=self.database_path,
                collection_name=self.collection_name
            )
            self.similarity_searcher.initialize(search_config)
            logger.info("Similarity searcher initialized")
            
            self.is_initialized = True
            logger.info("QueryProcessor initialization complete")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize QueryProcessor: {str(e)}"
            logger.error(error_msg)
            raise QueryProcessingError(error_msg) from e
    
    def process_query(self, 
                     image_input: Union[str, np.ndarray, "PILImage", IRImage],
                     query_id: Optional[str] = None,
                     options: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Process a complete image similarity query.
        
        Args:
            image_input: Input image (file path, numpy array, PIL Image, or IRImage)
            query_id: Optional unique identifier for the query
            options: Optional query-specific options
            
        Returns:
            QueryResult: Complete query results with top-K similar images
            
        Raises:
            QueryValidationError: If query validation fails
            QueryProcessingError: If processing fails
        """
        if not self.is_initialized:
            raise QueryProcessingError("QueryProcessor not initialized. Call initialize() first.")
        
        start_time = time.time()
        query_id = query_id or str(uuid.uuid4())
        options = options or {}
        
        logger.info(f"Processing query {query_id}")
        
        try:
            # Step 1: Validate and preprocess query
            preprocessing_start = time.time()
            processed_image = self._validate_and_preprocess_query(image_input, options)
            preprocessing_time = time.time() - preprocessing_start
            
            # Check cache if enabled
            cache_key = None
            if self.cache_queries:
                cache_key = self._generate_cache_key(processed_image, options)
                if cache_key in self.query_cache:
                    logger.info(f"Cache hit for query {query_id}")
                    cached_result = self.query_cache[cache_key]
                    # Update query_id and timestamp for cached result
                    cached_result.query_id = query_id
                    cached_result.timestamp = datetime.now()
                    return cached_result
            
            # Step 2: Extract embedding
            embedding_start = time.time()
            query_embedding = self._extract_query_embedding(processed_image)
            embedding_time = time.time() - embedding_start
            
            # Step 3: Perform similarity search
            search_start = time.time()
            similarity_results = self._perform_similarity_search(query_embedding, options)
            search_time = time.time() - search_start
            
            # Step 4: Rank and filter results
            final_results = self._rank_and_filter_results(similarity_results, options, query_embedding_vector=query_embedding)

            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Create query result
            query_result = QueryResult(
                query_id=query_id,
                results=final_results,
                processing_time=total_time,
                model_version=self._get_model_version(),
                timestamp=datetime.now()
            )
            
            # Update performance metrics
            self._update_performance_metrics(total_time, preprocessing_time, embedding_time, search_time)
            
            # Cache result if enabled
            if self.cache_queries and cache_key is not None:
                self._cache_query_result(cache_key, query_result)
            
            # Add to query history
            self.query_history.append(query_result)
            
            logger.info(f"Query {query_id} completed in {total_time:.3f}s with {len(final_results)} results")
            
            # Validate processing time requirement (< 2 seconds)
            if total_time > self.max_query_time:
                logger.warning(f"Query {query_id} exceeded max time limit: {total_time:.3f}s > {self.max_query_time}s")
            
            return query_result
            
        except QueryValidationError:
            raise
        except Exception as e:
            error_msg = f"Query processing failed for {query_id}: {str(e)}"
            logger.error(error_msg)
            raise QueryProcessingError(error_msg) from e
    
    def _validate_and_preprocess_query(self, 
                                     image_input: Union[str, np.ndarray, "PILImage", IRImage],
                                     options: Dict[str, Any]) -> np.ndarray:
        """
        Validate query input and preprocess image.
        
        Args:
            image_input: Input image in various formats
            options: Query options
            
        Returns:
            np.ndarray: Preprocessed image ready for embedding extraction
            
        Raises:
            QueryValidationError: If validation fails
        """
        try:
            # Convert input to numpy array
            if isinstance(image_input, str):
                # File path input
                if not Path(image_input).exists():
                    raise QueryValidationError(f"Image file not found: {image_input}")
                
                # Load image using IRImage for validation
                ir_image = IRImage.from_file_path(
                    file_path=image_input,
                    image_id="query_image",
                    object_class="unknown"
                )
                image_array = ir_image.image_data
                
            elif isinstance(image_input, IRImage):
                # Already processed IRImage
                image_array = image_input.image_data
                
            elif hasattr(image_input, 'mode') and hasattr(image_input, 'convert'):
                # PIL Image input - we know it's a PIL Image based on the attributes
                from PIL.Image import Image as PILImageType
                pil_image: PILImageType = image_input  # type: ignore
                
                # Convert to grayscale if needed
                if pil_image.mode != 'L':
                    pil_image = pil_image.convert('L')
                
                # Resize to target size
                pil_image = pil_image.resize((224, 224))
                
                # Convert to numpy array and normalize
                image_array = np.array(pil_image, dtype=np.float32) / 255.0
                
            elif isinstance(image_input, np.ndarray):
                # Numpy array input
                image_array = image_input.copy()
                
                # Ensure proper format
                if image_array.dtype != np.float32:
                    if image_array.max() > 1.0:
                        image_array = image_array.astype(np.float32) / 255.0
                    else:
                        image_array = image_array.astype(np.float32)
                
            else:
                raise QueryValidationError(f"Unsupported image input type: {type(image_input)}")
            
            # Validate image format for IR requirements
            if self.image_processor is not None:
                if not self.image_processor.validate_ir_format(image_array):
                    if options.get('strict_validation', True):
                        raise QueryValidationError("Image does not meet IR format requirements (white objects on black background)")
                    else:
                        logger.warning("Image may not meet IR format requirements - proceeding anyway")
            
                # Apply IR-specific preprocessing with fallback
                try:
                    processed_image = self.image_processor.preprocess_ir_image(image_array)
                except Exception as e:
                    logger.warning(f"IR preprocessing failed: {e}. Using simplified preprocessing")
                    # Fallback to simple preprocessing compatible with population method
                    from PIL import Image as PILImage
                    if len(image_array.shape) == 3:
                        pil_img = PILImage.fromarray(image_array.astype(np.uint8))
                        pil_img = pil_img.resize((224, 224))
                        processed_image = np.array(pil_img)
                    else:
                        pil_img = PILImage.fromarray((image_array * 255).astype(np.uint8))
                        pil_img = pil_img.resize((224, 224))
                        processed_image = np.array(pil_img, dtype=np.float32) / 255.0
            else:
                # Basic preprocessing if image processor not available
                processed_image = image_array
                if len(processed_image.shape) == 2:
                    # Ensure image is 224x224
                    if processed_image.shape != (224, 224):
                        from PIL import Image as PILImage
                        pil_img = PILImage.fromarray((processed_image * 255).astype(np.uint8))
                        pil_img = pil_img.resize((224, 224))
                        processed_image = np.array(pil_img, dtype=np.float32) / 255.0
            
            logger.debug(f"Query image validated and preprocessed: shape {processed_image.shape}")
            return processed_image
            
        except QueryValidationError:
            raise
        except Exception as e:
            raise QueryValidationError(f"Query validation failed: {str(e)}") from e
    
    def _extract_query_embedding(self, processed_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from preprocessed query image.
        
        Args:
            processed_image: Preprocessed image array
            
        Returns:
            np.ndarray: Embedding vector
            
        Raises:
            QueryProcessingError: If embedding extraction fails
        """
        try:
            if self.embedding_extractor is None:
                raise QueryProcessingError("Embedding extractor not initialized")
            
            # Extract embedding
            embedding_vector = self.embedding_extractor.extract_embedding(processed_image)
            
            # Validate embedding quality
            quality_score = self.embedding_extractor.validate_embedding_quality(embedding_vector)
            if quality_score < self.config.get('min_embedding_quality', 0.5):
                logger.warning(f"Low quality embedding detected: {quality_score:.3f}")
            
            logger.debug(f"Embedding extracted: shape {embedding_vector.shape}, quality {quality_score:.3f}")
            return embedding_vector
            
        except Exception as e:
            raise QueryProcessingError(f"Embedding extraction failed: {str(e)}") from e
    
    def _perform_similarity_search(self, 
                                 query_embedding: np.ndarray,
                                 options: Dict[str, Any]) -> List[SimilarityResult]:
        """
        Perform similarity search against vector database.
        
        Args:
            query_embedding: Query embedding vector
            options: Search options
            
        Returns:
            List[SimilarityResult]: Similarity search results
            
        Raises:
            QueryProcessingError: If search fails
        """
        try:
            # Override default search parameters if specified in options
            k = options.get('top_k', self.top_k_results)
            
            # Perform similarity search
            if self.similarity_searcher is not None:
                search_result = self.similarity_searcher.search_similar(
                    query_embedding=query_embedding,
                    k=k
                )
                # Handle tuple return (results, metrics)
                if isinstance(search_result, tuple):
                    results, metrics = search_result
                else:
                    results = search_result
            else:
                raise QueryProcessingError("Similarity searcher not initialized")
            
            print(f"Similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            raise QueryProcessingError(f"Similarity search failed: {str(e)}") from e
    
    def _rank_and_filter_results(self, 
                               results: List[SimilarityResult],
                               options: Dict[str, Any],
                               query_embedding_vector: Optional[np.ndarray] = None) -> List[SimilarityResult]:
        """
        Rank and filter similarity results using advanced ranking and confidence scoring.
        
        Args:
            results: Raw similarity results
            options: Filtering options
            query_embedding_vector: Original query embedding vector for confidence calculation
            
        Returns:
            List[SimilarityResult]: Filtered and ranked results with enhanced confidence scores
        """
        if not results:
            return []
        
        logger.debug(f"Ranking and filtering {len(results)} similarity results")
        
        # Update ranker configuration with options if provided
        if 'confidence_threshold' in options:
            from .ranker import RankingConfig
            updated_config = RankingConfig(
                strategy=self.result_ranker.config.strategy,
                min_confidence=options['confidence_threshold'],
                min_similarity=options.get('similarity_threshold', self.result_ranker.config.min_similarity),
                max_results=options.get('max_results', self.result_ranker.config.max_results),
                enable_diversity_filtering=self.result_ranker.config.enable_diversity_filtering,
                add_ranking_metadata=self.result_ranker.config.add_ranking_metadata
            )
            self.result_ranker.update_config(updated_config)
        
        # Create temporary Embedding object for confidence calculation if vector provided
        query_embedding = None
        if query_embedding_vector is not None:
            query_embedding = Embedding(
                id=f"query_{datetime.now().isoformat()}",
                vector=query_embedding_vector,
                image_id="query_image",
                model_version=self._get_model_version()
            )
        
        # Step 1: Enhanced confidence calculation for all results
        confidence_analyses = self.confidence_calculator.calculate_batch_confidence(
            results, query_embedding, options
        )
        
        # Update results with enhanced confidence scores
        for result, analysis in zip(results, confidence_analyses):
            result.confidence = analysis.final_confidence
            result.metadata.update({
                'confidence_explanation': analysis.explanation,
                'confidence_factors': analysis.confidence_factors,
                'uncertainty_estimate': analysis.uncertainty_estimate
            })
        
        # Step 2: Advanced ranking using ResultRanker
        query_context = {
            'query_options': options,
            'query_embedding': query_embedding,
            'total_candidates': len(results)
        }
        
        ranked_results, ranking_metrics = self.result_ranker.rank_results(
            results, query_context
        )
        
        # Step 3: Add final metadata
        for i, result in enumerate(ranked_results):
            result.metadata.update({
                'final_rank': i + 1,
                'ranking_timestamp': datetime.now().isoformat(),
                'ranking_metrics': {
                    'total_candidates': ranking_metrics.total_candidates,
                    'filtering_efficiency': ranking_metrics.final_results / max(ranking_metrics.total_candidates, 1),
                    'average_confidence': ranking_metrics.average_confidence
                }
            })
        
        logger.debug(f"Ranking complete: {len(ranked_results)} final results with "
                    f"avg confidence {ranking_metrics.average_confidence:.3f}")
        
        return ranked_results
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence score into human-readable levels."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.7:
            return "Medium"
        elif confidence >= 0.6:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_cache_key(self, image: np.ndarray, options: Dict[str, Any]) -> str:
        """Generate unique cache key for query."""
        # Create hash from image data and relevant options
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        options_str = str(sorted(options.items()))
        combined = f"{image_hash}_{options_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _cache_query_result(self, cache_key: str, result: QueryResult) -> None:
        """Cache query result with size management."""
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = result
    
    def _get_model_version(self) -> str:
        """Get current model version information."""
        if self.embedding_extractor and hasattr(self.embedding_extractor, 'get_model_info'):
            model_info = self.embedding_extractor.get_model_info()
            
            # Try to construct a meaningful version string from available info
            model_name = model_info.get('model_name', 'unknown')
            model_path = model_info.get('model_path', '')
            
            # If we have a custom model path, extract version info from it
            if model_path and model_path != "pretrained_default":
                # Extract filename from path
                filename = os.path.basename(model_path)
                # Remove extension
                version_info = os.path.splitext(filename)[0]
                return f"{model_name}_{version_info}"
            
            # Fallback to just model name if available
            if model_name != 'unknown':
                return model_name
            
            return model_info.get('version', 'unknown')
        return 'unknown'
    
    def _update_performance_metrics(self, total_time: float, preprocessing_time: float,
                                  embedding_time: float, search_time: float) -> None:
        """Update performance tracking metrics."""
        self.performance_metrics['processing_time'].append(total_time)
        self.performance_metrics['preprocessing_time'].append(preprocessing_time)
        self.performance_metrics['embedding_time'].append(embedding_time)
        self.performance_metrics['search_time'].append(search_time)
        
        # Keep only recent metrics (last 1000 queries)
        for metric_list in self.performance_metrics.values():
            if len(metric_list) > 1000:
                metric_list.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for the query processor.
        
        Returns:
            Dict[str, Dict[str, float]]: Performance statistics including
                                       average, min, max times for each metric
        """
        stats = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                stats[metric_name] = {
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'count': len(values)
                }
            else:
                stats[metric_name] = {
                    'average': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0
                }
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear query cache and reset performance metrics."""
        self.query_cache.clear()
        self.query_history.clear()
        for metric_list in self.performance_metrics.values():
            metric_list.clear()
        logger.info("Query cache and metrics cleared")
    
    def get_query_history(self, limit: Optional[int] = None) -> List[QueryResult]:
        """
        Get query history.
        
        Args:
            limit: Maximum number of recent queries to return
            
        Returns:
            List[QueryResult]: Recent query results
        """
        if limit:
            return self.query_history[-limit:]
        return self.query_history.copy()
    
    def validate_system_performance(self) -> Dict[str, bool]:
        """
        Validate that system meets performance requirements.
        
        Returns:
            Dict[str, bool]: Validation results for different performance criteria
        """
        stats = self.get_performance_stats()
        
        validation_results = {
            'meets_2_second_requirement': stats['processing_time']['average'] < 2.0,
            'consistent_performance': stats['processing_time']['std'] < 0.5,
            'reliable_preprocessing': stats['preprocessing_time']['average'] < 0.5,
            'efficient_embedding': stats['embedding_time']['average'] < 1.0,
            'fast_search': stats['search_time']['average'] < 0.3
        }
        
        return validation_results
    
    def update_ranking_config(self, new_config: RankingConfig) -> None:
        """Update ranking configuration."""
        self.result_ranker.update_config(new_config)
        logger.info("Ranking configuration updated")
    
    def update_confidence_config(self, new_config: ConfidenceConfig) -> None:
        """Update confidence calculation configuration."""
        self.confidence_calculator.update_config(new_config)
        logger.info("Confidence configuration updated")
    
    def get_ranking_statistics(self) -> Dict[str, Any]:
        """Get statistics about ranking performance."""
        return self.result_ranker.get_ranking_statistics()
    
    def get_confidence_calibration_metrics(self) -> Dict[str, float]:
        """Get confidence calibration metrics."""
        return self.confidence_calculator.get_calibration_metrics()
    
    def switch_model_type(self, new_model_type: str, new_model_path: Optional[str] = None) -> bool:
        """
        Switch to a different model type and corresponding database.
        
        Args:
            new_model_type: New model type ('resnet18' or 'resnet50')
            new_model_path: Optional path to fine-tuned model for the new type
            
        Returns:
            bool: True if switch was successful
        """
        try:
            logger.info(f"Switching from {self.model_type} to {new_model_type}")
            
            # Update model type and get new database configuration
            old_model_type = self.model_type
            self.model_type = new_model_type
            self.model_path = new_model_path
            
            # Get new database configuration
            self.database_path, self.collection_name = self._get_database_config(
                None, None, new_model_type
            )
            
            # Clear cache since we're switching models
            self.clear_cache()
            
            # Mark as uninitialized to force re-initialization
            self.is_initialized = False
            
            # Re-initialize with new configuration
            success = self.initialize()
            
            if success:
                logger.info(f"Successfully switched to {new_model_type} model")
                logger.info(f"New database: {self.database_path} / {self.collection_name}")
                return True
            else:
                # Rollback on failure
                logger.error(f"Failed to switch to {new_model_type}, rolling back to {old_model_type}")
                self.model_type = old_model_type
                self.database_path, self.collection_name = self._get_database_config(
                    None, None, old_model_type
                )
                self.initialize()  # Try to restore previous state
                return False
                
        except Exception as e:
            logger.error(f"Error switching model type: {e}")
            return False
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dict[str, Any]: Current model and database configuration
        """
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'database_path': self.database_path,
            'collection_name': self.collection_name,
            'is_initialized': self.is_initialized,
            'embedding_extractor_info': (
                self.embedding_extractor.get_model_info() 
                if self.embedding_extractor and hasattr(self.embedding_extractor, 'get_model_info')
                else None
            )
        }
    
    def update_confidence_accuracy(self, 
                                 predicted_confidence: float,
                                 actual_accuracy: float,
                                 object_class: str) -> None:
        """Update historical accuracy data for confidence calibration."""
        if hasattr(self.confidence_calculator, 'update_accuracy'):
            self.confidence_calculator.update_accuracy(
                predicted_confidence, actual_accuracy, object_class
            )


def create_query_processor_for_model(model_type: str, 
                                   model_path: Optional[str] = None,
                                   config_overrides: Optional[Dict[str, Any]] = None) -> QueryProcessor:
    """
    Factory function to create QueryProcessor for specific model type.
    
    Args:
        model_type: Model type ('resnet18' or 'resnet50')
        model_path: Optional path to fine-tuned model
        config_overrides: Optional configuration overrides
        
    Returns:
        QueryProcessor: Configured query processor for the specified model
    """
    from .config import QueryProcessorConfig
    
    # Create base configuration for the model type
    config = QueryProcessorConfig(model_type=model_type)
    
    # Apply any overrides
    if config_overrides:
        config_dict = config.to_dict()
        config_dict.update(config_overrides)
        config = QueryProcessorConfig.from_dict(config_dict)
    
    # Create and return processor
    processor = QueryProcessor(
        model_type=model_type,
        model_path=model_path,
        config=config.to_dict()
    )
    
    logger.info(f"Created QueryProcessor for {model_type} model")
    return processor


def create_resnet18_query_processor(model_path: Optional[str] = None,
                                  config_overrides: Optional[Dict[str, Any]] = None) -> QueryProcessor:
    """
    Convenience function to create ResNet18 QueryProcessor.
    
    Args:
        model_path: Optional path to fine-tuned ResNet18 model
        config_overrides: Optional configuration overrides
        
    Returns:
        QueryProcessor: ResNet18 query processor
    """
    return create_query_processor_for_model('resnet18', model_path, config_overrides)


def create_resnet50_query_processor(model_path: Optional[str] = None,
                                  config_overrides: Optional[Dict[str, Any]] = None) -> QueryProcessor:
    """
    Convenience function to create ResNet50 QueryProcessor.
    
    Args:
        model_path: Optional path to fine-tuned ResNet50 model
        config_overrides: Optional configuration overrides
        
    Returns:
        QueryProcessor: ResNet50 query processor
    """
    return create_query_processor_for_model('resnet50', model_path, config_overrides)
