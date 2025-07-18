"""
Query processing engine for IR Image Classification System.

This module provides the QueryProcessor class that handles end-to-end query
processing from image input to similarity search results. Includes query
validation, preprocessing, embedding extraction, and result ranking.
"""

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
                 database_path: str,
                 model_path: Optional[str] = None,
                 collection_name: str = "ir_embeddings",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize query processor with required components.
        
        Args:
            database_path: Path to the vector database
            model_path: Path to the fine-tuned embedding model
            collection_name: Name of the vector database collection
            config: Optional configuration parameters
        """
        self.database_path = database_path
        self.model_path = model_path
        self.collection_name = collection_name
        self.config = config or {}
        
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
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.7)
        self.top_k_results = self.config.get('top_k_results', 5)
        self.enable_result_reranking = self.config.get('enable_result_reranking', True)
        self.cache_queries = self.config.get('cache_queries', True)
        
        # Query cache for performance
        self.query_cache: Dict[str, QueryResult] = {}
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        
        self.is_initialized = False
        logger.info(f"QueryProcessor created with database: {database_path}")
    
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
            
            # Initialize embedding extractor
            self.embedding_extractor = EmbeddingExtractor()
            if self.model_path:
                # Load custom model
                self.embedding_extractor.load_model(self.model_path)
                logger.info(f"Embedding extractor initialized with custom model: {self.model_path}")
            else:
                # Load default pretrained model (without fine-tuning)
                try:
                    self.embedding_extractor.load_model(None)  # This will use base ResNet50
                    logger.info("Embedding extractor initialized with default ResNet50 model")
                except Exception as e:
                    logger.warning(f"Failed to load default model: {e}")
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
            print(f"   📊 Similarity search completed by processor: {len(similarity_results)} results found")
            search_time = time.time() - search_start
            
            # Step 4: Rank and filter results
            final_results = self._rank_and_filter_results(similarity_results, options, query_embedding_vector=query_embedding)

            print(f"   📊 Final results after ranking and filtering in Processor: {len(final_results)} items found")
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
                print(f"   📊 Similarity search completed by searcher: {search_result}")
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
    
    def update_confidence_accuracy(self, 
                                 predicted_confidence: float,
                                 actual_accuracy: float,
                                 object_class: str) -> None:
        """Update historical accuracy data for confidence calibration."""
        self.confidence_calculator.update_historical_accuracy(
            predicted_confidence, actual_accuracy, object_class
        )
