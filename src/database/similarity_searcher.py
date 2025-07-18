"""
Similarity search implementation for IR image embeddings.

This module provides the SimilaritySearcher class for performing efficient
similarity searches using both exact and approximate methods. Optimized
for military IR image classification with cosine similarity.
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings

from ..models.data_models import Embedding, SimilarityResult


class SearchMode(Enum):
    """Search mode enumeration."""
    EXACT = "exact"
    APPROXIMATE = "approximate"
    HYBRID = "hybrid"  # Try approximate first, fallback to exact if needed


@dataclass
class SearchConfig:
    """Configuration for similarity search operations."""
    mode: SearchMode = SearchMode.APPROXIMATE
    distance_metric: str = "cosine"  # cosine, euclidean, inner_product
    k: int = 5  # Number of results to return
    confidence_threshold: float = 0.7  # Minimum confidence for results
    max_search_time_ms: float = 2000.0  # Maximum search time in milliseconds
    enable_reranking: bool = True  # Enable result re-ranking
    cache_queries: bool = True  # Cache query results


@dataclass
class SearchMetrics:
    """Metrics for search performance tracking."""
    search_time_ms: float
    total_candidates_examined: int
    results_returned: int
    cache_hit: bool
    confidence_scores: List[float]
    search_mode_used: SearchMode


class SimilaritySearcher:
    """
    High-performance similarity searcher for IR image embeddings.
    
    Provides both exact and approximate search capabilities with intelligent
    fallback mechanisms and performance optimization for military applications.
    """
    
    def __init__(self, database_path: str, collection_name: str = "ir_embeddings"):
        """
        Initialize similarity searcher.
        
        Args:
            database_path: Path to the ChromaDB database
            collection_name: Name of the collection to search
        """
        self.database_path = database_path
        self.collection_name = collection_name
        self.client = None  # chromadb.PersistentClient
        self.collection = None  # chromadb Collection
        self.config = SearchConfig()
        self.query_cache = {}  # Simple query result cache
        self.is_initialized = False
        self.search_metrics = []
        
    def initialize(self, config: Optional[SearchConfig] = None) -> bool:
        """
        Initialize the similarity searcher.
        
        Args:
            config: Optional search configuration
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            if config:
                self.config = config
            
            # Connect to ChromaDB
            self.client = chromadb.PersistentClient(
                path=self.database_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            # Get the collection
            self.collection = self.client.get_collection(name=self.collection_name)
            
            self.is_initialized = True
            print(f"SimilaritySearcher initialized for collection: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize SimilaritySearcher: {str(e)}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, 
                      k: Optional[int] = None,
                      mode: Optional[SearchMode] = None,
                      filters: Optional[Dict[str, Any]] = None) -> Tuple[List[SimilarityResult], SearchMetrics]:
        """
        Search for similar embeddings using specified mode.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return (uses config default if None)
            mode: Search mode to use (uses config default if None)
            filters: Optional metadata filters
            
        Returns:
            Tuple[List[SimilarityResult], SearchMetrics]: Results and search metrics
        """
        if not self.is_initialized:
            raise RuntimeError("SimilaritySearcher must be initialized before searching")
        
        # Use provided parameters or defaults from config
        k = k or self.config.k
        mode = mode or self.config.mode
        
        start_time = time.time()
        
        # Check cache first if enabled
        cache_key = self._generate_cache_key(query_embedding, k, mode, filters)
        if self.config.cache_queries and cache_key in self.query_cache:
            cached_results, cached_metrics = self.query_cache[cache_key]
            cached_metrics.cache_hit = True
            return cached_results, cached_metrics
        
        # Validate query embedding
        self._validate_query_embedding(query_embedding)
        
        # Perform search based on mode
        try:
            if mode == SearchMode.EXACT:
                results, total_candidates = self._exact_search(query_embedding, k, filters)
            elif mode == SearchMode.APPROXIMATE:
                results, total_candidates = self._approximate_search(query_embedding, k, filters)
            elif mode == SearchMode.HYBRID:
                results, total_candidates = self._hybrid_search(query_embedding, k, filters)
            else:
                raise ValueError(f"Unsupported search mode: {mode}")
            
            search_time_ms = (time.time() - start_time) * 1000
     
            # Create search metrics
            metrics = SearchMetrics(
                search_time_ms=search_time_ms,
                total_candidates_examined=total_candidates,
                results_returned=len(results),
                cache_hit=False,
                confidence_scores=[r.confidence for r in results],
                search_mode_used=mode
            )
            
            # Cache results if enabled
            if self.config.cache_queries:
                self.query_cache[cache_key] = (results, metrics)
                # Limit cache size
                if len(self.query_cache) > 1000:
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
            
            # Store metrics for analysis
            self.search_metrics.append(metrics)

            return results, metrics

        except Exception as e:
            search_time_ms = (time.time() - start_time) * 1000
            print(f"Search failed: {str(e)}")
            
            error_metrics = SearchMetrics(
                search_time_ms=search_time_ms,
                total_candidates_examined=0,
                results_returned=0,
                cache_hit=False,
                confidence_scores=[],
                search_mode_used=mode
            )
            
            return [], error_metrics
    
    def _exact_search(self, query_embedding: np.ndarray, k: int, 
                     filters: Optional[Dict[str, Any]]) -> Tuple[List[SimilarityResult], int]:
        """
        Perform exact similarity search.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Tuple[List[SimilarityResult], int]: Results and total candidates examined
        """
        # ChromaDB performs exact search by default for small collections
        # For large collections, we might need to implement our own exact search
        
        # Prepare where clause for filtering
        where_clause = self._build_where_clause(filters) if filters else None
        
        # Check collection is available
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['metadatas', 'distances'],
            where=where_clause
        )
        
        # Convert to SimilarityResult objects
        similarity_results = self._convert_chroma_results(results)
        
        # For exact search, we examine all vectors (or all matching filters)
        total_examined = self.collection.count()
        if where_clause:
            # Estimate filtered count (ChromaDB doesn't provide this directly)
            total_examined = min(total_examined, k * 10)  # Rough estimate
        
        return similarity_results, total_examined
    
    def _approximate_search(self, query_embedding: np.ndarray, k: int,
                           filters: Optional[Dict[str, Any]]) -> Tuple[List[SimilarityResult], int]:
        """
        Perform approximate similarity search using HNSW.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Tuple[List[SimilarityResult], int]: Results and total candidates examined
        """
        # ChromaDB uses HNSW by default for approximate search
        where_clause = self._build_where_clause(filters) if filters else None
        
        print(f"Performing approximate search with k={k}, filters={filters}")
        
        # Check collection is available
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        # Perform approximate search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['metadatas', 'distances'],
            where=where_clause
        )

        print(f"   📊 Approximate search results: k={len(results.keys())}, v={len(results.values())} found")
        print("   📊 Approximate search results:", results)

        # Convert to SimilarityResult objects
        similarity_results = self._convert_chroma_results(results)
        
        print(f"   📊 Approximate search results converted: {len(similarity_results)} items")
        
        # For approximate search, estimate candidates examined based on HNSW parameters
        # This is an approximation since ChromaDB doesn't expose this information
        collection_size = self.collection.count()
        ef_search = 100  # Default or configured value
        total_examined = min(ef_search, collection_size)
        
        return similarity_results, total_examined
    
    def _hybrid_search(self, query_embedding: np.ndarray, k: int,
                      filters: Optional[Dict[str, Any]]) -> Tuple[List[SimilarityResult], int]:
        """
        Perform hybrid search: approximate first, then exact if needed.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Tuple[List[SimilarityResult], int]: Results and total candidates examined
        """
        # Start with approximate search
        approx_results, approx_examined = self._approximate_search(query_embedding, k, filters)
        
        # Check if approximate results are sufficient
        if len(approx_results) >= k and all(r.confidence >= self.config.confidence_threshold for r in approx_results):
            return approx_results, approx_examined
        
        # If not sufficient, fall back to exact search
        print("Approximate search insufficient, falling back to exact search")
        exact_results, exact_examined = self._exact_search(query_embedding, k, filters)
        
        # Combine examination counts
        total_examined = approx_examined + exact_examined
        
        return exact_results, total_examined
    
    def _convert_chroma_results(self, chroma_results: chromadb.QueryResult) -> List[SimilarityResult]:
        """
        Convert ChromaDB results to SimilarityResult objects.

        Args:
            chroma_results: Raw results from ChromaDB query

        Returns:
            List[SimilarityResult]: Converted similarity results
        """
        similarity_results = []

        ids = chroma_results.get("ids", [[]])[0]
        print(f"📊 ids: {ids}")
        distances = chroma_results.get("distances", [[]])
        if distances is None or len(distances) == 0:
            distances = []
        else:
            distances = distances[0]
        metadatas = chroma_results.get("metadatas", [[]])
        if metadatas is None or len(metadatas) == 0:
            metadatas = []
        else:
            metadatas = metadatas[0]

        if not ids:
            print("⚠️ No results found in ChromaDB response.")
            return []

        print(f"📊 Converting {len(ids)} ChromaDB results to SimilarityResult objects")

        for rank, (result_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
            # Compute similarity score
            if self.config.distance_metric == "cosine":
                similarity_score = max(0.0, 1.0 - distance)
            elif self.config.distance_metric == "euclidean":
                similarity_score = 1.0 / (1.0 + distance)
            else:
                similarity_score = max(0.0, 1.0 - distance)  # default fallback

            # Compute confidence
            confidence = self._calculate_confidence(similarity_score, rank, len(ids))

            # Extract metadata safely
            image_id = str(metadata.get("image_id", result_id))
            object_class = str(metadata.get("object_class", "unknown"))

            # Build result
            similarity_results.append(SimilarityResult(
                image_id=image_id,
                similarity_score=similarity_score,
                confidence=confidence,
                object_class=object_class,
                metadata={
                    "embedding_id": result_id,
                    "model_version": metadata.get("model_version", "unknown"),
                    "extraction_timestamp": metadata.get("extraction_timestamp"),
                    "rank": rank + 1,
                    "raw_distance": distance,
                    "search_method": "chroma_db"
                }
            ))

        return similarity_results
    
    def _calculate_confidence(self, similarity_score: float, rank: int, total_results: int) -> float:
        """
        Calculate confidence score based on similarity and rank.

        Args:
            similarity_score: Similarity score (0-1)
            rank: Rank of the result (0-based)
            total_results: Total number of results

        Returns:
            float: Confidence score (0-1)
        """
        # Base confidence directly from similarity
        base_confidence = similarity_score

        # Penalize by rank position
        if total_results > 1:
            rank_penalty = 0.1 * (rank / (total_results - 1))
        else:
            rank_penalty = 0.0

        confidence = max(0.0, base_confidence - rank_penalty)

        # Slight bonus for very high similarity
        if similarity_score > 0.9:
            confidence = min(1.0, confidence + 0.1)

        return confidence
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from filters.
        
        Args:
            filters: Filter dictionary
            
        Returns:
            Dict[str, Any]: ChromaDB where clause
        """
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Handle multiple values
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                # Handle range queries
                where_clause[key] = value
            else:
                # Handle exact match
                where_clause[key] = value
        
        return where_clause
    
    def _validate_query_embedding(self, query_embedding: np.ndarray) -> None:
        """
        Validate query embedding format and properties.
        
        Args:
            query_embedding: Query embedding to validate
            
        Raises:
            ValueError: If embedding is invalid
        """
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("Query embedding must be a numpy array")
        
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be 1-dimensional")
        
        if len(query_embedding) == 0:
            raise ValueError("Query embedding cannot be empty")
        
        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains non-finite values")
        
        print(f"Query embedding validated: shape={query_embedding.shape}, dtype={query_embedding.dtype}")
        
        # Check if embedding is normalized (for cosine similarity)
        if self.config.distance_metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            if abs(norm - 1.0) > 0.1:  # Allow some tolerance
                print(f"Warning: Query embedding may not be normalized (norm={norm:.3f})")
    
    def _generate_cache_key(self, query_embedding: np.ndarray, k: int,
                           mode: SearchMode, filters: Optional[Dict[str, Any]]) -> str:
        """
        Generate cache key for query.
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            mode: Search mode
            filters: Optional filters
            
        Returns:
            str: Cache key
        """
        # Create a hash of the query parameters
        import hashlib
        
        # Combine parameters into a string
        key_parts = [
            np.array2string(query_embedding, precision=4),
            str(k),
            mode.value,
            str(sorted(filters.items()) if filters else "")
        ]
        key_string = "|".join(key_parts)
        
        # Create hash
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive search performance statistics.
        
        Returns:
            Dict[str, Any]: Search statistics
        """
        if not self.search_metrics:
            return {"message": "No search operations performed yet"}
        
        search_times = [m.search_time_ms for m in self.search_metrics]
        confidence_scores = []
        for m in self.search_metrics:
            confidence_scores.extend(m.confidence_scores)
        
        stats = {
            "total_searches": len(self.search_metrics),
            "avg_search_time_ms": np.mean(search_times),
            "min_search_time_ms": np.min(search_times),
            "max_search_time_ms": np.max(search_times),
            "std_search_time_ms": np.std(search_times),
            "total_results_returned": sum(m.results_returned for m in self.search_metrics),
            "avg_results_per_search": np.mean([m.results_returned for m in self.search_metrics]),
            "cache_hit_rate": sum(1 for m in self.search_metrics if m.cache_hit) / len(self.search_metrics),
            "search_mode_distribution": {}
        }
        
        # Search mode distribution
        for metric in self.search_metrics:
            mode = metric.search_mode_used.value
            stats["search_mode_distribution"][mode] = stats["search_mode_distribution"].get(mode, 0) + 1
        
        # Confidence statistics
        if confidence_scores:
            stats["confidence_statistics"] = {
                "avg_confidence": np.mean(confidence_scores),
                "min_confidence": np.min(confidence_scores),
                "max_confidence": np.max(confidence_scores),
                "std_confidence": np.std(confidence_scores)
            }
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the query result cache."""
        self.query_cache.clear()
        print("Query cache cleared")
    
    def clear_metrics(self) -> None:
        """Clear stored search metrics."""
        self.search_metrics.clear()
        print("Search metrics cleared")
