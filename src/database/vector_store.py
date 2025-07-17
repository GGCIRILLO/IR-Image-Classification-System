"""
Simplified Vector database implementation using ChromaDB for IR image embeddings.
"""

import os
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from datetime import datetime

from ..models.interfaces import IVectorStore
from ..models.data_models import Embedding, SimilarityResult

logger = logging.getLogger(__name__)


class ChromaVectorStore(IVectorStore):
    """
    ChromaDB implementation of vector store for IR image embeddings.
    
    This implementation provides local vector storage with HNSW indexing
    for efficient similarity search operations.
    """
    
    def __init__(self, db_path: str = "./data/chroma_db", collection_name: str = "ir_embeddings"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Name of the collection to use
        """
        self._db_path = db_path
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_dimension = None
        self.is_initialized = False
        
        # Ensure directory exists
        os.makedirs(db_path, exist_ok=True)
    
    def initialize_database(self, config: Dict[str, Any]) -> bool:
        """Initialize the ChromaDB database and collection."""
        try:
            # Update configuration if provided
            if config.get('db_path'):
                self._db_path = config['db_path']
                os.makedirs(self._db_path, exist_ok=True)
            
            if config.get('collection_name'):
                self._collection_name = config['collection_name']
            
            # Create ChromaDB client with new API
            self._client = chromadb.PersistentClient(path=self._db_path)
            
            # Create or get collection
            try:
                self._collection = self._client.get_collection(name=self._collection_name)
                logger.info(f"Retrieved existing collection: {self._collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self._collection = self._client.create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self._collection_name}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def store_embedding(self, embedding: Embedding) -> bool:
        """Store a single embedding vector in the database."""
        if not self.is_initialized or not self._collection:
            raise RuntimeError("Database not initialized")
        
        try:
            # Convert datetime to ISO string for ChromaDB
            timestamp_str = embedding.extraction_timestamp.isoformat()
            
            # Prepare metadata
            metadata = {
                'image_id': embedding.image_id,
                'model_version': embedding.model_version,
                'extraction_timestamp': timestamp_str,
                'dimension': len(embedding.vector)
            }
            
            # Store in ChromaDB
            self._collection.add(
                embeddings=[embedding.vector.tolist()],
                metadatas=[metadata],
                ids=[embedding.id]
            )
            
            logger.info(f"Stored embedding {embedding.id} for image {embedding.image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding {embedding.id}: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[SimilarityResult]:
        """Find the k most similar embeddings to the query."""
        if not self.is_initialized or not self._collection:
            raise RuntimeError("Database not initialized")
        
        try:
            # Query ChromaDB for similar embeddings
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=['metadatas', 'distances']
            )
            
            similarity_results = []
            
            # Process results safely
            if results.get('ids') and results['ids'][0]:
                ids = results['ids'][0]
                distances_list = results.get('distances')
                metadatas_list = results.get('metadatas')
                
                distances = distances_list[0] if distances_list and distances_list[0] else []
                metadatas = metadatas_list[0] if metadatas_list and metadatas_list[0] else []
                
                for i, (result_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
                    # Convert distance to similarity score
                    similarity_score = max(0.0, min(1.0, 1.0 - float(distance)))  # Clamp to [0,1]
                    confidence = similarity_score  # Simple confidence calculation
                    
                    # Safe metadata extraction
                    image_id = str(metadata.get('image_id', result_id)) if metadata else str(result_id)
                    object_class = str(metadata.get('object_class', 'unknown')) if metadata else 'unknown'
                    
                    similarity_result = SimilarityResult(
                        image_id=image_id,
                        similarity_score=similarity_score,
                        confidence=confidence,
                        object_class=object_class,
                        metadata={
                            'embedding_id': str(result_id),
                            'rank': i + 1,
                            'raw_distance': float(distance)
                        }
                    )
                    
                    similarity_results.append(similarity_result)
            
            return similarity_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_embedding(self, embedding_id: str) -> Optional[Embedding]:
        """Retrieve a specific embedding by its ID."""
        if not self.is_initialized or not self._collection:
            raise RuntimeError("Database not initialized")
        
        try:
            results = self._collection.get(
                ids=[embedding_id],
                include=['embeddings', 'metadatas']
            )
            
            if results.get('ids') and len(results.get('ids', [])) > 0 and results['ids'][0]:
                embeddings_list = results.get('embeddings')
                metadatas_list = results.get('metadatas')
                
                if (embeddings_list is not None and len(embeddings_list) > 0 and 
                    metadatas_list is not None and len(metadatas_list) > 0):
                    embedding_vector = np.array(embeddings_list[0])
                    metadata = metadatas_list[0]
                    
                    # Parse timestamp with type safety
                    timestamp_str = metadata.get('extraction_timestamp', '')
                    try:
                        extraction_timestamp = datetime.fromisoformat(str(timestamp_str))
                    except (ValueError, TypeError):
                        extraction_timestamp = datetime.now()
                    
                    return Embedding(
                        id=embedding_id,
                        vector=embedding_vector,
                        image_id=str(metadata.get('image_id', '')),
                        model_version=str(metadata.get('model_version', 'unknown')),
                        extraction_timestamp=extraction_timestamp
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve embedding {embedding_id}: {e}")
            return None
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding from the database."""
        if not self.is_initialized or not self._collection:
            raise RuntimeError("Database not initialized")
        
        try:
            self._collection.delete(ids=[embedding_id])
            logger.info(f"Deleted embedding {embedding_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embedding {embedding_id}: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.is_initialized or not self._collection:
            return {
                'status': 'not_initialized',
                'collection_count': 0,
                'embedding_dimension': None
            }
        
        try:
            collection_count = self._collection.count()
            
            return {
                'status': 'initialized',
                'collection_name': self._collection_name,
                'collection_count': collection_count,
                'embedding_dimension': self._embedding_dimension,
                'db_path': self._db_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'collection_count': 0
            }
    
    def create_index(self, index_type: str = "hnsw") -> bool:
        """Create or rebuild the similarity search index."""
        if not self.is_initialized or not self._collection:
            raise RuntimeError("Database not initialized")
        
        try:
            # ChromaDB automatically maintains HNSW index
            # No explicit index creation needed
            logger.info(f"Index type {index_type} is automatically maintained by ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def store_embeddings_batch(self, embeddings: List[Embedding]) -> bool:
        """Store multiple embeddings in a single batch operation."""
        if not self.is_initialized or not self._collection:
            raise RuntimeError("Database not initialized")
        
        if not embeddings:
            return True
        
        try:
            # Prepare batch data
            embedding_vectors = []
            metadatas = []
            ids = []
            
            for embedding in embeddings:
                # Convert datetime to ISO string
                timestamp_str = embedding.extraction_timestamp.isoformat()
                
                metadata = {
                    'image_id': embedding.image_id,
                    'model_version': embedding.model_version,
                    'extraction_timestamp': timestamp_str,
                    'dimension': len(embedding.vector)
                }
                
                embedding_vectors.append(embedding.vector.tolist())
                metadatas.append(metadata)
                ids.append(embedding.id)
            
            # Batch insert
            self._collection.add(
                embeddings=embedding_vectors,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored batch of {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding batch: {e}")
            return False
    
    def close(self) -> None:
        """Close database connections."""
        if self._client:
            try:
                # ChromaDB client doesn't need explicit closing
                self.is_initialized = False
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}")
