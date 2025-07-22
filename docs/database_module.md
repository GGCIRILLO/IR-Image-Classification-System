# Database Module Documentation

The `database` module provides functionality for storing, indexing, and searching vector embeddings of IR images. It uses ChromaDB as the underlying vector database and implements various search strategies and optimizations.

## Files and Classes

### `db_manager.py`

This file contains the `DatabaseManager` class which handles database operations, migrations, and configuration.

#### `DatabaseManager`

Manages the vector database operations, including initialization, migration, backup, and configuration.

**Key Methods:**

- `__init__(self, database_path: str = "./chroma_db", collection_name: str = "ir_embeddings")`: Initializes the database manager with path and collection name.
- `initialize_fresh_database(self, config: Optional[Dict[str, Any]] = None)`: Creates a new database with optional configuration.
- `validate_database(self)`: Validates the integrity of the database.
- `migrate_database(self, target_version: str)`: Migrates the database to a specified version.
- `backup_database(self, backup_path: str)`: Creates a backup of the database.
- `restore_database(self, backup_path: str)`: Restores the database from a backup.
- `load_configuration(self)`: Loads the database configuration.
- `update_configuration(self, new_config: Dict[str, Any])`: Updates the database configuration.
- `get_migration_history(self)`: Retrieves the history of database migrations.
- `cleanup_database(self)`: Performs cleanup operations on the database.

**Migration Methods:**

- `_perform_migration(self, from_version: str, to_version: str)`: Performs a migration between versions.
- `_migrate_1_0_to_1_1(self)`: Specific migration from version 1.0 to 1.1.
- `_migrate_1_1_to_2_0(self)`: Specific migration from version 1.1 to 2.0.
- `_log_migration(self, operation: str, message: str, details: Dict[str, Any])`: Logs migration operations.

### `index_manager.py`

This file contains classes for managing and optimizing vector indices for efficient similarity search.

#### `IndexConfig`

Configuration class for vector index settings.

#### `IndexMetrics`

Class for tracking and reporting index performance metrics.

#### `IndexManager`

Manages the creation, optimization, and maintenance of vector indices.

**Key Methods:**

- `__init__(self, database_path: str, collection_name: str = "ir_embeddings")`: Initializes the index manager.
- `initialize(self, config: Optional[IndexConfig] = None)`: Initializes the index with optional configuration.
- `create_hnsw_index(self, force_rebuild: bool = False)`: Creates a Hierarchical Navigable Small World (HNSW) index.
- `optimize_search_parameters(self, test_queries: List[np.ndarray], ground_truth: List[List[str]], k: int = 5)`: Optimizes search parameters for better performance.
- `validate_index_integrity(self)`: Validates the integrity of the index.
- `get_index_statistics(self)`: Gets statistics about the index.
- `get_collection_info(self)`: Gets information about the collection.
- `is_ready(self)`: Checks if the index is ready for use.
- `export_index_config(self, filepath: str)`: Exports the index configuration to a file.
- `_estimate_index_memory_usage(self, num_vectors: int, dimension: int)`: Estimates memory usage of the index.

### `similarity_searcher.py`

This file contains classes for performing similarity searches on vector embeddings.

#### `SearchMode`

Enum defining different search modes (exact, approximate, hybrid).

#### `SearchConfig`

Configuration class for similarity search settings.

#### `SearchMetrics`

Class for tracking and reporting search performance metrics.

#### `SimilaritySearcher`

Performs similarity searches on vector embeddings with various strategies.

**Key Methods:**

- `__init__(self, database_path: str, collection_name: str = "ir_embeddings")`: Initializes the similarity searcher.
- `initialize(self, config: Optional[SearchConfig] = None)`: Initializes the searcher with optional configuration.
- `search_similar(self, query_embedding: np.ndarray, k: Optional[int] = None, mode: Optional[SearchMode] = None, filters: Optional[Dict[str, Any]] = None)`: Searches for similar embeddings.
- `get_search_statistics(self)`: Gets statistics about search performance.
- `clear_cache(self)`: Clears the search cache.
- `clear_metrics(self)`: Clears the search metrics.

**Search Strategy Methods:**

- `_exact_search(self, query_embedding: np.ndarray, k: int, filters: Optional[Dict[str, Any]])`: Performs an exact search.
- `_approximate_search(self, query_embedding: np.ndarray, k: int, filters: Optional[Dict[str, Any]])`: Performs an approximate search.
- `_hybrid_search(self, query_embedding: np.ndarray, k: int, filters: Optional[Dict[str, Any]])`: Performs a hybrid search.
- `_validate_query_embedding(self, query_embedding: np.ndarray)`: Validates the query embedding.
- `_build_where_clause(self, filters: Dict[str, Any])`: Builds a where clause for filtering results.
- `_convert_chroma_results(self, chroma_results: chromadb.QueryResult)`: Converts ChromaDB results to the application's format.
- `_calculate_confidence(self, similarity_score: float, rank: int, total_results: int)`: Calculates confidence scores for results.
- `_generate_cache_key(self, query_embedding: np.ndarray, k: int, mode: SearchMode, filters: Optional[Dict[str, Any]])`: Generates a cache key for search results.
- `_extract_object_class_from_id(self, image_id: str)`: Extracts object class information from an image ID.

### `vector_store.py`

This file contains the `ChromaVectorStore` class which provides an interface to the ChromaDB vector database.

#### `ChromaVectorStore`

Provides an interface to the ChromaDB vector database for storing and retrieving embeddings.

**Key Methods:**

- `__init__(self, db_path: str = "./data/chroma_db", collection_name: str = "ir_embeddings")`: Initializes the vector store.
- `initialize_database(self, config: Dict[str, Any])`: Initializes the database with configuration.
- `store_embedding(self, embedding: Embedding)`: Stores a single embedding.
- `store_embeddings_batch(self, embeddings: List[Embedding])`: Stores multiple embeddings in a batch.
- `get_embedding(self, embedding_id: str)`: Retrieves an embedding by ID.
- `delete_embedding(self, embedding_id: str)`: Deletes an embedding by ID.
- `search_similar(self, query_embedding: np.ndarray, k: int = 5)`: Searches for similar embeddings.
- `create_index(self, index_type: str = "hnsw")`: Creates an index of the specified type.
- `get_database_stats(self)`: Gets statistics about the database.
- `close(self)`: Closes the database connection.

The database module is a critical component of the IR image classification system, providing efficient storage and retrieval of vector embeddings. It implements various optimization techniques to ensure fast similarity searches, even with large collections of embeddings.
