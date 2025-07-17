#!/usr/bin/env python3
"""
Database initialization and migration scripts for IR image classification system.

This script provides utilities to initialize a fresh ChromaDB database,
perform migrations, and manage database configuration for military deployment.
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.database.vector_store import ChromaVectorStore
from src.database.db_manager import DatabaseManager
from src.database.index_manager import IndexManager, IndexConfig
from src.database.similarity_searcher import SimilaritySearcher, SearchConfig


def create_default_config() -> Dict[str, Any]:
    """
    Create default database configuration for military deployment.
    
    Returns:
        Dict[str, Any]: Default configuration parameters
    """
    return {
        "database": {
            "type": "chromadb",
            "path": "./data/vector_db",
            "collection_name": "ir_embeddings",
            "embedding_dimension": 512,
            "distance_metric": "cosine"
        },
        "index": {
            "type": "hnsw",
            "hnsw_m": 16,
            "hnsw_ef_construction": 200,
            "hnsw_ef_search": 100,
            "enable_exact_search": True,
            "cache_size": 1000
        },
        "search": {
            "default_k": 5,
            "confidence_threshold": 0.7,
            "max_search_time_ms": 2000.0,
            "enable_reranking": True,
            "cache_queries": True
        },
        "security": {
            "disable_telemetry": True,
            "local_only": True,
            "encryption_enabled": False  # Can be enabled for classified data
        },
        "performance": {
            "batch_size": 100,
            "parallel_workers": 4,
            "memory_limit_gb": 8
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "deployment_type": "military",
            "description": "IR Image Classification Vector Database"
        }
    }


def initialize_database(config_path: Optional[str] = None, 
                       database_path: Optional[str] = None,
                       force: bool = False) -> bool:
    """
    Initialize a fresh database with configuration.
    
    Args:
        config_path: Path to configuration file
        database_path: Override database path
        force: Force initialization even if database exists
        
    Returns:
        bool: True if initialization was successful
    """
    print("=== IR Image Classification Database Initialization ===")
    
    # Load or create configuration
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print("Using default configuration")
        config = create_default_config()
    
    # Override database path if provided
    if database_path:
        config["database"]["path"] = database_path
    
    db_path = config["database"]["path"]
    collection_name = config["database"]["collection_name"]
    
    # Check if database already exists
    if os.path.exists(db_path) and not force:
        response = input(f"Database already exists at {db_path}. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Initialization cancelled")
            return False
    
    # Create database directory
    os.makedirs(db_path, exist_ok=True)
    print(f"Database directory: {db_path}")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(db_path, collection_name)
        
        # Initialize fresh database
        print("Initializing ChromaDB...")
        success = db_manager.initialize_fresh_database(config["database"])
        
        if not success:
            print("❌ Failed to initialize database")
            return False
        
        print("✅ Database initialized successfully")
        
        # Test database connection
        print("Testing database connection...")
        vector_store = ChromaVectorStore(db_path, collection_name)
        if vector_store.initialize_database(config["database"]):
            print("✅ Database connection test passed")
        else:
            print("❌ Database connection test failed")
            return False
        
        # Save complete configuration
        config_file = os.path.join(db_path, "database_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration saved to {config_file}")
        
        # Create initialization log
        log_file = os.path.join(db_path, "initialization.log")
        with open(log_file, 'w') as f:
            f.write(f"Database initialized at {datetime.now().isoformat()}\n")
            f.write(f"Configuration: {json.dumps(config, indent=2)}\n")
        
        print(f"✅ Initialization log saved to {log_file}")
        print("\n=== Database initialization completed successfully ===")
        
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        return False


def setup_indexes(database_path: str, config_path: Optional[str] = None) -> bool:
    """
    Set up optimized indexes for the database.
    
    Args:
        database_path: Path to the database
        config_path: Path to configuration file
        
    Returns:
        bool: True if index setup was successful
    """
    print("\n=== Setting up database indexes ===")
    
    # Load configuration
    config_file = config_path or os.path.join(database_path, "database_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        print("No configuration found, using defaults")
        config = create_default_config()
    
    try:
        # Create index configuration
        index_config = IndexConfig(
            index_type=config["index"]["type"],
            distance_metric=config["database"]["distance_metric"],
            hnsw_m=config["index"]["hnsw_m"],
            hnsw_ef_construction=config["index"]["hnsw_ef_construction"],
            hnsw_ef_search=config["index"]["hnsw_ef_search"],
            enable_exact_search=config["index"]["enable_exact_search"],
            cache_size=config["index"]["cache_size"]
        )
        
        # Initialize index manager
        collection_name = config["database"]["collection_name"]
        index_manager = IndexManager(database_path, collection_name)
        
        if not index_manager.initialize(index_config):
            print("❌ Failed to initialize index manager")
            return False
        
        # Create HNSW index
        print("Creating HNSW index...")
        if index_manager.create_hnsw_index():
            print("✅ HNSW index created successfully")
        else:
            print("❌ Failed to create HNSW index")
            return False
        
        # Get index statistics
        stats = index_manager.get_index_statistics()
        print(f"Index statistics: {json.dumps(stats, indent=2)}")
        
        # Validate index integrity
        print("Validating index integrity...")
        validation = index_manager.validate_index_integrity()
        
        if validation["is_valid"]:
            print("✅ Index validation passed")
            for check in validation["checks_performed"]:
                print(f"  ✓ {check}")
        else:
            print("❌ Index validation failed")
            for error in validation["errors"]:
                print(f"  ✗ {error}")
            for warning in validation["warnings"]:
                print(f"  ⚠ {warning}")
        
        # Export index configuration
        index_config_file = os.path.join(database_path, "index_config.json")
        index_manager.export_index_config(index_config_file)
        
        print("✅ Index setup completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Index setup failed: {str(e)}")
        return False


def test_search_functionality(database_path: str, config_path: Optional[str] = None) -> bool:
    """
    Test search functionality with sample queries.
    
    Args:
        database_path: Path to the database
        config_path: Path to configuration file
        
    Returns:
        bool: True if search tests passed
    """
    print("\n=== Testing search functionality ===")
    
    # Load configuration
    config_file = config_path or os.path.join(database_path, "database_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        print("No configuration found, using defaults")
        config = create_default_config()
    
    try:
        # Initialize similarity searcher
        collection_name = config["database"]["collection_name"]
        searcher = SimilaritySearcher(database_path, collection_name)
        
        # Create search configuration
        search_config = SearchConfig(
            k=config["search"]["default_k"],
            confidence_threshold=config["search"]["confidence_threshold"],
            max_search_time_ms=config["search"]["max_search_time_ms"],
            enable_reranking=config["search"]["enable_reranking"],
            cache_queries=config["search"]["cache_queries"]
        )
        
        if not searcher.initialize(search_config):
            print("❌ Failed to initialize similarity searcher")
            return False
        
        # Check if database has embeddings
        if searcher.collection and searcher.collection.count() == 0:
            print("⚠ Database is empty - skipping search tests")
            print("  Add embeddings to the database first to test search functionality")
            return True
        
        # Create a dummy test query (random vector)
        import numpy as np
        embedding_dim = config["database"]["embedding_dimension"]
        test_query = np.random.random(embedding_dim).astype(np.float32)
        
        # Normalize for cosine similarity
        if config["database"]["distance_metric"] == "cosine":
            test_query = test_query / np.linalg.norm(test_query)
        
        print(f"Testing with random {embedding_dim}-dimensional query vector")
        
        # Test approximate search
        print("Testing approximate search...")
        results, metrics = searcher.search_similar(test_query, k=5)
        
        print(f"✅ Approximate search completed in {metrics.search_time_ms:.2f}ms")
        print(f"  Results returned: {metrics.results_returned}")
        print(f"  Candidates examined: {metrics.total_candidates_examined}")
        
        if results:
            print("  Sample results:")
            for i, result in enumerate(results[:3]):
                print(f"    {i+1}. Similarity: {result.similarity_score:.3f}, "
                      f"Confidence: {result.confidence:.3f}")
        
        # Get search statistics
        stats = searcher.get_search_statistics()
        print(f"Search statistics: {json.dumps(stats, indent=2)}")
        
        print("✅ Search functionality tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Search functionality test failed: {str(e)}")
        return False


def backup_database(database_path: str, backup_path: str) -> bool:
    """
    Create a backup of the database.
    
    Args:
        database_path: Path to the database to backup
        backup_path: Path for the backup
        
    Returns:
        bool: True if backup was successful
    """
    print(f"\n=== Creating database backup ===")
    print(f"Source: {database_path}")
    print(f"Destination: {backup_path}")
    
    try:
        if os.path.exists(backup_path):
            response = input(f"Backup already exists at {backup_path}. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Backup cancelled")
                return False
            shutil.rmtree(backup_path)
        
        # Create backup directory and copy database
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copytree(database_path, backup_path)
        
        # Create backup metadata
        backup_info = {
            "backup_created_at": datetime.now().isoformat(),
            "source_path": database_path,
            "backup_path": backup_path,
            "backup_size_mb": sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(backup_path)
                for filename in filenames
            ) / (1024 * 1024)
        }
        
        backup_info_file = os.path.join(backup_path, "backup_info.json")
        with open(backup_info_file, 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        print(f"✅ Database backup created successfully")
        print(f"  Backup size: {backup_info['backup_size_mb']:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Database backup failed: {str(e)}")
        return False


def restore_database(backup_path: str, database_path: str) -> bool:
    """
    Restore database from backup.
    
    Args:
        backup_path: Path to the backup
        database_path: Path to restore the database to
        
    Returns:
        bool: True if restore was successful
    """
    print(f"\n=== Restoring database from backup ===")
    print(f"Source: {backup_path}")
    print(f"Destination: {database_path}")
    
    try:
        if not os.path.exists(backup_path):
            print(f"❌ Backup not found at {backup_path}")
            return False
        
        if os.path.exists(database_path):
            response = input(f"Database exists at {database_path}. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Restore cancelled")
                return False
            shutil.rmtree(database_path)
        
        # Restore database
        shutil.copytree(backup_path, database_path)
        
        # Remove backup info file from restored database
        backup_info_file = os.path.join(database_path, "backup_info.json")
        if os.path.exists(backup_info_file):
            os.remove(backup_info_file)
        
        print("✅ Database restored successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Database restore failed: {str(e)}")
        return False


def main():
    """Main function for database management CLI."""
    parser = argparse.ArgumentParser(
        description="IR Image Classification Database Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize a new database
  python init_database.py init --database-path ./data/vector_db

  # Initialize with custom configuration
  python init_database.py init --config config.json

  # Set up indexes for existing database
  python init_database.py setup-indexes --database-path ./data/vector_db

  # Test search functionality
  python init_database.py test-search --database-path ./data/vector_db

  # Create backup
  python init_database.py backup --database-path ./data/vector_db --backup-path ./backups/db_backup

  # Restore from backup
  python init_database.py restore --backup-path ./backups/db_backup --database-path ./data/vector_db
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize new database')
    init_parser.add_argument('--database-path', default='./data/vector_db',
                           help='Path for the database directory')
    init_parser.add_argument('--config', help='Path to configuration file')
    init_parser.add_argument('--force', action='store_true',
                           help='Force initialization even if database exists')
    
    # Setup indexes command
    index_parser = subparsers.add_parser('setup-indexes', help='Set up database indexes')
    index_parser.add_argument('--database-path', required=True,
                            help='Path to the database directory')
    index_parser.add_argument('--config', help='Path to configuration file')
    
    # Test search command
    test_parser = subparsers.add_parser('test-search', help='Test search functionality')
    test_parser.add_argument('--database-path', required=True,
                           help='Path to the database directory')
    test_parser.add_argument('--config', help='Path to configuration file')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup')
    backup_parser.add_argument('--database-path', required=True,
                             help='Path to the database directory')
    backup_parser.add_argument('--backup-path', required=True,
                             help='Path for the backup')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore database from backup')
    restore_parser.add_argument('--backup-path', required=True,
                              help='Path to the backup')
    restore_parser.add_argument('--database-path', required=True,
                              help='Path to restore the database to')
    
    # Generate config command
    config_parser = subparsers.add_parser('generate-config', help='Generate default configuration file')
    config_parser.add_argument('--output', default='database_config.json',
                             help='Output configuration file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'init':
        success = initialize_database(args.config, args.database_path, args.force)
        if success:
            print(f"\n🎉 Database ready at {args.database_path}")
            print("Next steps:")
            print(f"  1. Add embeddings to the database")
            print(f"  2. Set up indexes: python {sys.argv[0]} setup-indexes --database-path {args.database_path}")
            print(f"  3. Test search: python {sys.argv[0]} test-search --database-path {args.database_path}")
        sys.exit(0 if success else 1)
    
    elif args.command == 'setup-indexes':
        success = setup_indexes(args.database_path, args.config)
        sys.exit(0 if success else 1)
    
    elif args.command == 'test-search':
        success = test_search_functionality(args.database_path, args.config)
        sys.exit(0 if success else 1)
    
    elif args.command == 'backup':
        success = backup_database(args.database_path, args.backup_path)
        sys.exit(0 if success else 1)
    
    elif args.command == 'restore':
        success = restore_database(args.backup_path, args.database_path)
        sys.exit(0 if success else 1)
    
    elif args.command == 'generate-config':
        config = create_default_config()
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Default configuration saved to {args.output}")
        sys.exit(0)


if __name__ == "__main__":
    main()
