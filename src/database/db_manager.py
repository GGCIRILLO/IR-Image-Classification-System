"""
Database management utilities for IR image classification system.

This module provides utilities for database initialization, migration,
and maintenance operations for the ChromaDB vector store.

Enhanced with multi-database support for different model types (ResNet18, ResNet50).
"""

import os
import json
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pathlib import Path

from .vector_store import ChromaVectorStore


class ModelType(Enum):
    """Supported model types for database configuration."""
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"


class DatabaseManager:
    """
    Manages database initialization, migration, and maintenance operations.
    
    Enhanced with multi-database support for different model types.
    """
    
    def __init__(self, database_path: Optional[str] = None, 
                 collection_name: Optional[str] = None,
                 model_type: Optional[ModelType] = None):
        """
        Initialize database manager with flexible configuration options.
        
        Args:
            database_path: Path to the database directory (optional if model_type provided)
            collection_name: Name of the collection (optional if model_type provided)
            model_type: Model type for automatic database configuration
            
        Note:
            If model_type is provided, it takes precedence over database_path and collection_name.
            If neither model_type nor database_path is provided, defaults to ResNet50 configuration.
        """
        self.model_type = model_type
        self.multi_db_config_file = Path("config/multi_database_config.json")
        
        # Determine database configuration
        if model_type:
            # Use model type configuration
            db_config = self._get_model_database_config(model_type)
            self.database_path = db_config["database_path"]
            self.collection_name = db_config["collection_name"]
        elif database_path and collection_name:
            # Use provided parameters (legacy mode)
            self.database_path = database_path
            self.collection_name = collection_name
            self.model_type = ModelType.RESNET50  # Default assumption
        else:
            # Default to ResNet50 configuration
            self.model_type = ModelType.RESNET50
            db_config = self._get_model_database_config(ModelType.RESNET50)
            self.database_path = db_config["database_path"]
            self.collection_name = db_config["collection_name"]
        
        self.config_file = os.path.join(self.database_path, "db_config.json")
        self.migration_log = os.path.join(self.database_path, "migrations.log")
        
        # Initialize multi-database configuration if it doesn't exist
        self._ensure_multi_db_config()
    
    def _get_default_multi_db_config(self) -> Dict[str, Any]:
        """Get default multi-database configuration."""
        return {
            "version": "1.0.0",
            "description": "Multi-database configuration for different model types",
            "databases": {
                ModelType.RESNET50.value: {
                    "model_type": ModelType.RESNET50.value,
                    "database_path": "./data/vector_db",
                    "collection_name": "ir_embeddings",
                    "embedding_dimension": 512,
                    "distance_metric": "cosine",
                    "description": "ResNet50 model database (existing/legacy)"
                },
                ModelType.RESNET18.value: {
                    "model_type": ModelType.RESNET18.value,
                    "database_path": "./data/vector_db_resnet18",
                    "collection_name": "ir_embeddings_resnet18",
                    "embedding_dimension": 512,
                    "distance_metric": "cosine",
                    "description": "ResNet18 model database (new)"
                }
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
            }
        }
    
    def _ensure_multi_db_config(self) -> None:
        """Ensure multi-database configuration file exists."""
        if not self.multi_db_config_file.exists():
            try:
                # Create config directory if it doesn't exist
                self.multi_db_config_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Create default configuration
                default_config = self._get_default_multi_db_config()
                
                with open(self.multi_db_config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                print(f"Created multi-database configuration at {self.multi_db_config_file}")
                
            except Exception as e:
                print(f"Warning: Could not create multi-database config: {e}")
    
    def _get_model_database_config(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get database configuration for specified model type.
        
        Args:
            model_type: Model type to get configuration for
            
        Returns:
            Dict[str, Any]: Database configuration
        """
        try:
            if self.multi_db_config_file.exists():
                with open(self.multi_db_config_file, 'r') as f:
                    config_data = json.load(f)
                
                databases = config_data.get("databases", {})
                if model_type.value in databases:
                    return databases[model_type.value]
            
            # Fallback to defaults
            defaults = self._get_default_multi_db_config()
            return defaults["databases"][model_type.value]
            
        except Exception as e:
            print(f"Error loading model config for {model_type.value}: {e}")
            # Hard-coded fallback
            if model_type == ModelType.RESNET18:
                return {
                    "database_path": "./data/vector_db_resnet18",
                    "collection_name": "ir_embeddings_resnet18",
                    "embedding_dimension": 512,
                    "distance_metric": "cosine"
                }
            else:
                return {
                    "database_path": "./data/vector_db",
                    "collection_name": "ir_embeddings",
                    "embedding_dimension": 512,
                    "distance_metric": "cosine"
                }
    
    def get_available_model_types(self) -> List[ModelType]:
        """
        Get list of available model types.
        
        Returns:
            List[ModelType]: Available model types
        """
        try:
            if self.multi_db_config_file.exists():
                with open(self.multi_db_config_file, 'r') as f:
                    config_data = json.load(f)
                
                databases = config_data.get("databases", {})
                return [ModelType(model_type) for model_type in databases.keys() 
                       if model_type in [mt.value for mt in ModelType]]
            
            return [ModelType.RESNET50, ModelType.RESNET18]
            
        except Exception as e:
            print(f"Error getting available model types: {e}")
            return [ModelType.RESNET50, ModelType.RESNET18]
    
    def switch_database(self, model_type: ModelType) -> bool:
        """
        Switch to a different database for the specified model type.
        
        Args:
            model_type: Model type to switch to
            
        Returns:
            bool: True if switch was successful
        """
        try:
            # Get new database configuration
            db_config = self._get_model_database_config(model_type)
            
            # Update configuration
            self.model_type = model_type
            self.database_path = db_config["database_path"]
            self.collection_name = db_config["collection_name"]
            self.config_file = os.path.join(self.database_path, "db_config.json")
            self.migration_log = os.path.join(self.database_path, "migrations.log")
            
            print(f"Switched to {model_type.value} database: {self.database_path}")
            return True
            
        except Exception as e:
            print(f"Failed to switch to {model_type.value} database: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the current database configuration.
        
        Returns:
            Dict[str, Any]: Database information
        """
        return {
            "model_type": self.model_type.value if self.model_type else "unknown",
            "database_path": self.database_path,
            "collection_name": self.collection_name,
            "config_file": self.config_file,
            "migration_log": self.migration_log,
            "database_exists": os.path.exists(self.database_path)
        }
    
    def initialize_fresh_database(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize a fresh database with default configuration.
        
        Args:
            config: Optional configuration parameters
            
        Returns:
            bool: True if initialization was successful
        """
        # Get model-specific configuration
        model_config = self._get_model_database_config(self.model_type) if self.model_type else {}
        
        # Default configuration
        default_config = {
            "embedding_dimension": model_config.get("embedding_dimension", 512),
            "distance_metric": model_config.get("distance_metric", "cosine"),
            "index_type": "hnsw",
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "model_type": self.model_type.value if self.model_type else "resnet50"
        }
        
        if config:
            default_config.update(config)
        
        try:
            # Create database directory
            os.makedirs(self.database_path, exist_ok=True)
            
            # Save configuration
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            # Initialize vector store
            vector_store = ChromaVectorStore(self.database_path, self.collection_name)
            success = vector_store.initialize_database(default_config)
            
            if success:
                self._log_migration("INIT", "Database initialized successfully", default_config)
                print(f"Database initialized at {self.database_path}")
                return True
            else:
                print("Failed to initialize vector store")
                return False
                
        except Exception as e:
            print(f"Failed to initialize database: {str(e)}")
            return False
    
    def initialize_all_databases(self) -> Dict[str, bool]:
        """
        Initialize databases for all available model types.
        
        Returns:
            Dict[str, bool]: Results for each model type
        """
        results = {}
        available_types = self.get_available_model_types()
        
        for model_type in available_types:
            print(f"\nInitializing database for {model_type.value}...")
            
            # Create a new manager instance for this model type
            manager = DatabaseManager(model_type=model_type)
            
            # Initialize the database
            success = manager.initialize_fresh_database()
            results[model_type.value] = success
            
            if success:
                print(f"✅ {model_type.value} database initialized successfully")
            else:
                print(f"❌ {model_type.value} database initialization failed")
        
        return results
    
    def get_all_database_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all configured databases.
        
        Returns:
            Dict[str, Dict[str, Any]]: Status for each model type
        """
        status = {}
        available_types = self.get_available_model_types()
        
        for model_type in available_types:
            try:
                db_config = self._get_model_database_config(model_type)
                db_path = db_config["database_path"]
                
                status[model_type.value] = {
                    "database_path": db_path,
                    "collection_name": db_config["collection_name"],
                    "exists": os.path.exists(db_path),
                    "config_exists": os.path.exists(os.path.join(db_path, "db_config.json")),
                    "embedding_dimension": db_config.get("embedding_dimension", 512)
                }
                
                # Try to get database stats if it exists
                if status[model_type.value]["exists"]:
                    try:
                        manager = DatabaseManager(model_type=model_type)
                        validation = manager.validate_database()
                        status[model_type.value]["valid"] = validation["valid"]
                        status[model_type.value]["stats"] = validation.get("stats", {})
                    except Exception as e:
                        status[model_type.value]["validation_error"] = str(e)
                
            except Exception as e:
                status[model_type.value] = {
                    "error": f"Failed to get status: {e}"
                }
        
        return status
    
    def backup_all_databases(self, backup_base_path: str) -> Dict[str, bool]:
        """
        Backup all configured databases.
        
        Args:
            backup_base_path: Base path for backups
            
        Returns:
            Dict[str, bool]: Backup results for each model type
        """
        results = {}
        available_types = self.get_available_model_types()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_type in available_types:
            try:
                manager = DatabaseManager(model_type=model_type)
                backup_path = os.path.join(backup_base_path, f"{model_type.value}_{timestamp}")
                
                success = manager.backup_database(backup_path)
                results[model_type.value] = success
                
            except Exception as e:
                print(f"Failed to backup {model_type.value} database: {e}")
                results[model_type.value] = False
        
        return results
    
    def print_multi_database_status(self) -> None:
        """Print status of all configured databases."""
        print("="*80)
        print("MULTI-DATABASE STATUS")
        print("="*80)
        
        status = self.get_all_database_status()
        
        for model_type, info in status.items():
            print(f"\n{model_type.upper()}:")
            
            if "error" in info:
                print(f"  ❌ Error: {info['error']}")
                continue
            
            print(f"  Database path: {info['database_path']}")
            print(f"  Collection: {info['collection_name']}")
            print(f"  Embedding dimension: {info['embedding_dimension']}")
            
            exists_icon = "✅" if info['exists'] else "❌"
            print(f"  Database exists: {exists_icon}")
            
            config_icon = "✅" if info.get('config_exists', False) else "❌"
            print(f"  Config exists: {config_icon}")
            
            if info.get('valid') is not None:
                valid_icon = "✅" if info['valid'] else "❌"
                print(f"  Database valid: {valid_icon}")
            
            if 'stats' in info and 'database' in info['stats']:
                db_stats = info['stats']['database']
                print(f"  Embeddings: {db_stats.get('embedding_count', 'unknown')}")
    
    def load_configuration(self) -> Optional[Dict[str, Any]]:
        """
        Load database configuration from file.
        
        Returns:
            Optional[Dict[str, Any]]: Configuration if found, None otherwise
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Failed to load configuration: {str(e)}")
            return None
    
    def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """
        Update database configuration.
        
        Args:
            new_config: New configuration parameters
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Load existing configuration
            current_config = self.load_configuration() or {}
            
            # Update with new values
            current_config.update(new_config)
            current_config["updated_at"] = datetime.now().isoformat()
            
            # Save updated configuration
            with open(self.config_file, 'w') as f:
                json.dump(current_config, f, indent=2)
            
            self._log_migration("CONFIG_UPDATE", "Configuration updated", new_config)
            return True
            
        except Exception as e:
            print(f"Failed to update configuration: {str(e)}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the entire database.
        
        Args:
            backup_path: Path to store the backup
            
        Returns:
            bool: True if backup was successful
        """
        try:
            if not os.path.exists(self.database_path):
                print("Database does not exist, nothing to backup")
                return False
            
            # Create backup directory
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Copy entire database directory
            shutil.copytree(self.database_path, backup_path, dirs_exist_ok=True)
            
            # Add backup metadata
            backup_metadata = {
                "backup_timestamp": datetime.now().isoformat(),
                "source_path": self.database_path,
                "backup_path": backup_path,
                "collection_name": self.collection_name
            }
            
            metadata_file = os.path.join(backup_path, "backup_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            self._log_migration("BACKUP", f"Database backed up to {backup_path}", backup_metadata)
            print(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            print(f"Failed to backup database: {str(e)}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to the backup directory
            
        Returns:
            bool: True if restore was successful
        """
        try:
            if not os.path.exists(backup_path):
                print(f"Backup path does not exist: {backup_path}")
                return False
            
            # Verify backup metadata
            metadata_file = os.path.join(backup_path, "backup_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    backup_metadata = json.load(f)
                print(f"Restoring backup from {backup_metadata.get('backup_timestamp', 'unknown time')}")
            
            # Remove existing database if it exists
            if os.path.exists(self.database_path):
                shutil.rmtree(self.database_path)
            
            # Copy backup to database location
            shutil.copytree(backup_path, self.database_path)
            
            # Remove backup metadata from restored database
            restored_metadata_file = os.path.join(self.database_path, "backup_metadata.json")
            if os.path.exists(restored_metadata_file):
                os.remove(restored_metadata_file)
            
            self._log_migration("RESTORE", f"Database restored from {backup_path}", {})
            print(f"Database restored from {backup_path}")
            return True
            
        except Exception as e:
            print(f"Failed to restore database: {str(e)}")
            return False
    
    def migrate_database(self, target_version: str) -> bool:
        """
        Migrate database to a new version.
        
        Args:
            target_version: Target version to migrate to
            
        Returns:
            bool: True if migration was successful
        """
        try:
            current_config = self.load_configuration()
            if not current_config:
                print("No configuration found, cannot migrate")
                return False
            
            current_version = current_config.get("version", "1.0.0")
            
            if current_version == target_version:
                print(f"Database is already at version {target_version}")
                return True
            
            print(f"Migrating database from version {current_version} to {target_version}")
            
            # Perform version-specific migrations
            if self._perform_migration(current_version, target_version):
                # Update version in configuration
                current_config["version"] = target_version
                current_config["migrated_at"] = datetime.now().isoformat()
                
                with open(self.config_file, 'w') as f:
                    json.dump(current_config, f, indent=2)
                
                self._log_migration("MIGRATE", f"Migrated from {current_version} to {target_version}", {
                    "from_version": current_version,
                    "to_version": target_version
                })
                
                print(f"Migration to version {target_version} completed successfully")
                return True
            else:
                print(f"Migration to version {target_version} failed")
                return False
                
        except Exception as e:
            print(f"Failed to migrate database: {str(e)}")
            return False
    
    def _perform_migration(self, from_version: str, to_version: str) -> bool:
        """
        Perform version-specific migration logic.
        
        Args:
            from_version: Current version
            to_version: Target version
            
        Returns:
            bool: True if migration was successful
        """
        # Define migration paths
        migration_paths = {
            ("1.0.0", "1.1.0"): self._migrate_1_0_to_1_1,
            ("1.1.0", "2.0.0"): self._migrate_1_1_to_2_0,
        }
        
        migration_key = (from_version, to_version)
        if migration_key in migration_paths:
            return migration_paths[migration_key]()
        else:
            print(f"No migration path defined from {from_version} to {to_version}")
            return False
    
    def _migrate_1_0_to_1_1(self) -> bool:
        """
        Migrate from version 1.0.0 to 1.1.0.
        
        Returns:
            bool: True if migration was successful
        """
        try:
            # Example migration: Add new metadata fields
            print("Performing migration 1.0.0 -> 1.1.0")
            
            # Initialize vector store to access data
            vector_store = ChromaVectorStore(self.database_path, self.collection_name)
            config = self.load_configuration()
            
            if config is None:
                print("No configuration found, cannot initialize vector store for migration")
                return False

            if not vector_store.initialize_database(config):
                return False
            
            # Migration logic would go here
            # For example: update metadata format, reindex, etc.
            
            print("Migration 1.0.0 -> 1.1.0 completed")
            return True
            
        except Exception as e:
            print(f"Migration 1.0.0 -> 1.1.0 failed: {str(e)}")
            return False
    
    def _migrate_1_1_to_2_0(self) -> bool:
        """
        Migrate from version 1.1.0 to 2.0.0.
        
        Returns:
            bool: True if migration was successful
        """
        try:
            print("Performing migration 1.1.0 -> 2.0.0")
            
            # Major version migration logic would go here
            # For example: schema changes, data format updates, etc.
            
            print("Migration 1.1.0 -> 2.0.0 completed")
            return True
            
        except Exception as e:
            print(f"Migration 1.1.0 -> 2.0.0 failed: {str(e)}")
            return False
    
    def _log_migration(self, operation: str, message: str, details: Dict[str, Any]) -> None:
        """
        Log migration operations.
        
        Args:
            operation: Type of operation (INIT, MIGRATE, BACKUP, etc.)
            message: Description of the operation
            details: Additional details about the operation
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "message": message,
                "details": details
            }
            
            # Append to migration log
            with open(self.migration_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            print(f"Failed to log migration: {str(e)}")
    
    def get_migration_history(self) -> list:
        """
        Get the history of migration operations.
        
        Returns:
            list: List of migration log entries
        """
        try:
            if not os.path.exists(self.migration_log):
                return []
            
            history = []
            with open(self.migration_log, 'r') as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line.strip()))
            
            return history
            
        except Exception as e:
            print(f"Failed to get migration history: {str(e)}")
            return []
    
    def validate_database(self) -> Dict[str, Any]:
        """
        Validate database integrity and configuration.
        
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Check if database directory exists
            if not os.path.exists(self.database_path):
                validation_results["valid"] = False
                validation_results["errors"].append("Database directory does not exist")
                return validation_results
            
            # Check configuration file
            config = self.load_configuration()
            if not config:
                validation_results["warnings"].append("No configuration file found")
            else:
                validation_results["stats"]["config"] = config
            
            # Initialize and validate vector store
            vector_store = ChromaVectorStore(self.database_path, self.collection_name)
            if config is not None and isinstance(config, dict):
                if vector_store.initialize_database(config):
                    stats = vector_store.get_database_stats()
                    validation_results["stats"]["database"] = stats
                    
                    if not stats.get("initialized", False):
                        validation_results["valid"] = False
                        validation_results["errors"].append("Vector store not properly initialized")
                else:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Failed to initialize vector store")
            else:
                validation_results["valid"] = False
                validation_results["errors"].append("Configuration is None or invalid, cannot initialize vector store")
            
            return validation_results
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            return validation_results
    
    def cleanup_database(self) -> bool:
        """
        Perform database cleanup operations.
        
        Returns:
            bool: True if cleanup was successful
        """
        try:
            print("Starting database cleanup...")
            
            # Initialize vector store
            config = self.load_configuration()
            if not config:
                print("No configuration found for cleanup")
                return False
            
            vector_store = ChromaVectorStore(self.database_path, self.collection_name)
            if not vector_store.initialize_database(config):
                print("Failed to initialize vector store for cleanup")
                return False
            
            # Get current stats
            stats_before = vector_store.get_database_stats()
            print(f"Database stats before cleanup: {stats_before['embedding_count']} embeddings")
            
            # Cleanup operations would go here
            # For example: remove duplicates, optimize indexes, etc.
            
            # Log cleanup operation
            self._log_migration("CLEANUP", "Database cleanup completed", {
                "embeddings_before": stats_before.get("embedding_count", 0)
            })
            
            print("Database cleanup completed")
            return True
            
        except Exception as e:
            print(f"Database cleanup failed: {str(e)}")
            return False


# Factory functions for easy creation of DatabaseManager instances

def create_database_manager(model_type: ModelType) -> DatabaseManager:
    """
    Factory function to create DatabaseManager for specified model type.
    
    Args:
        model_type: Model type to create manager for
        
    Returns:
        DatabaseManager: Configured database manager
    """
    return DatabaseManager(model_type=model_type)


def create_resnet18_manager() -> DatabaseManager:
    """
    Convenience function to create ResNet18 database manager.
    
    Returns:
        DatabaseManager: ResNet18 database manager
    """
    return create_database_manager(ModelType.RESNET18)


def create_resnet50_manager() -> DatabaseManager:
    """
    Convenience function to create ResNet50 database manager.
    
    Returns:
        DatabaseManager: ResNet50 database manager
    """
    return create_database_manager(ModelType.RESNET50)


def create_legacy_manager(database_path: str, collection_name: str = "ir_embeddings") -> DatabaseManager:
    """
    Create database manager using legacy parameters for backward compatibility.
    
    Args:
        database_path: Path to the database directory
        collection_name: Name of the collection
        
    Returns:
        DatabaseManager: Configured database manager
    """
    return DatabaseManager(database_path=database_path, collection_name=collection_name)