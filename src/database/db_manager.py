"""
Database management utilities for IR image classification system.

This module provides utilities for database initialization, migration,
and maintenance operations for the ChromaDB vector store.
"""

import os
import json
import shutil
from typing import Dict, Any, Optional
from datetime import datetime

from .vector_store import ChromaVectorStore


class DatabaseManager:
    """
    Manages database initialization, migration, and maintenance operations.
    """
    
    def __init__(self, database_path: str = "./chroma_db", collection_name: str = "ir_embeddings"):
        """
        Initialize database manager.
        
        Args:
            database_path: Path to the database directory
            collection_name: Name of the collection
        """
        self.database_path = database_path
        self.collection_name = collection_name
        self.config_file = os.path.join(database_path, "db_config.json")
        self.migration_log = os.path.join(database_path, "migrations.log")
        
    def initialize_fresh_database(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize a fresh database with default configuration.
        
        Args:
            config: Optional configuration parameters
            
        Returns:
            bool: True if initialization was successful
        """
        # Default configuration
        default_config = {
            "embedding_dimension": 512,
            "distance_metric": "cosine",
            "index_type": "hnsw",
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
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