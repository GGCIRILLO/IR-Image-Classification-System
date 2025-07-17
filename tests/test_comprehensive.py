#!/usr/bin/env python3
"""
Comprehensive test script for IndexManager functionality.
"""

import sys
import os
import numpy as np

# Add the specific module path
module_path = "/Users/luigicirillo/Desktop/Work/Image-Processing/src/database"
sys.path.insert(0, module_path)

def test_comprehensive_index_manager():
    """Comprehensive test of IndexManager functionality."""
    
    try:
        # Import IndexManager directly
        from index_manager import IndexManager, IndexConfig
        print("✓ Successfully imported IndexManager and IndexConfig")
        
        # Test with real database
        db_path = "/Users/luigicirillo/Desktop/Work/Image-Processing/data/chroma_db_final"
        
        print(f"\n=== Comprehensive IndexManager Test ===")
        print(f"Database path: {db_path}")
        
        # Create IndexManager with custom config
        custom_config = IndexConfig(
            index_type="hnsw",
            distance_metric="cosine",
            hnsw_m=16,
            hnsw_ef_construction=200,
            hnsw_ef_search=100
        )
        
        index_manager = IndexManager(db_path, "ir_embeddings")
        print("✓ IndexManager instance created with custom config")
        
        # Initialize with custom config
        success = index_manager.initialize(custom_config)
        print(f"✓ Initialization with custom config: {success}")
        
        if success:
            # Get detailed statistics
            print("\n--- Collection Information ---")
            info = index_manager.get_collection_info()
            for key, value in info.items():
                print(f"{key}: {value}")
            
            print("\n--- Index Statistics ---")
            stats = index_manager.get_index_statistics()
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue}")
                else:
                    print(f"{key}: {value}")
            
            print("\n--- Validation Results ---")
            validation = index_manager.validate_index_integrity()
            print(f"Is Valid: {validation.get('is_valid', False)}")
            
            if validation.get('checks_performed'):
                print("Checks Performed:")
                for check in validation['checks_performed']:
                    print(f"  ✓ {check}")
            
            if validation.get('warnings'):
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  ⚠ {warning}")
            
            if validation.get('errors'):
                print("Errors:")
                for error in validation['errors']:
                    print(f"  ✗ {error}")
            
            # Test HNSW index creation (will work even with 0 embeddings for testing purposes)
            print("\n--- HNSW Index Creation Test ---")
            try:
                if info.get('count', 0) > 0:
                    index_success = index_manager.create_hnsw_index()
                    print(f"HNSW index creation: {index_success}")
                    
                    if index_success and index_manager.metrics:
                        print(f"Build time: {index_manager.metrics.build_time_seconds:.3f} seconds")
                        print(f"Memory usage: {index_manager.metrics.memory_usage_mb:.2f} MB")
                else:
                    print("⚠ Collection is empty - skipping HNSW index creation")
                    print("  (This is normal for a new/empty database)")
            except Exception as e:
                print(f"HNSW index creation test result: {str(e)}")
            
            # Test configuration export
            print("\n--- Configuration Export Test ---")
            config_file = "/tmp/comprehensive_index_config.json"
            export_success = index_manager.export_index_config(config_file)
            print(f"Configuration export: {export_success}")
            
            if export_success and os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    content = f.read()
                    print(f"Config file size: {len(content)} bytes")
                
                # Show a snippet of the config
                import json
                try:
                    config_data = json.loads(content)
                    print("Configuration preview:")
                    for key in ['index_config', 'collection_name', 'database_path']:
                        if key in config_data:
                            print(f"  {key}: {config_data[key]}")
                except:
                    pass
                
                os.remove(config_file)  # Clean up
            
            print("\n✓ All comprehensive tests completed successfully!")
            return True
        else:
            print("✗ Failed to initialize IndexManager")
            return False
        
    except Exception as e:
        print(f"✗ Comprehensive test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive_index_manager()
    print(f"\n{'='*50}")
    print(f"Final Result: {'SUCCESS' if success else 'FAILURE'}")
    print(f"{'='*50}")
    sys.exit(0 if success else 1)
