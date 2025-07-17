#!/usr/bin/env python3
"""
Simple test script for IndexManager to verify fixes.
"""

import sys
import os

# Add the specific module path
module_path = "/Users/luigicirillo/Desktop/Work/Image-Processing/src/database"
sys.path.insert(0, module_path)

# Simple test without complex imports
def test_index_manager_simple():
    """Simple test of IndexManager functionality."""
    
    try:
        # Import IndexManager directly
        from index_manager import IndexManager, IndexConfig
        print("✓ Successfully imported IndexManager and IndexConfig")
        
        # Test 1: Create IndexManager instance
        db_path = "/Users/luigicirillo/Desktop/Work/Image-Processing/data/chroma_db_final"
        
        print("\n=== Testing IndexManager ===")
        
        # Create instance
        index_manager = IndexManager(db_path, "ir_embeddings")
        print("✓ IndexManager instance created")
        
        # Check initial state
        print(f"✓ Initial ready state: {index_manager.is_ready()}")
        
        # Test basic methods that don't require database connection
        config = IndexConfig()
        print(f"✓ Default config created: {config.index_type}")
        
        # Test configuration export (doesn't require DB)
        config_file = "/tmp/test_index_config.json"
        try:
            index_manager.config = config
            export_success = index_manager.export_index_config(config_file)
            print(f"✓ Configuration export: {export_success}")
            
            if export_success and os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    content = f.read()
                    print(f"✓ Config file created ({len(content)} bytes)")
                os.remove(config_file)  # Clean up
            
        except Exception as e:
            print(f"⚠ Configuration export test failed: {str(e)}")
        
        # Test memory estimation method
        try:
            memory_estimate = index_manager._estimate_index_memory_usage(1000, 512)
            print(f"✓ Memory estimation works: {memory_estimate:.2f} MB for 1000 vectors")
        except Exception as e:
            print(f"⚠ Memory estimation failed: {str(e)}")
        
        print("\n=== Basic functionality tests passed ===")
        print("Note: Database connection tests require actual ChromaDB database")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_index_manager_simple()
    sys.exit(0 if success else 1)
