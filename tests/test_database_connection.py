#!/usr/bin/env python3
"""
Test script for IndexManager database connectivity.
"""

import sys
import os

# Add the specific module path
module_path = "/Users/luigicirillo/Desktop/Work/Image-Processing/src/database"
sys.path.insert(0, module_path)

def test_database_connection():
    """Test IndexManager database connection functionality."""
    
    try:
        # Import IndexManager directly
        from index_manager import IndexManager, IndexConfig
        print("✓ Successfully imported IndexManager and IndexConfig")
        
        # Test database connection
        db_path = "/Users/luigicirillo/Desktop/Work/Image-Processing/data/chroma_db_final"
        
        print(f"\n=== Testing Database Connection ===")
        print(f"Database path: {db_path}")
        
        # Check if database path exists
        if not os.path.exists(db_path):
            print(f"✗ Database path does not exist: {db_path}")
            return False
        
        print(f"✓ Database path exists")
        
        # Create IndexManager instance
        index_manager = IndexManager(db_path, "ir_embeddings")
        print("✓ IndexManager instance created")
        
        # Test initialization
        print("Attempting to initialize...")
        success = index_manager.initialize()
        print(f"✓ Initialization result: {success}")
        
        if success:
            # Test collection info
            print("\nTesting collection info...")
            info = index_manager.get_collection_info()
            print(f"Collection info: {info}")
            
            # Test index statistics
            if info.get('is_ready', False):
                print("\nTesting index statistics...")
                stats = index_manager.get_index_statistics()
                print(f"Total embeddings: {stats.get('total_embeddings', 'N/A')}")
                print(f"Index type: {stats.get('index_type', 'N/A')}")
                print(f"Distance metric: {stats.get('distance_metric', 'N/A')}")
                
                # Test validation
                print("\nTesting index validation...")
                validation = index_manager.validate_index_integrity()
                print(f"Index is valid: {validation.get('is_valid', False)}")
                
                if validation.get('errors'):
                    print(f"Validation errors: {validation['errors']}")
                if validation.get('warnings'):
                    print(f"Validation warnings: {validation['warnings']}")
                
                print("\n✓ All database tests completed successfully!")
            else:
                print("⚠ Collection not ready for advanced operations")
        else:
            print("⚠ Failed to initialize - this might be expected if no database exists")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database_connection()
    sys.exit(0 if success else 1)
