#!/usr/bin/env python3
"""
Test script for IndexManager to verify fixes.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Set PYTHONPATH to include src
os.environ['PYTHONPATH'] = src_path + ':' + os.environ.get('PYTHONPATH', '')

try:
    from database.index_manager import IndexManager, IndexConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct import...")
    # Try importing directly without the package structure
    sys.path.insert(0, os.path.join(src_path, 'database'))
    from index_manager import IndexManager, IndexConfig

def test_index_manager():
    """Test the IndexManager functionality."""
    
    # Database path
    db_path = "/Users/luigicirillo/Desktop/Work/Image-Processing/data/chroma_db_final"
    
    print("=== Testing IndexManager ===")
    
    # Test 1: Initialize IndexManager
    print("\n1. Testing initialization...")
    index_manager = IndexManager(db_path, "ir_embeddings")
    
    # Check if ready before initialization
    print(f"Ready before init: {index_manager.is_ready()}")
    
    # Initialize
    success = index_manager.initialize()
    print(f"Initialization successful: {success}")
    print(f"Ready after init: {index_manager.is_ready()}")
    
    if not success:
        print("Failed to initialize IndexManager. Exiting test.")
        return False
    
    # Test 2: Get collection info
    print("\n2. Testing collection info...")
    info = index_manager.get_collection_info()
    print(f"Collection info: {info}")
    
    # Test 3: Get index statistics
    print("\n3. Testing index statistics...")
    stats = index_manager.get_index_statistics()
    print(f"Index statistics keys: {list(stats.keys())}")
    print(f"Total embeddings: {stats.get('total_embeddings', 'N/A')}")
    
    # Test 4: Validate index integrity
    print("\n4. Testing index validation...")
    validation = index_manager.validate_index_integrity()
    print(f"Index is valid: {validation.get('is_valid', False)}")
    print(f"Validation errors: {len(validation.get('errors', []))}")
    print(f"Validation warnings: {len(validation.get('warnings', []))}")
    
    # Test 5: Create HNSW index
    print("\n5. Testing HNSW index creation...")
    try:
        index_success = index_manager.create_hnsw_index()
        print(f"HNSW index creation successful: {index_success}")
    except Exception as e:
        print(f"HNSW index creation failed: {str(e)}")
    
    # Test 6: Export configuration
    print("\n6. Testing configuration export...")
    config_file = "/tmp/index_config_test.json"
    export_success = index_manager.export_index_config(config_file)
    print(f"Configuration export successful: {export_success}")
    
    if export_success and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            print(f"Config file size: {len(f.read())} bytes")
        os.remove(config_file)  # Clean up
    
    print("\n=== Test completed ===")
    return True

if __name__ == "__main__":
    test_index_manager()
