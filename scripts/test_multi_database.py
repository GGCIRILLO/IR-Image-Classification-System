#!/usr/bin/env python3
"""
Test script for multi-database configuration functionality.

This script demonstrates the enhanced database configuration system
that supports multiple databases for different model types.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager, ModelType, create_resnet18_manager, create_resnet50_manager
from src.database.similarity_searcher import SimilaritySearcher, create_resnet18_searcher, create_resnet50_searcher


def test_database_manager():
    """Test the enhanced DatabaseManager functionality."""
    print("="*60)
    print("TESTING DATABASE MANAGER")
    print("="*60)
    
    # Test default manager (should default to ResNet50)
    print("\n1. Testing default manager:")
    default_manager = DatabaseManager()
    info = default_manager.get_database_info()
    print(f"   Model type: {info['model_type']}")
    print(f"   Database path: {info['database_path']}")
    print(f"   Collection: {info['collection_name']}")
    
    # Test ResNet18 manager
    print("\n2. Testing ResNet18 manager:")
    resnet18_manager = create_resnet18_manager()
    info = resnet18_manager.get_database_info()
    print(f"   Model type: {info['model_type']}")
    print(f"   Database path: {info['database_path']}")
    print(f"   Collection: {info['collection_name']}")
    
    # Test ResNet50 manager
    print("\n3. Testing ResNet50 manager:")
    resnet50_manager = create_resnet50_manager()
    info = resnet50_manager.get_database_info()
    print(f"   Model type: {info['model_type']}")
    print(f"   Database path: {info['database_path']}")
    print(f"   Collection: {info['collection_name']}")
    
    # Test switching databases
    print("\n4. Testing database switching:")
    manager = DatabaseManager(model_type=ModelType.RESNET50)
    print(f"   Initial: {manager.get_database_info()['model_type']}")
    
    success = manager.switch_database(ModelType.RESNET18)
    print(f"   Switch to ResNet18: {'✅' if success else '❌'}")
    print(f"   Current: {manager.get_database_info()['model_type']}")
    
    success = manager.switch_database(ModelType.RESNET50)
    print(f"   Switch to ResNet50: {'✅' if success else '❌'}")
    print(f"   Current: {manager.get_database_info()['model_type']}")
    
    # Test available model types
    print("\n5. Available model types:")
    available_types = manager.get_available_model_types()
    for model_type in available_types:
        print(f"   - {model_type.value}")
    
    # Test multi-database status
    print("\n6. Multi-database status:")
    manager.print_multi_database_status()


def test_similarity_searcher():
    """Test the enhanced SimilaritySearcher functionality."""
    print("\n" + "="*60)
    print("TESTING SIMILARITY SEARCHER")
    print("="*60)
    
    # Test default searcher (should default to ResNet50)
    print("\n1. Testing default searcher:")
    default_searcher = SimilaritySearcher()
    info = default_searcher.get_database_info()
    print(f"   Model type: {info['model_type']}")
    print(f"   Database path: {info['database_path']}")
    print(f"   Collection: {info['collection_name']}")
    
    # Test ResNet18 searcher
    print("\n2. Testing ResNet18 searcher:")
    resnet18_searcher = create_resnet18_searcher()
    info = resnet18_searcher.get_database_info()
    print(f"   Model type: {info['model_type']}")
    print(f"   Database path: {info['database_path']}")
    print(f"   Collection: {info['collection_name']}")
    
    # Test ResNet50 searcher
    print("\n3. Testing ResNet50 searcher:")
    resnet50_searcher = create_resnet50_searcher()
    info = resnet50_searcher.get_database_info()
    print(f"   Model type: {info['model_type']}")
    print(f"   Database path: {info['database_path']}")
    print(f"   Collection: {info['collection_name']}")
    
    # Test switching databases
    print("\n4. Testing searcher database switching:")
    searcher = SimilaritySearcher(model_type=ModelType.RESNET50)
    print(f"   Initial: {searcher.get_current_model_type().value}")
    
    success = searcher.switch_database(ModelType.RESNET18)
    print(f"   Switch to ResNet18: {'✅' if success else '❌'}")
    print(f"   Current: {searcher.get_current_model_type().value}")
    
    success = searcher.switch_database(ModelType.RESNET50)
    print(f"   Switch to ResNet50: {'✅' if success else '❌'}")
    print(f"   Current: {searcher.get_current_model_type().value}")
    
    # Test legacy compatibility
    print("\n5. Testing legacy compatibility:")
    legacy_searcher = SimilaritySearcher(
        database_path="./data/vector_db",
        collection_name="ir_embeddings"
    )
    info = legacy_searcher.get_database_info()
    print(f"   Model type: {info['model_type']}")
    print(f"   Database path: {info['database_path']}")
    print(f"   Collection: {info['collection_name']}")


def test_configuration_files():
    """Test configuration file creation and management."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION FILES")
    print("="*60)
    
    # Create a manager to trigger config file creation
    manager = DatabaseManager()
    
    # Check if multi-database config was created
    config_file = Path("config/multi_database_config.json")
    if config_file.exists():
        print("✅ Multi-database configuration file created")
        print(f"   Location: {config_file}")
        
        # Show file size
        size = config_file.stat().st_size
        print(f"   Size: {size} bytes")
    else:
        print("❌ Multi-database configuration file not found")
    
    # Test database status for all types
    print("\nDatabase status for all model types:")
    status = manager.get_all_database_status()
    for model_type, info in status.items():
        exists = "✅" if info.get('exists', False) else "❌"
        print(f"   {model_type}: {exists} ({info.get('database_path', 'unknown')})")


def main():
    """Run all tests."""
    print("Multi-Database Configuration Test")
    print("This test demonstrates the enhanced database system")
    print("that supports multiple databases for different model types.\n")
    
    try:
        test_database_manager()
        test_similarity_searcher()
        test_configuration_files()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nThe multi-database system is ready for use!")
        print("You can now:")
        print("- Use separate databases for ResNet18 and ResNet50")
        print("- Switch between databases dynamically")
        print("- Maintain backward compatibility with existing code")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())