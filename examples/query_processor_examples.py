"""
Example usage of QueryProcessor with different configurations.

This script demonstrates how to use the QueryProcessor class with
various configuration presets for different deployment environments.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query import (
    QueryProcessor, 
    QueryProcessorConfig,
    MilitaryQueryConfig,
    DevelopmentQueryConfig,
    ValidationMode,
    CachePolicy,
    get_config_for_environment
)
from src.models.data_models import IRImage


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('query_processor_example.log')
        ]
    )


def example_basic_usage():
    """Basic QueryProcessor usage example."""
    print("\n" + "="*60)
    print("Basic QueryProcessor Usage Example")
    print("="*60)
    
    # Use default configuration
    config = QueryProcessorConfig()
    
    processor = QueryProcessor(
        database_path="data/chroma_db_final",
        model_path=None,  # Will be loaded if available
        collection_name="ir_embeddings",
        config=config.to_dict()
    )
    
    try:
        if processor.initialize():
            print("✓ QueryProcessor initialized with default configuration")
            
            # Example query
            sample_image = "data/processed/M-1A1 Abrams Tank/image_001.png"
            if Path(sample_image).exists():
                result = processor.process_query(
                    image_input=sample_image,
                    query_id="basic_example_001"
                )
                
                print(f"✓ Query completed in {result.processing_time:.3f}s")
                print(f"✓ Found {len(result.results)} similar images")
                
                for i, sim_result in enumerate(result.results[:3], 1):
                    print(f"  {i}. {sim_result.object_class} "
                          f"(similarity: {sim_result.similarity_score:.3f}, "
                          f"confidence: {sim_result.confidence:.3f})")
            else:
                print(f"Sample image not found: {sample_image}")
    
    except Exception as e:
        print(f"✗ Error: {e}")


def example_military_deployment():
    """Military deployment configuration example."""
    print("\n" + "="*60)
    print("Military Deployment Configuration Example")
    print("="*60)
    
    # Use military-specific configuration
    config = MilitaryQueryConfig()
    
    # Validate configuration
    try:
        config.validate()
        print("✓ Military configuration validated")
    except ValueError as e:
        print(f"✗ Configuration validation failed: {e}")
        return
    
    processor = QueryProcessor(
        database_path="data/chroma_db_final",
        model_path=None,
        collection_name="ir_embeddings",
        config=config.to_dict()
    )
    
    print(f"Military Configuration Settings:")
    print(f"  - Max query time: {config.max_query_time}s")
    print(f"  - Min confidence threshold: {config.min_confidence_threshold}")
    print(f"  - Validation mode: {config.validation_mode.value}")
    print(f"  - Cache policy: {config.cache_policy.value}")
    print(f"  - GPU acceleration: {config.enable_gpu_acceleration}")
    
    try:
        if processor.initialize():
            print("✓ QueryProcessor initialized for military deployment")
            
            # Demonstrate strict validation
            import numpy as np
            
            # Create a test image that may not meet strict IR requirements
            test_image = np.random.random((224, 224)).astype(np.float32)
            
            try:
                result = processor.process_query(
                    image_input=test_image,
                    query_id="military_test_001",
                    options={'strict_validation': True}
                )
                print("✓ Strict validation passed")
            except Exception as e:
                print(f"✗ Strict validation failed as expected: {e}")
    
    except Exception as e:
        print(f"✗ Error: {e}")


def example_development_setup():
    """Development configuration example."""
    print("\n" + "="*60)
    print("Development Configuration Example")
    print("="*60)
    
    # Use development configuration
    config = DevelopmentQueryConfig()
    
    processor = QueryProcessor(
        database_path="data/chroma_db_final",
        model_path=None,
        collection_name="ir_embeddings",
        config=config.to_dict()
    )
    
    print(f"Development Configuration Settings:")
    print(f"  - Debug mode: {config.debug_mode}")
    print(f"  - Log level: {config.log_level}")
    print(f"  - Validation mode: {config.validation_mode.value}")
    print(f"  - Allow color images: {config.allow_color_images}")
    print(f"  - Min confidence threshold: {config.min_confidence_threshold}")
    
    try:
        if processor.initialize():
            print("✓ QueryProcessor initialized for development")
            
            # Test with various image types
            test_cases = [
                # Numpy array test
                np.random.random((224, 224)).astype(np.float32),
                # Color image simulation (would be converted to grayscale)
                np.random.random((224, 224, 3)).astype(np.float32),
            ]
            
            for i, test_image in enumerate(test_cases, 1):
                try:
                    result = processor.process_query(
                        image_input=test_image,
                        query_id=f"dev_test_{i:03d}",
                        options={'strict_validation': False}
                    )
                    print(f"✓ Test case {i} passed: {len(result.results)} results in {result.processing_time:.3f}s")
                except Exception as e:
                    print(f"✗ Test case {i} failed: {e}")
    
    except Exception as e:
        print(f"✗ Error: {e}")


def example_custom_configuration():
    """Custom configuration example."""
    print("\n" + "="*60)
    print("Custom Configuration Example")
    print("="*60)
    
    # Create custom configuration
    custom_config = QueryProcessorConfig(
        max_query_time=3.0,  # Extended time for complex queries
        min_confidence_threshold=0.6,  # Lower threshold for more results
        top_k_results=10,  # More results
        validation_mode=ValidationMode.RELAXED,
        enable_result_reranking=True,
        enable_gpu_acceleration=True,
        cache_policy=CachePolicy.ENABLED,
        max_cache_size=500,
        similarity_threshold=0.4,  # Lower similarity threshold
        custom_preprocessing={'enhance_contrast': True},
        debug_mode=False
    )
    
    try:
        custom_config.validate()
        print("✓ Custom configuration validated")
    except ValueError as e:
        print(f"✗ Custom configuration validation failed: {e}")
        return
    
    processor = QueryProcessor(
        database_path="data/chroma_db_final",
        model_path=None,
        collection_name="ir_embeddings",
        config=custom_config.to_dict()
    )
    
    print(f"Custom Configuration Settings:")
    print(f"  - Max query time: {custom_config.max_query_time}s")
    print(f"  - Top-K results: {custom_config.top_k_results}")
    print(f"  - Similarity threshold: {custom_config.similarity_threshold}")
    print(f"  - Cache size: {custom_config.max_cache_size}")
    print(f"  - Custom preprocessing: {custom_config.custom_preprocessing}")
    
    try:
        if processor.initialize():
            print("✓ QueryProcessor initialized with custom configuration")
            
            # Demonstrate performance monitoring
            import numpy as np
            
            # Run multiple queries to gather statistics
            for i in range(3):
                test_image = np.random.random((224, 224)).astype(np.float32)
                result = processor.process_query(
                    image_input=test_image,
                    query_id=f"custom_test_{i+1:03d}"
                )
                print(f"  Query {i+1}: {len(result.results)} results in {result.processing_time:.3f}s")
            
            # Show performance statistics
            stats = processor.get_performance_stats()
            print(f"\nPerformance Statistics:")
            for metric, values in stats.items():
                if values['count'] > 0:
                    print(f"  {metric}: avg={values['average']:.3f}s, "
                          f"min={values['min']:.3f}s, max={values['max']:.3f}s")
            
            # Validate performance requirements
            validation = processor.validate_system_performance()
            print(f"\nPerformance Validation:")
            for requirement, passed in validation.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {requirement}")
    
    except Exception as e:
        print(f"✗ Error: {e}")


def example_environment_configs():
    """Example of using environment-specific configurations."""
    print("\n" + "="*60)
    print("Environment-Specific Configurations Example")
    print("="*60)
    
    environments = ["development", "production", "military", "testing"]
    
    for env in environments:
        try:
            config = get_config_for_environment(env)
            print(f"\n{env.upper()} Environment:")
            print(f"  - Max query time: {config.max_query_time}s")
            print(f"  - Min confidence: {config.min_confidence_threshold}")
            print(f"  - Validation mode: {config.validation_mode.value}")
            print(f"  - Cache policy: {config.cache_policy.value}")
            print(f"  - Debug mode: {config.debug_mode}")
            
        except ValueError as e:
            print(f"✗ Failed to get config for {env}: {e}")


def main():
    """Run all examples."""
    setup_logging("INFO")
    
    print("IR Image Classification System - QueryProcessor Examples")
    print("This script demonstrates various QueryProcessor configurations and usage patterns.")
    
    # Run all examples
    example_basic_usage()
    example_military_deployment()
    example_development_setup()
    example_custom_configuration()
    example_environment_configs()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check the log file 'query_processor_example.log' for detailed output.")


if __name__ == "__main__":
    main()
