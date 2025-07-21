#!/usr/bin/env python3
"""
Test script to verify that different query images produce different results.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query.processor import QueryProcessor

def create_test_image(pattern: str, filename: str) -> str:
    """Create a test IR-like image with different patterns."""
    # Create 224x224 grayscale image
    if pattern == "random":
        img_array = np.random.rand(224, 224) * 255
    elif pattern == "gradient":
        img_array = np.zeros((224, 224))
        for i in range(224):
            img_array[i, :] = (i / 224) * 255
    elif pattern == "circles":
        img_array = np.zeros((224, 224))
        center = 112
        for i in range(224):
            for j in range(224):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if 50 < dist < 70 or 90 < dist < 110:
                    img_array[i, j] = 255
    elif pattern == "stripes":
        img_array = np.zeros((224, 224))
        for i in range(224):
            if (i // 20) % 2 == 0:
                img_array[i, :] = 255
    else:  # solid
        img_array = np.ones((224, 224)) * 128
    
    # Convert to uint8 and save
    img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(filename)
    return filename

def main():
    print("🧪 Testing Different Query Images")
    print("=" * 50)
    
    # Initialize processor
    processor = QueryProcessor('data/chroma_db_final')
    processor.initialize()
    
    # Clear cache to ensure fresh results
    processor.clear_cache()
    
    # Create different test images
    patterns = ["random", "gradient", "circles", "stripes", "solid"]
    
    results_summary = []
    
    for i, pattern in enumerate(patterns):
        print(f"\n🔍 Test {i+1}: {pattern.upper()} pattern")
        
        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image = create_test_image(pattern, tmp_file.name)
            
            try:
                # Process query with cache disabled
                result = processor.process_query(
                    image_input=test_image,
                    options={
                        'confidence_threshold': 0.1,
                        'similarity_threshold': 0.05,
                        'max_results': 5,
                        'disable_cache': True
                    }
                )
                
                if result.results:
                    top_classes = [r.object_class for r in result.results[:3]]
                    similarities = [f"{r.similarity_score:.3f}" for r in result.results[:3]]
                    confidences = [f"{r.confidence:.3f}" for r in result.results[:3]]
                    
                    print(f"   Results: {len(result.results)} found")
                    print(f"   Top 3 classes: {top_classes}")
                    print(f"   Similarities: {similarities}")
                    print(f"   Confidences: {confidences}")
                    
                    results_summary.append({
                        'pattern': pattern,
                        'classes': top_classes,
                        'similarities': similarities,
                        'confidences': confidences
                    })
                else:
                    print("   No results found")
                    results_summary.append({
                        'pattern': pattern,
                        'classes': [],
                        'similarities': [],
                        'confidences': []
                    })
                    
            except Exception as e:
                print(f"   Error: {e}")
                results_summary.append({
                    'pattern': pattern,
                    'classes': ['ERROR'],
                    'similarities': ['ERROR'],
                    'confidences': ['ERROR']
                })
            finally:
                # Clean up temporary file
                os.unlink(test_image)
    
    # Analysis
    print("\n" + "="*50)
    print("📊 RESULTS ANALYSIS")
    print("="*50)
    
    # Check if all results are identical
    all_identical = True
    first_result = results_summary[0]['classes'] if results_summary else []
    
    for result in results_summary[1:]:
        if result['classes'] != first_result:
            all_identical = False
            break
    
    if all_identical and first_result:
        print("❌ PROBLEM DETECTED: All queries return identical results!")
        print(f"   All results: {first_result}")
        print("\n🔧 POSSIBLE CAUSES:")
        print("   1. Query caching is still active")
        print("   2. Embedding extraction is not working properly")
        print("   3. Database search is returning fixed results")
        print("   4. Image preprocessing is producing identical outputs")
    else:
        print("✅ GOOD: Different queries produce different results!")
        
    print("\n📋 DETAILED RESULTS:")
    for result in results_summary:
        print(f"   {result['pattern']:10} -> {result['classes']}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if all_identical:
        print("   1. Check if caching is properly disabled")
        print("   2. Verify embedding extraction with different inputs")
        print("   3. Test similarity search directly with different embeddings")
        print("   4. Check image preprocessing pipeline")
    else:
        print("   1. System is working correctly with different inputs")
        print("   2. The issue might be specific to your test image")
        print("   3. Try using different real IR images for testing")

if __name__ == "__main__":
    main()