#!/usr/bin/env python3
"""
Diagnostic script to identify why similarity and confidence scores are always the same.

This script will help identify the root cause of consistent scoring issues
by testing different components and showing intermediate values.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query.processor import QueryProcessor
from src.database.similarity_searcher import SimilaritySearcher, SearchConfig
from src.embedding.extractor import EmbeddingExtractor
from src.models.data_models import SimilarityResult


def diagnose_embedding_extraction(extractor: EmbeddingExtractor, image_paths: List[str]) -> Dict[str, Any]:
    """Diagnose embedding extraction to check for consistency issues."""
    print("🔍 Diagnosing embedding extraction...")
    
    embeddings = []
    for i, image_path in enumerate(image_paths):
        try:
            # Load and process image
            from PIL import Image
            img = Image.open(image_path).convert('L').resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Extract embedding
            embedding = extractor.extract_embedding(img_array)
            embeddings.append(embedding)
            
            print(f"   Image {i+1}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
            print(f"   First 5 values: {embedding[:5]}")
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {e}")
    
    # Check for identical embeddings
    if len(embeddings) > 1:
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
                print(f"   Cosine similarity between embedding {i+1} and {j+1}: {sim:.6f}")
        
        return {
            'embeddings_count': len(embeddings),
            'similarities': similarities,
            'all_identical': all(s > 0.999 for s in similarities),
            'very_similar': all(s > 0.95 for s in similarities)
        }
    
    return {'embeddings_count': len(embeddings)}


def diagnose_similarity_calculation(searcher: SimilaritySearcher, query_embedding: np.ndarray) -> Dict[str, Any]:
    """Diagnose similarity calculation by intercepting intermediate values."""
    print("🔍 Diagnosing similarity calculation...")
    
    # Store original method
    original_convert = searcher._convert_chroma_results
    
    # Create diagnostic wrapper
    def diagnostic_convert(chroma_results):
        print(f"   Raw ChromaDB results: {chroma_results}")
        
        ids = chroma_results.get("ids", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0] if chroma_results.get("distances") else []
        metadatas = chroma_results.get("metadatas", [[]])[0] if chroma_results.get("metadatas") else []
        
        print(f"   IDs: {ids}")
        print(f"   Raw distances: {distances}")
        
        # Call original method
        results = original_convert(chroma_results)
        
        # Show conversion results
        for i, result in enumerate(results):
            raw_distance = distances[i] if i < len(distances) else "N/A"
            print(f"   Result {i+1}:")
            print(f"     Raw distance: {raw_distance}")
            print(f"     Converted similarity: {result.similarity_score:.6f}")
            print(f"     Calculated confidence: {result.confidence:.6f}")
            print(f"     Object class: {result.object_class}")
        
        return results
    
    # Replace method temporarily
    searcher._convert_chroma_results = diagnostic_convert
    
    try:
        # Perform search
        results, metrics = searcher.search_similar(query_embedding, k=5)
        
        # Restore original method
        searcher._convert_chroma_results = original_convert
        
        return {
            'results_count': len(results),
            'unique_similarities': len(set(r.similarity_score for r in results)),
            'unique_confidences': len(set(r.confidence for r in results)),
            'similarity_range': (min(r.similarity_score for r in results), max(r.similarity_score for r in results)) if results else (0, 0),
            'confidence_range': (min(r.confidence for r in results), max(r.confidence for r in results)) if results else (0, 0)
        }
    
    except Exception as e:
        # Restore original method
        searcher._convert_chroma_results = original_convert
        print(f"   ❌ Error during similarity search: {e}")
        return {'error': str(e)}


def diagnose_confidence_calculation(searcher: SimilaritySearcher) -> Dict[str, Any]:
    """Diagnose confidence calculation by testing with different inputs."""
    print("🔍 Diagnosing confidence calculation...")
    
    # Test confidence calculation with different similarity scores and ranks
    test_cases = [
        (0.9, 0, 5),  # High similarity, rank 1
        (0.8, 1, 5),  # Good similarity, rank 2
        (0.7, 2, 5),  # Medium similarity, rank 3
        (0.6, 3, 5),  # Lower similarity, rank 4
        (0.5, 4, 5),  # Low similarity, rank 5
    ]
    
    confidence_results = []
    for similarity, rank, total in test_cases:
        confidence = searcher._calculate_confidence(similarity, rank, total)
        confidence_results.append(confidence)
        print(f"   Similarity: {similarity:.1f}, Rank: {rank+1}, Confidence: {confidence:.6f}")
    
    return {
        'test_cases': len(test_cases),
        'unique_confidences': len(set(confidence_results)),
        'confidence_values': confidence_results,
        'all_same': len(set(confidence_results)) == 1
    }


def diagnose_database_content(searcher: SimilaritySearcher) -> Dict[str, Any]:
    """Diagnose database content to check for duplicate embeddings."""
    print("🔍 Diagnosing database content...")
    
    try:
        # Get collection info
        collection = searcher.collection
        if not collection:
            return {'error': 'Collection not available'}
        
        count = collection.count()
        print(f"   Total embeddings in database: {count}")
        
        # Sample a few embeddings to check for duplicates
        sample_results = collection.query(
            query_embeddings=[np.random.random(512).tolist()],  # Random query
            n_results=min(10, count),
            include=['embeddings', 'metadatas', 'distances']
        )
        
        if sample_results.get('embeddings'):
            embeddings = sample_results['embeddings'][0]
            print(f"   Sample embeddings retrieved: {len(embeddings)}")
            
            # Check for identical embeddings
            if len(embeddings) > 1:
                identical_count = 0
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        if np.allclose(embeddings[i], embeddings[j], atol=1e-6):
                            identical_count += 1
                
                print(f"   Identical embedding pairs found: {identical_count}")
                
                return {
                    'total_count': count,
                    'sample_size': len(embeddings),
                    'identical_pairs': identical_count,
                    'potential_duplicates': identical_count > 0
                }
        
        return {'total_count': count, 'sample_retrieved': False}
    
    except Exception as e:
        print(f"   ❌ Error accessing database: {e}")
        return {'error': str(e)}


def diagnose_caching_issues(processor: QueryProcessor, test_image: str) -> Dict[str, Any]:
    """Diagnose potential caching issues."""
    print("🔍 Diagnosing caching issues...")
    
    try:
        # Clear cache first
        processor.clear_cache()
        
        # Run same query multiple times
        results = []
        for i in range(3):
            print(f"   Running query {i+1}/3...")
            result = processor.process_query(test_image, query_id=f"test_{i}")
            results.append(result)
        
        # Check if results are identical
        if len(results) > 1:
            first_result = results[0]
            identical_results = True
            
            for result in results[1:]:
                if len(result.results) != len(first_result.results):
                    identical_results = False
                    break
                
                for j, (r1, r2) in enumerate(zip(first_result.results, result.results)):
                    if (abs(r1.similarity_score - r2.similarity_score) > 1e-6 or
                        abs(r1.confidence - r2.confidence) > 1e-6):
                        identical_results = False
                        break
                
                if not identical_results:
                    break
            
            return {
                'queries_run': len(results),
                'identical_results': identical_results,
                'cache_suspected': identical_results
            }
    
    except Exception as e:
        print(f"   ❌ Error testing caching: {e}")
        return {'error': str(e)}
    
    return {'queries_run': 0}


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(description="Diagnose score consistency issues")
    parser.add_argument('--database', '-d', required=True, help='Database path')
    parser.add_argument('--model', '-m', help='Model path (optional)')
    parser.add_argument('--collection', default='ir_embeddings', help='Collection name')
    parser.add_argument('--test-images', nargs='+', help='Test image paths')
    parser.add_argument('--test-image', help='Single test image path')
    
    args = parser.parse_args()
    
    print("🔧 IR Image Classification - Score Consistency Diagnostic Tool")
    print("=" * 70)
    
    try:
        # Initialize components
        print("Initializing system components...")
        
        extractor = EmbeddingExtractor()
        if args.model:
            extractor.load_model(args.model)
        else:
            extractor.load_model(None)  # Default model
        
        searcher = SimilaritySearcher(args.database, args.collection)
        searcher.initialize()
        
        processor = QueryProcessor(
            database_path=args.database,
            model_path=args.model,
            collection_name=args.collection
        )
        processor.initialize()
        
        print("✅ Components initialized successfully\n")
        
        # Run diagnostics
        diagnostics = {}
        
        # 1. Test embedding extraction
        if args.test_images:
            diagnostics['embedding_extraction'] = diagnose_embedding_extraction(
                extractor, args.test_images
            )
        
        # 2. Test similarity calculation
        if args.test_image or args.test_images:
            test_img = args.test_image or args.test_images[0]
            from PIL import Image
            img = Image.open(test_img).convert('L').resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            query_embedding = extractor.extract_embedding(img_array)
            
            diagnostics['similarity_calculation'] = diagnose_similarity_calculation(
                searcher, query_embedding
            )
        
        # 3. Test confidence calculation
        diagnostics['confidence_calculation'] = diagnose_confidence_calculation(searcher)
        
        # 4. Test database content
        diagnostics['database_content'] = diagnose_database_content(searcher)
        
        # 5. Test caching issues
        if args.test_image:
            diagnostics['caching_issues'] = diagnose_caching_issues(processor, args.test_image)
        
        # Summary
        print("\n" + "=" * 70)
        print("🔍 DIAGNOSTIC SUMMARY")
        print("=" * 70)
        
        issues_found = []
        
        for test_name, results in diagnostics.items():
            print(f"\n{test_name.replace('_', ' ').title()}:")
            
            if 'error' in results:
                print(f"   ❌ Error: {results['error']}")
                issues_found.append(f"{test_name}: {results['error']}")
            else:
                for key, value in results.items():
                    print(f"   {key}: {value}")
                
                # Check for specific issues
                if test_name == 'embedding_extraction':
                    if results.get('all_identical'):
                        issues_found.append("All embeddings are identical - model may not be working")
                    elif results.get('very_similar'):
                        issues_found.append("Embeddings are very similar - limited discrimination")
                
                elif test_name == 'similarity_calculation':
                    if results.get('unique_similarities', 0) <= 1:
                        issues_found.append("All similarity scores are the same")
                    if results.get('unique_confidences', 0) <= 1:
                        issues_found.append("All confidence scores are the same")
                
                elif test_name == 'confidence_calculation':
                    if results.get('all_same'):
                        issues_found.append("Confidence calculation produces identical values")
                
                elif test_name == 'database_content':
                    if results.get('potential_duplicates'):
                        issues_found.append("Database contains duplicate embeddings")
                
                elif test_name == 'caching_issues':
                    if results.get('cache_suspected'):
                        issues_found.append("Caching may be returning identical results")
        
        print(f"\n🚨 ISSUES IDENTIFIED ({len(issues_found)}):")
        if issues_found:
            for i, issue in enumerate(issues_found, 1):
                print(f"   {i}. {issue}")
        else:
            print("   No obvious issues detected.")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if any("identical" in issue.lower() for issue in issues_found):
            print("   - Check if the model is properly loaded and trained")
            print("   - Verify that different images produce different embeddings")
            print("   - Consider retraining or using a different model")
        
        if any("confidence" in issue.lower() for issue in issues_found):
            print("   - Review confidence calculation formula")
            print("   - Check if similarity-to-confidence mapping is too narrow")
            print("   - Consider using different confidence strategies")
        
        if any("database" in issue.lower() for issue in issues_found):
            print("   - Rebuild the database with unique embeddings")
            print("   - Check data preprocessing pipeline")
        
        if any("caching" in issue.lower() for issue in issues_found):
            print("   - Disable caching temporarily to test")
            print("   - Clear cache between different queries")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()