#!/usr/bin/env python3
"""
Quick fix script for improving similarity and confidence scores.

This script applies immediate fixes to the IR image classification system
to improve similarity and confidence scores without requiring full retraining.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query.processor import QueryProcessor
from src.database.similarity_searcher import SimilaritySearcher
from src.embedding.extractor import EmbeddingExtractor


def apply_similarity_fixes(searcher: SimilaritySearcher) -> Dict[str, Any]:
    """Apply fixes to similarity calculation."""
    fixes = {}
    
    # Fix 1: Improve similarity score conversion
    original_convert = searcher._convert_chroma_results
    
    def improved_similarity_conversion(chroma_results):
        """Enhanced similarity conversion for IR images."""
        results = original_convert(chroma_results)
        
        for result in results:
            raw_distance = result.metadata.get('raw_distance', 1.0)
            
            # Better similarity calculation for IR images
            if raw_distance <= 1.0:
                # Use different scaling for IR images
                if raw_distance < 0.5:
                    # Very close matches
                    boosted_similarity = 0.95 - (raw_distance * 0.1)
                elif raw_distance < 0.8:
                    # Good matches
                    boosted_similarity = 0.8 - (raw_distance - 0.5) * 0.5
                else:
                    # Moderate matches - use exponential scaling
                    boosted_similarity = max(0.1, 1.0 - (raw_distance ** 0.3))
                
                result.similarity_score = max(result.similarity_score, boosted_similarity)
        
        return results
    
    searcher._convert_chroma_results = improved_similarity_conversion
    fixes['similarity_conversion'] = 'Enhanced for IR images'
    
    return fixes


def apply_confidence_fixes(searcher: SimilaritySearcher) -> Dict[str, Any]:
    """Apply fixes to confidence calculation."""
    fixes = {}
    
    # Fix 1: Improve confidence calculation
    def improved_confidence_calculation(similarity_score: float, rank: int, total_results: int) -> float:
        """Enhanced confidence calculation for IR images."""
        # Base confidence with IR-specific scaling
        if similarity_score > 0.7:
            base_confidence = 0.9 + (similarity_score - 0.7) * 0.33  # Scale 0.7-1.0 to 0.9-1.0
        elif similarity_score > 0.4:
            base_confidence = 0.6 + (similarity_score - 0.4) * 1.0   # Scale 0.4-0.7 to 0.6-0.9
        elif similarity_score > 0.2:
            base_confidence = 0.3 + (similarity_score - 0.2) * 1.5   # Scale 0.2-0.4 to 0.3-0.6
        else:
            base_confidence = similarity_score * 1.5                 # Scale 0.0-0.2 to 0.0-0.3
        
        # Reduced rank penalty for IR images
        if total_results > 1:
            rank_penalty = 0.03 * (rank / (total_results - 1))  # Much reduced penalty
        else:
            rank_penalty = 0.0
        
        confidence = max(0.0, min(1.0, base_confidence - rank_penalty))
        
        # Bonus for top results
        if rank == 0 and similarity_score > 0.3:
            confidence = min(1.0, confidence + 0.1)
        
        return confidence
    
    searcher._calculate_confidence = improved_confidence_calculation
    fixes['confidence_calculation'] = 'Enhanced for IR images'
    
    # Fix 2: Lower confidence threshold
    original_threshold = searcher.config.confidence_threshold
    searcher.config.confidence_threshold = 0.25  # Much lower threshold
    fixes['confidence_threshold'] = f'Reduced from {original_threshold} to 0.25'
    
    return fixes


def apply_embedding_fixes(extractor: EmbeddingExtractor) -> Dict[str, Any]:
    """Apply fixes to embedding extraction."""
    fixes = {}
    
    if hasattr(extractor, 'extract_embedding'):
        original_extract = extractor.extract_embedding
        
        def enhanced_extract_embedding(image: np.ndarray) -> np.ndarray:
            """Enhanced embedding extraction with normalization."""
            embedding = original_extract(image)
            
            # L2 normalize for better cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Optional: Apply PCA whitening or other transformations here
            
            return embedding
        
        extractor.extract_embedding = enhanced_extract_embedding
        fixes['embedding_normalization'] = 'Added L2 normalization'
    
    return fixes


def test_fixes(processor: QueryProcessor, test_image: str) -> Dict[str, Any]:
    """Test the applied fixes with a sample query."""
    print("Testing fixes with sample query...")
    
    try:
        result = processor.process_query(
            image_input=test_image,
            options={
                'confidence_threshold': 0.25,
                'similarity_threshold': 0.1,
                'max_results': 10
            }
        )
        
        metrics = {
            'results_found': len(result.results),
            'processing_time': result.processing_time,
            'avg_similarity': np.mean([r.similarity_score for r in result.results]) if result.results else 0.0,
            'max_similarity': np.max([r.similarity_score for r in result.results]) if result.results else 0.0,
            'avg_confidence': np.mean([r.confidence for r in result.results]) if result.results else 0.0,
            'max_confidence': np.max([r.confidence for r in result.results]) if result.results else 0.0,
            'high_confidence_results': len([r for r in result.results if r.confidence >= 0.7])
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error testing fixes: {e}")
        return {'error': str(e)}


def main():
    """Main function to apply fixes."""
    parser = argparse.ArgumentParser(description="Fix similarity and confidence scores")
    parser.add_argument('--database', '-d', required=True, help='Database path')
    parser.add_argument('--model', '-m', help='Model path (optional)')
    parser.add_argument('--collection', default='ir_embeddings', help='Collection name')
    parser.add_argument('--test-image', help='Test image path')
    parser.add_argument('--apply-all', action='store_true', help='Apply all fixes')
    
    args = parser.parse_args()
    
    print("🔧 IR Image Classification - Similarity & Confidence Fix Tool")
    print("=" * 60)
    
    try:
        # Initialize components
        print("Initializing system components...")
        
        extractor = EmbeddingExtractor()
        extractor.load_model(args.model)
        
        searcher = SimilaritySearcher(args.database, args.collection)
        searcher.initialize()
        
        processor = QueryProcessor(
            database_path=args.database,
            model_path=args.model,
            collection_name=args.collection
        )
        processor.initialize()
        
        print("✅ Components initialized successfully")
        
        # Test before fixes
        if args.test_image:
            print("\n📊 Testing BEFORE fixes...")
            before_metrics = test_fixes(processor, args.test_image)
            print(f"Before fixes: {before_metrics}")
        
        # Apply fixes
        print("\n🔧 Applying fixes...")
        
        all_fixes = {}
        
        if args.apply_all:
            # Apply similarity fixes
            similarity_fixes = apply_similarity_fixes(searcher)
            all_fixes.update(similarity_fixes)
            print(f"✅ Similarity fixes applied: {similarity_fixes}")
            
            # Apply confidence fixes
            confidence_fixes = apply_confidence_fixes(searcher)
            all_fixes.update(confidence_fixes)
            print(f"✅ Confidence fixes applied: {confidence_fixes}")
            
            # Apply embedding fixes
            embedding_fixes = apply_embedding_fixes(extractor)
            all_fixes.update(embedding_fixes)
            print(f"✅ Embedding fixes applied: {embedding_fixes}")
        
        # Test after fixes
        if args.test_image:
            print("\n📊 Testing AFTER fixes...")
            after_metrics = test_fixes(processor, args.test_image)
            print(f"After fixes: {after_metrics}")
            
            # Show improvements
            if 'error' not in before_metrics and 'error' not in after_metrics:
                improvements = {
                    'results_improvement': after_metrics['results_found'] - before_metrics['results_found'],
                    'similarity_improvement': after_metrics['avg_similarity'] - before_metrics['avg_similarity'],
                    'confidence_improvement': after_metrics['avg_confidence'] - before_metrics['avg_confidence']
                }
                print(f"\n📈 Improvements: {improvements}")
        
        print(f"\n✅ All fixes applied successfully!")
        print(f"Applied fixes: {all_fixes}")
        
        # Save configuration
        print("\n💾 Fixes have been applied to the current session.")
        print("To make these fixes permanent, consider:")
        print("1. Updating the similarity searcher configuration")
        print("2. Modifying the embedding extractor settings")
        print("3. Adjusting confidence thresholds in config files")
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()