#!/usr/bin/env python3
"""
Integration test for the ChromaVectorStore.

This script tests the core vector database functionality.
"""

import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.vector_store import ChromaVectorStore
from src.models.data_models import Embedding

def test_chroma_vector_store():
    """Test the ChromaVectorStore functionality."""
    print("🚀 Starting ChromaVectorStore integration test...")
    
    # 1. Initialize database
    print("\n1. Initializing database...")
    db_path = f"./data/test_chroma_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {
        'db_path': db_path,
        'collection_name': 'test_embeddings',
    }
    
    vector_store = ChromaVectorStore(db_path=db_path, collection_name='test_embeddings')
    init_success = vector_store.initialize_database(config)
    
    if not init_success:
        print("❌ Database initialization failed")
        return False
    
    print("✅ Database initialized successfully")
    
    # 2. Create test embeddings
    print("\n2. Creating test embeddings...")
    test_embeddings = []
    embedding_dim = 128
    
    # Create diverse test vectors for military object classes
    military_objects = [
        "M1_Abrams_Tank", "F16_Fighter", "Apache_Helicopter", 
        "Humvee_Vehicle", "Patriot_Missile"
    ]
    
    for i, obj_class in enumerate(military_objects):
        # Create a characteristic vector for each object type
        vector = np.random.rand(embedding_dim).astype(np.float32)
        vector[i] = 1.0  # Make each vector distinctive
        
        embedding = Embedding(
            id=f"emb_{i:03d}",
            vector=vector,
            image_id=f"img_{i:03d}",
            model_version="test_v1.0",
            extraction_timestamp=datetime.now()
        )
        test_embeddings.append(embedding)
    
    print(f"✅ Created {len(test_embeddings)} test embeddings")
    
    # 3. Store embeddings
    print("\n3. Storing embeddings...")
    
    # Store individual embeddings
    for i, embedding in enumerate(test_embeddings[:3]):
        success = vector_store.store_embedding(embedding)
        if not success:
            print(f"❌ Failed to store embedding {embedding.id}")
            return False
        print(f"   Stored embedding {embedding.id}")
    
    # Store batch embeddings
    batch_success = vector_store.store_embeddings_batch(test_embeddings[3:])
    if not batch_success:
        print("❌ Failed to store batch embeddings")
        return False
    
    print("✅ All embeddings stored successfully")
    
    # 4. Check database stats
    print("\n4. Checking database statistics...")
    stats = vector_store.get_database_stats()
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Count: {stats['collection_count']}")
    print(f"   Status: {stats['status']}")
    print(f"   Path: {stats['db_path']}")
    
    if stats['collection_count'] != len(test_embeddings):
        print(f"❌ Expected {len(test_embeddings)} embeddings, found {stats['collection_count']}")
        return False
    
    print("✅ Database statistics correct")
    
    # 5. Test retrieval
    print("\n5. Testing embedding retrieval...")
    for i, test_embedding in enumerate(test_embeddings[:3]):
        test_id = test_embedding.id
        retrieved = vector_store.get_embedding(test_id)
        
        if not retrieved:
            print(f"❌ Failed to retrieve embedding {test_id}")
            return False
        
        if not np.allclose(retrieved.vector, test_embedding.vector, rtol=1e-5, atol=1e-6):
            print(f"❌ Retrieved vector doesn't match original for {test_id}")
            print(f"   Max difference: {np.max(np.abs(retrieved.vector - test_embedding.vector))}")
            return False
        
        print(f"   ✅ Retrieved {test_id}: {retrieved.image_id}")
    
    print("✅ Embedding retrieval successful")
    
    # 6. Test similarity search
    print("\n6. Testing similarity search...")
    query_vector = test_embeddings[0].vector.copy()
    
    # Test search with various k values
    for k in [1, 3, 5]:
        search_results = vector_store.search_similar(query_vector, k=k)
        
        if not search_results:
            print(f"❌ No search results returned for k={k}")
            return False
        
        print(f"   k={k}: {len(search_results)} results")
        
        # Check if we got the expected number of results
        expected_results = min(k, len(test_embeddings))
        if len(search_results) != expected_results:
            print(f"❌ Expected {expected_results} results, got {len(search_results)}")
            return False
        
        # Check if the first result is the exact match (highest similarity)
        first_result = search_results[0]
        if first_result.similarity_score < 0.99:
            print(f"❌ First result similarity too low: {first_result.similarity_score}")
            return False
        
        print(f"      Best similarity: {first_result.similarity_score:.4f}")
        print(f"      Best match image: {first_result.image_id}")
    
    print("✅ Similarity search successful")
    
    # 7. Test create_index method
    print("\n7. Testing index creation...")
    index_success = vector_store.create_index("hnsw")
    if index_success:
        print("✅ Index creation successful")
    else:
        print("❌ Index creation failed")
        return False
    
    # 8. Test deletion
    print("\n8. Testing embedding deletion...")
    delete_id = test_embeddings[-1].id
    delete_success = vector_store.delete_embedding(delete_id)
    
    if delete_success:
        print(f"✅ Successfully deleted embedding {delete_id}")
        
        # Verify deletion
        deleted_embedding = vector_store.get_embedding(delete_id)
        if deleted_embedding is None:
            print("✅ Deletion verified")
        else:
            print("❌ Embedding still exists after deletion")
            return False
    else:
        print(f"❌ Failed to delete embedding {delete_id}")
        return False
    
    # 9. Final statistics
    print("\n9. Final database state...")
    final_stats = vector_store.get_database_stats()
    expected_count = len(test_embeddings) - 1  # After deletion
    
    if final_stats['collection_count'] == expected_count:
        print(f"✅ Final count correct: {final_stats['collection_count']}")
    else:
        print(f"❌ Final count incorrect: expected {expected_count}, got {final_stats['collection_count']}")
        return False
    
    # 10. Test edge cases
    print("\n10. Testing edge cases...")
    
    # Test search with empty database result
    random_vector = np.random.rand(embedding_dim).astype(np.float32) * 10  # Very different vector
    edge_results = vector_store.search_similar(random_vector, k=10)
    print(f"   Search with random vector: {len(edge_results)} results")
    
    # Test invalid embedding ID
    invalid_retrieval = vector_store.get_embedding("non_existent_id")
    if invalid_retrieval is None:
        print("✅ Correctly handled invalid embedding ID")
    else:
        print("❌ Should return None for invalid ID")
        return False
    
    # Cleanup
    vector_store.close()
    
    print("\n🎉 All tests passed! ChromaVectorStore is working correctly.")
    print(f"\nTest database location: {db_path}")
    print("The vector database system is ready for IR image classification tasks.")
    
    return True

if __name__ == "__main__":
    try:
        success = test_chroma_vector_store()
        if success:
            print("\n✅ ChromaVectorStore integration test completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ ChromaVectorStore integration test failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Integration test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
