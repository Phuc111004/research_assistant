#!/usr/bin/env python3
"""
Test script to check if the deployed embedding model is working correctly.
"""

import sys
import numpy as np
from api.embedding_client import EmbeddingClient
from api.config import settings

def test_embedding_model():
    print(f"Testing connection to embedding model at {settings.embedding_api_url}")
    print(f"Model: {settings.embedding_model}")
    
    try:
        # Initialize the embedding client
        client = EmbeddingClient()
        
        # Test simple embedding
        test_text = "This is a test sentence to check if the embedding model is working correctly."
        
        print("\nGenerating embedding for test text...")
        embedding = client.encode(test_text)
        
        # Check the embedding shape
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dimension: {len(embedding)}")
        
        # Check if embedding values look reasonable
        print(f"Embedding min value: {np.min(embedding)}")
        print(f"Embedding max value: {np.max(embedding)}")
        print(f"Embedding mean value: {np.mean(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Test batch embedding
        test_batch = [
            "First test sentence for batch embedding.",
            "Second test sentence with different content.",
            "Third test sentence to verify batch processing."
        ]
        
        print("\nGenerating batch embeddings...")
        batch_embeddings = client.encode(test_batch)
        
        print(f"Batch embeddings shape: {batch_embeddings.shape}")
        print(f"Number of embeddings: {len(batch_embeddings)}")
        
        # Test similarity between sentences
        if len(batch_embeddings) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            
            print("\nCalculating cosine similarities between sentences:")
            similarity_matrix = cosine_similarity(batch_embeddings)
            
            for i in range(len(test_batch)):
                for j in range(i + 1, len(test_batch)):
                    print(f"Similarity between sentence {i+1} and {j+1}: {similarity_matrix[i][j]:.4f}")
        
        print("\n✅ Embedding model is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing embedding model: {e}")
        return False

if __name__ == "__main__":
    success = test_embedding_model()
    sys.exit(0 if success else 1) 