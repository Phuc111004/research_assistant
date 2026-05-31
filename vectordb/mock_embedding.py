import numpy as np
from typing import List, Union

class MockEmbeddingClient:
    """
    Mock embedding client that returns random embeddings for testing
    when the real embedding service is unavailable
    """
    def __init__(self, vector_size=384):
        self.vector_size = vector_size
        print(f"Using Mock Embedding client with vector size {vector_size}")
        
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate random embeddings of the correct dimension
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Return empty array for empty input
        if not texts:
            return np.array([])
        
        # Generate a random vector for each text
        embeddings = np.random.rand(len(texts), self.vector_size)
        
        # Normalize the vectors
        for i in range(len(embeddings)):
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] = embeddings[i] / norm
                
        # If we sent a single text, return a single embedding
        if len(texts) == 1:
            return embeddings[0]
            
        return embeddings
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return the fixed embedding dimension"""
        return self.vector_size
    
    def get_max_seq_length(self) -> int:
        """Return a reasonable max sequence length"""
        return 4096 