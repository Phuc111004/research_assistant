import os
import requests
import numpy as np
from typing import List, Union, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import settings

class EmbeddingClient:
    def __init__(self, api_url: str = None, api_key: str = None, model: str = None, target_dim: int = None):
        """
        Initialize embedding client for deployed embedding model

        Args:
            api_url: Embedding API URL, defaults to environment variable
            api_key: Embedding API key, defaults to environment variable
            model: Embedding model to use
            target_dim: Target dimension for embeddings (for compatibility)
        """
        self.api_url = api_url or settings.embedding_api_url
        self.api_key = api_key or settings.embedding_api_key
        self.model = model or settings.embedding_model
        # Default to OpenAI text-embedding-3-small native dim (1536). Override via EMBEDDING_DIM env.
        self.target_dim = target_dim or settings.embedding_dim

        # Optional fallback endpoint (e.g. OpenAI) used when the primary one is unreachable.
        self.fallback_api_url = settings.embedding_fallback_api_url
        self.fallback_api_key = settings.embedding_fallback_api_key
        self.fallback_model = settings.embedding_fallback_model or self.model

        self._set_headers(self.api_key)

        # Test connection; switch to fallback endpoint if the primary one fails.
        self.api_available = self._test_connection()
        if not self.api_available and self.fallback_api_url:
            print(f"Primary embedding API at {self.api_url} unavailable. "
                  f"Switching to fallback: {self.fallback_api_url}")
            self.api_url = self.fallback_api_url
            self.api_key = self.fallback_api_key
            self.model = self.fallback_model
            self._set_headers(self.api_key)
            self.api_available = self._test_connection()

        if not self.api_available:
            print(f"Warning: Could not connect to Embedding API at {self.api_url}. Fallback mode will be used.")

    def _set_headers(self, api_key: str) -> None:
        """Build request headers for the currently selected endpoint."""
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # Some VLLM deployments also accept the key as a separate header
        if api_key:
            self.headers["api-key"] = api_key

    def _test_connection(self) -> bool:
        """Test connection to Embedding API"""
        try:
            # Simple model info request to check connectivity
            response = requests.get(
                f"{self.api_url}/models",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                return True
            else:
                print(f"API connection error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error connecting to Embedding API: {e}")
            return False
    
    def _adjust_dimension(self, embedding: np.ndarray, target_dim: int = None) -> np.ndarray:
        """
        Adjust the embedding dimension to match the target dimension
        
        Args:
            embedding: The original embedding vector
            target_dim: Target dimension (default: self.target_dim)
            
        Returns:
            Adjusted embedding vector
        """
        target = target_dim or self.target_dim
        orig_dim = embedding.shape[0]
        
        if orig_dim == target:
            return embedding
            
        print(f"Adjusting embedding dimension from {orig_dim} to {target}")
        
        if orig_dim > target:
            # Truncate to the target dimension
            return embedding[:target]
        else:
            # Pad with zeros to reach the target dimension
            padded = np.zeros(target)
            padded[:orig_dim] = embedding
            return padded
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the given texts
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            numpy array of embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Return empty array for empty input
        if not texts:
            return np.array([])
            
        try:
            # Call the Embedding API (OpenAI compatible endpoint)
            payload = {
                "model": self.model,
                "input": texts
            }
            
            response = requests.post(
                f"{self.api_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract embeddings in the correct order
                embeddings = [item["embedding"] for item in result["data"]]
                
                # Adjust dimensions if needed
                adjusted_embeddings = []
                for emb in embeddings:
                    adjusted_embeddings.append(self._adjust_dimension(np.array(emb)))
                
                # If we sent a single text, return a single embedding
                if len(texts) == 1:
                    return adjusted_embeddings[0]
                return np.array(adjusted_embeddings)
            else:
                print(f"Error from Embedding API: {response.status_code} - {response.text}")
                raise RuntimeError(f"Embedding API error: {response.status_code}")
                
        except Exception as e:
            print(f"Error calling Embedding API: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    def get_sentence_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the model
        
        Returns:
            Embedding vector dimension (adjusted to target_dim)
        """
        # Return the target dimension for consistency
        return self.target_dim
    
    def get_max_seq_length(self) -> int:
        """
        Get the maximum sequence length of the model
        
        Returns:
            Maximum sequence length
        """
        # This is a best guess based on common values
        return 4096 