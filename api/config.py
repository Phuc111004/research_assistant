import os
from pydantic import validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    # API settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # LLM API settings (replacing OpenAI)
    vllm_api_url: str = os.getenv("VLLM_API_URL", "http://172.16.0.115:8082/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "hf-Abc13579@")
    
    # Embedding API settings
    embedding_api_url: str = os.getenv("EMBEDDING_API_URL", "https://api.openai.com/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    # Embedding fallback (used when the primary endpoint above is unreachable).
    embedding_fallback_api_url: str = os.getenv("EMBEDDING_FALLBACK_API_URL", "")
    embedding_fallback_api_key: str = os.getenv("EMBEDDING_FALLBACK_API_KEY", "")
    embedding_fallback_model: str = os.getenv("EMBEDDING_FALLBACK_MODEL", "")

    # Reranker API settings
    reranker_api_url: str = os.getenv("RERANKER_API_URL", "http://172.16.0.116:8091/v1")
    reranker_model: str = os.getenv("RERANKER_MODEL", "rerank")
    reranker_api_key: str = os.getenv("RERANKER_API_KEY", "hf-Abc13579@")
    
    # Qdrant settings
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "research_papers")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    
    # Research Assistant settings
    top_k: int = int(os.getenv("TOP_K", "3"))
    
    @validator('top_k')
    def ensure_minimum_top_k(cls, v):
        """Ensure top_k is at least 3"""
        if v < 3:
            print(f"Warning: TOP_K value ({v}) is less than minimum (3). Setting to 3.")
            return 3
        return v
    
    def log_config(self):
        """Log the current configuration (excluding sensitive values)"""
        print("\n--- Research Assistant Configuration ---")
        print(f"Server: {self.host}:{self.port}")
        print(f"VLLM Model: {self.vllm_model}")
        print(f"VLLM API URL: {self.vllm_api_url}")
        print(f"Embedding Model: {self.embedding_model}")
        print(f"Embedding API URL: {self.embedding_api_url}")
        print(f"Reranker Model: {self.reranker_model}")
        print(f"Reranker API URL: {self.reranker_api_url}")
        print(f"Qdrant: {self.qdrant_host}:{self.qdrant_port}")
        print(f"Qdrant Collection: {self.qdrant_collection}")
        print(f"Top K Results: {self.top_k}")
        print("---------------------------------------\n")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Initialize and log settings
settings = Settings()
settings.log_config() 