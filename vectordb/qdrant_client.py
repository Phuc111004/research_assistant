import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from api.config import settings
from api.embedding_client import EmbeddingClient
from vectordb.mock_embedding import MockEmbeddingClient

class QdrantVectorDB:
    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = None,
        embedding_api_url: str = None,
        embedding_api_key: str = None,
        host: str = None,
        port: int = None
    ):
        # Initialize Qdrant client. api_key is required when the Qdrant server
        # has QDRANT__SERVICE__API_KEY set; left empty for unauthenticated servers.
        self.client = QdrantClient(
            host=host or settings.qdrant_host,
            port=port or settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            https=False,
        )
        
        self.collection_name = collection_name or settings.qdrant_collection
        
        # Initialize embedding client with fallback
        model_name = embedding_model or settings.embedding_model
        api_url = embedding_api_url or settings.embedding_api_url
        api_key = embedding_api_key or settings.embedding_api_key
        
        print(f"Initializing embedding client for model: {model_name}")
        print(f"Embedding API URL: {api_url}")
        
        try:
            self.embedder = EmbeddingClient(
                api_url=api_url,
                api_key=api_key,
                model=model_name
            )
            self.vector_size = self.embedder.get_sentence_embedding_dimension()
            print(f"Successfully connected to embedding service")
            print(f"Embedding vector dimension: {self.vector_size}")
            print(f"Model max sequence length: {self.embedder.get_max_seq_length()}")
        except Exception as e:
            print(f"Error initializing embedding client for {model_name}: {e}")
            print(f"Using mock embedding client as fallback")
            # Use mock embedding client with default dimension of 384
            self.embedder = MockEmbeddingClient(vector_size=384)
            self.vector_size = self.embedder.get_sentence_embedding_dimension()
            print(f"Mock embedding client initialized with vector size: {self.vector_size}")
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()

    # Payload indexes used for fast filtering. Field name -> Qdrant schema type.
    # KEYWORD: exact-match string. INTEGER: range/eq numeric. DATETIME: ISO timestamps.
    PAYLOAD_INDEXES = {
        "user_id": models.PayloadSchemaType.KEYWORD,
        "keywords": models.PayloadSchemaType.KEYWORD,
        "doi": models.PayloadSchemaType.KEYWORD,
        "conference_name": models.PayloadSchemaType.KEYWORD,
        "journal_name": models.PayloadSchemaType.KEYWORD,
        "file_format": models.PayloadSchemaType.KEYWORD,
        "publication_year": models.PayloadSchemaType.INTEGER,
        "citations_count": models.PayloadSchemaType.INTEGER,
    }

    def _create_collection_if_not_exists(self):
        """Create collection if missing, then ensure all payload indexes exist."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            print(f"Creating new collection: {self.collection_name} with vector size {self.vector_size}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
        else:
            print(f"Using existing collection: {self.collection_name}")
            collection_info = self.client.get_collection(self.collection_name)
            existing_vector_size = collection_info.config.params.vectors.size
            if existing_vector_size != self.vector_size:
                print(f"Warning: Collection vector size ({existing_vector_size}) does not match current model vector size ({self.vector_size})")
                self.vector_size = existing_vector_size
                if isinstance(self.embedder, MockEmbeddingClient):
                    self.embedder = MockEmbeddingClient(vector_size=existing_vector_size)
                    print(f"Updated mock embedding client to match collection vector size: {existing_vector_size}")

        # Ensure all payload indexes exist (idempotent — qdrant returns ok if already present).
        for field_name, schema in self.PAYLOAD_INDEXES.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as e:
                # Already-exists errors are fine; log others.
                if "already exists" not in str(e).lower():
                    print(f"Could not create payload index '{field_name}': {e}")

    def add_paper(
        self, 
        paper_id: str,
        title: str, 
        abstract: str, 
        keywords: List[str], 
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a paper to the vector database
        
        Args:
            paper_id: unique identifier for the paper
            title: paper title
            abstract: paper abstract
            keywords: list of keywords
            user_id: user who added/owns the paper
            metadata: additional metadata (e.g., conference, journal)
            
        Returns:
            bool: True if successful
        """
        # Include keywords in the embedded text so terminology-heavy queries
        # (e.g. "graph neural network molecule") match papers whose abstract
        # uses paraphrases but whose keyword list contains the exact term.
        keywords_str = ", ".join(keywords) if keywords else ""
        parts = [p for p in [title, abstract, keywords_str] if p]
        text_to_embed = "\n".join(parts)

        # Get embedding vector
        embedding = self.embedder.encode(text_to_embed)

        # Prepare payload
        payload = {
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "user_id": user_id
        }

        # Add metadata if provided
        if metadata and isinstance(metadata, dict):
            for key, value in metadata.items():
                if value is None or value == "":
                    continue
                if key not in payload:  # Avoid overwriting existing fields
                    payload[key] = value
        
        # Add the point to the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=paper_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            ]
        )
        
        return True

    def search(
        self, 
        query: str, 
        limit: int = 3, 
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for papers similar to the query
        
        Args:
            query: search query
            limit: number of results to return (minimum 3)
            user_id: filter results by user_id (optional)
            
        Returns:
            List of matching papers with similarity scores
        """
        # Ensure at least 3 results will be returned
        actual_limit = max(limit, 3)
        
        # Get query embedding
        query_vector = self.embedder.encode(query).tolist()
        
        # Set up filter if user_id is provided
        filter_param = None
        if user_id:
            filter_param = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )
        
        # Search the collection. qdrant-client >=1.10 removed .search() in favor
        # of .query_points(); the returned object exposes .points with the same
        # ScoredPoint shape (id, score, payload).
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=actual_limit,
            query_filter=filter_param,
            with_payload=True,
        ).points
        
        # Format results
        results = []
        for result in search_results:
            # Extract all fields from payload
            paper_data = {
                "paper_id": result.id,
                "score": result.score,
                "title": result.payload.get("title", ""),
                "abstract": result.payload.get("abstract", ""),
                "keywords": result.payload.get("keywords", []),
                "user_id": result.payload.get("user_id", "")
            }
            
            # Ensure keywords is always an array
            if not isinstance(paper_data["keywords"], list):
                if isinstance(paper_data["keywords"], str):
                    paper_data["keywords"] = [k.strip() for k in paper_data["keywords"].split(",")]
                else:
                    paper_data["keywords"] = []
            
            # Include any additional metadata fields
            for key, value in result.payload.items():
                if key not in paper_data:
                    paper_data[key] = value
            
            results.append(paper_data)
            
        return results

    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from the database"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[paper_id]
            )
        )
        return True 