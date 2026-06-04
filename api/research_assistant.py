import os
from typing import List, Dict, Any, Optional, Tuple
from api.vllm_client import VLLMClient
from api.citations import build_citations, default_citation_system_prompt
from vectordb.qdrant_client import QdrantVectorDB

class ResearchAssistant:
    def __init__(
        self,
        vector_db: Optional[QdrantVectorDB] = None,
        vllm_client: Optional[VLLMClient] = None,
        top_k: int = 3
    ):
        """
        Initialize the research assistant
        
        Args:
            vector_db: Vector database client
            vllm_client: VLLM client
            top_k: Number of papers to retrieve (minimum 3)
        """
        self.vector_db = vector_db
        self.vllm_client = vllm_client
        # Ensure top_k is at least 3
        self.top_k = max(top_k, 3)
        
        # Check if core components are available
        if self.vector_db is None:
            print("WARNING: Vector database is not available. Search functionality will be limited.")
    
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
        if self.vector_db is None:
            print("Cannot add paper: Vector database is not available")
            return False
            
        return self.vector_db.add_paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            keywords=keywords,
            user_id=user_id,
            metadata=metadata
        )
    
    def query(
        self,
        query_text: str,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a research query
        
        Args:
            query_text: User's research question
            user_id: Optional user ID to filter results
            system_prompt: Optional system prompt for VLLM
            
        Returns:
            Dictionary with answer, retrieved papers, and fallback status
        """
        # Handle case where vector_db is not available
        if self.vector_db is None:
            return {
                "query": query_text,
                "answer": "I'm sorry, but the vector database is currently unavailable. Please try again later.",
                "papers": [],
                "citations": [],
                "using_fallback": True,
            }
            
        # Retrieve relevant papers (ensure at least 3)
        papers = self.vector_db.search(
            query=query_text,
            limit=self.top_k,
            user_id=user_id
        )
        
        # Ensure keywords are properly formatted as array
        for paper in papers:
            if "keywords" in paper and paper["keywords"] is not None:
                # Make sure keywords is always a list
                if isinstance(paper["keywords"], str):
                    paper["keywords"] = [k.strip() for k in paper["keywords"].split(",")]
                elif not isinstance(paper["keywords"], list):
                    paper["keywords"] = []
            else:
                paper["keywords"] = []
        
        # Check if VLLM client is available
        if self.vllm_client is None:
            # Use a fallback response if no VLLM client
            from api.vllm_client import VLLMClient
            temp_client = VLLMClient()
            answer = temp_client._generate_fallback_response(query_text, papers)
            using_fallback = True
        else:
            prompt = system_prompt or default_citation_system_prompt()
            answer, using_fallback = self.vllm_client.generate_response(
                query=query_text,
                papers=papers,
                system_prompt=prompt,
            )

        citations = build_citations(papers)

        return {
            "query": query_text,
            "answer": answer,
            "papers": papers,
            "citations": citations,
            "using_fallback": using_fallback,
        }

    def search_papers(
        self,
        query_text: str,
        limit: int = 20,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Vector search only (no LLM). Used by backend recommendations."""
        if self.vector_db is None:
            return []

        papers = self.vector_db.search(
            query=query_text,
            limit=max(limit, 1),
            user_id=user_id,
        )

        for paper in papers:
            if "keywords" in paper and paper["keywords"] is not None:
                if isinstance(paper["keywords"], str):
                    paper["keywords"] = [
                        k.strip() for k in paper["keywords"].split(",")
                    ]
                elif not isinstance(paper["keywords"], list):
                    paper["keywords"] = []
            else:
                paper["keywords"] = []

        return papers
    
    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from the database"""
        if self.vector_db is None:
            print("Cannot delete paper: Vector database is not available")
            return False
            
        return self.vector_db.delete_paper(paper_id) 