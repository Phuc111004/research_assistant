import os
import requests
import json
from typing import List, Dict, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import settings
from .citations import default_citation_system_prompt

class VLLMClient:
    def __init__(self, api_url: str = None, api_key: str = None, model: str = None):
        """
        Initialize VLLM client
        
        Args:
            api_url: VLLM API URL, defaults to environment variable
            api_key: VLLM API key, defaults to environment variable
            model: VLLM model to use
        """
        self.api_url = api_url or settings.vllm_api_url
        self.api_key = api_key or settings.vllm_api_key
        self.model = model or settings.vllm_model
        
        # Update headers with the correct authorization format
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Some VLLM deployments also accept the key as a separate header
        if self.api_key:
            self.headers["api-key"] = self.api_key
        
        # Test connection
        self.api_available = self._test_connection()
        if not self.api_available:
            print(f"Warning: Could not connect to VLLM API at {self.api_url}. Fallback mode will be used.")
            print(f"This is expected if you don't have a local LLM service running.")
            # This is not an error condition - just using the fallback
    
    def _test_connection(self) -> bool:
        """Test connection to VLLM API"""
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
            print(f"Error connecting to VLLM API: {e}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_response(
        self, 
        query: str, 
        papers: List[Dict[str, Any]], 
        system_prompt: str = None
    ) -> Tuple[str, bool]:
        """
        Generate a response based on the query and retrieved papers
        
        Args:
            query: User's research question
            papers: List of papers retrieved from the vector database
            system_prompt: Optional system prompt to override default
            
        Returns:
            Tuple[str, bool]: (Generated response text, using_fallback flag)
        """
        # Check if there are any papers
        if not papers:
            return "I couldn't find any relevant papers for your query.", False
            
        # Check if API is available
        if not self.api_available:
            # Return a fallback response using the papers directly
            return self._generate_fallback_response(query, papers), True
            
        if not system_prompt:
            system_prompt = default_citation_system_prompt()
        
        # Format papers as context
        paper_context = ""
        for i, paper in enumerate(papers, 1):
            paper_context += f"\nPaper {i}:\n"
            paper_context += f"Title: {paper['title']}\n"
            paper_context += f"Abstract: {paper['abstract']}\n"
            if paper.get('keywords'):
                paper_context += f"Keywords: {', '.join(paper['keywords'])}\n"
            paper_context += f"Relevance Score: {paper['score']:.2f}\n"
            
        # Construct the full prompt
        user_message = f"Query: {query}\n\nRelevant Papers:{paper_context}"
        
        try:
            # Call VLLM API (OpenAI compatible endpoint)
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"VLLM response: {result}")
                
                # Check if the response has the expected structure
                if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                    return result["choices"][0]["message"]["content"], False
                else:
                    print(f"Unexpected VLLM response structure: {result}")
                    return self._generate_fallback_response(query, papers), True
            else:
                print(f"Error from VLLM API: {response.status_code} - {response.text}")
                return self._generate_fallback_response(query, papers), True
                
        except Exception as e:
            print(f"Error calling VLLM API: {e}")
            return self._generate_fallback_response(query, papers), True
    
    def _generate_fallback_response(self, query: str, papers: List[Dict[str, Any]]) -> str:
        """
        Generate a simple fallback response when VLLM API is not available
        
        Args:
            query: User's research question
            papers: List of papers retrieved from the vector database
            
        Returns:
            Simple response based on paper data
        """
        # Create a more useful summarized response based on top papers
        if not papers:
            return "I couldn't find any relevant papers for your query."
            
        # Get the top 3 most relevant papers
        top_papers = sorted(papers, key=lambda p: p.get('score', 0), reverse=True)[:3]
        
        # Extract key information from the papers
        response = f"Here are some relevant papers for your query: '{query}':\n\n"
        
        for i, paper in enumerate(top_papers, 1):
            title = paper.get('title', 'Untitled Paper')
            abstract = paper.get('abstract', '')
            snippet = abstract[:200] + ("..." if len(abstract) > 200 else "")
            keywords = paper.get('keywords', [])

            response += f"• {title} [{i}]\n"
            if snippet:
                response += f"  {snippet}\n"
            if keywords and len(keywords) > 0:
                response += f"  Keywords: {', '.join(keywords[:5])}\n"
            response += "\n"

        response += (
            "\nNote: LLM is unavailable. Hover citation markers [n] or open Sources "
            "for full paper details."
        )
        
        return response 