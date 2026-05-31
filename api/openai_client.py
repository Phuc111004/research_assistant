import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import settings

class OpenAIClient:
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key, defaults to environment variable
            model: OpenAI model to use
        """
        if api_key is None:
            api_key = settings.openai_api_key
        
        self.api_key_valid = True
        if not api_key:
            self.api_key_valid = False
            print("Warning: No OpenAI API key provided. The system will use fallback mode.")
            api_key = "dummy_key"  # Placeholder key
            
        try:
            self.client = OpenAI(api_key=api_key)
            self.model = model or settings.openai_model
        except Exception as e:
            self.api_key_valid = False
            print(f"Error initializing OpenAI client: {e}")
            self.client = None
            self.model = None
    
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
            
        # Check if API key is valid
        if not self.api_key_valid or self.client is None:
            # Return a fallback response using the papers directly
            return self._generate_fallback_response(query, papers), True
            
        if not system_prompt:
            system_prompt = (
                "You are a helpful research assistant that provides accurate information based on scientific papers. "
                "Answer the user's query using the provided research papers as references. "
                "If the papers don't contain relevant information to answer the query, state this clearly. "
                "Always cite the paper titles when referring to specific information from them."
            )
        
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
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content, False
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._generate_fallback_response(query, papers), True
    
    def _generate_fallback_response(self, query: str, papers: List[Dict[str, Any]]) -> str:
        """
        Generate a simple fallback response when OpenAI API is not available
        
        Args:
            query: User's research question
            papers: List of papers retrieved from the vector database
            
        Returns:
            Simple response based on paper data
        """
        response = f"Here are some relevant papers for your query: '{query}'\n\n"
        
        for i, paper in enumerate(papers, 1):
            response += f"Paper {i}: {paper['title']}\n"
            response += f"Relevance Score: {paper['score']:.2f}\n"
            response += f"Abstract: {paper['abstract'][:200]}...\n\n"
            
        response += "Note: OpenAI API is currently unavailable. This is a basic response showing the most relevant papers."
        return response 