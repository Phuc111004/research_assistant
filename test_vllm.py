#!/usr/bin/env python3
"""
Test script to check if the deployed VLLM model is working correctly.
"""

import sys
from api.vllm_client import VLLMClient
from api.config import settings

def test_vllm_model():
    print(f"Testing connection to VLLM model at {settings.vllm_api_url}")
    print(f"Model: {settings.vllm_model}")
    
    try:
        # Initialize the VLLM client
        client = VLLMClient()
        
        # Test with simple papers
        test_papers = [
            {
                "paper_id": "test1",
                "title": "Understanding Machine Learning Fundamentals",
                "abstract": "This paper provides an overview of fundamental concepts in machine learning, including supervised learning, unsupervised learning, and reinforcement learning.",
                "keywords": ["machine learning", "artificial intelligence", "supervised learning"],
                "score": 0.95,
                "user_id": "test_user"
            },
            {
                "paper_id": "test2",
                "title": "Advances in Natural Language Processing",
                "abstract": "We review recent advances in natural language processing, focusing on transformer-based models and their applications to various tasks such as text generation, summarization, and translation.",
                "keywords": ["NLP", "transformers", "language models"],
                "score": 0.85,
                "user_id": "test_user"
            }
        ]
        
        # Test simple prompt
        test_query = "What are the key concepts in machine learning?"
        
        print("\nGenerating response for test query...")
        response, fallback = client.generate_response(
            query=test_query,
            papers=test_papers
        )
        
        print(f"\nQuery: {test_query}")
        print(f"Fallback used: {fallback}")
        print(f"Response: {response[:500]}..." if len(response) > 500 else f"Response: {response}")
        
        if not fallback:
            print("\n✅ VLLM model is working correctly!")
            return True
        else:
            print("\n⚠️ VLLM model is not available, fallback was used!")
            return False
        
    except Exception as e:
        print(f"\n❌ Error testing VLLM model: {e}")
        return False

if __name__ == "__main__":
    success = test_vllm_model()
    sys.exit(0 if success else 1) 