# Research Assistant with VLLM and Custom Embedding Models

This project is a research assistant application that uses:
- VLLM for serving language models
- Custom embedding models for semantic search
- Qdrant for vector storage

## Architecture

The system connects to the following remote services:
- **VLLM API**: Hosted at `http://172.16.0.116:8082/v1`
- **Embedding API**: Hosted at `http://172.16.0.116:8090/v1`
- **Reranker API**: Hosted at `http://172.16.0.116:8091/v1`

## Installation and Setup

### Prerequisites
- Python 3.8+
- Pip package manager

### Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in your .env file or export them directly:
```bash
export VLLM_API_URL="http://172.16.0.116:8082/v1"
export VLLM_API_KEY="hf-Abc13579@"
export VLLM_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

export EMBEDDING_API_URL="http://172.16.0.116:8090/v1"
export EMBEDDING_API_KEY="hf-Abc13579@"
export EMBEDDING_MODEL="intfloat/e5-mistral-7b-instruct"

export RERANKER_API_URL="http://172.16.0.116:8091/v1"
export RERANKER_API_KEY="hf-Abc13579@"
export RERANKER_MODEL="rerank"
```

### Running Tests

Before running the full server, you can test if the remote services are working:

```bash
# Test the embedding model
python test_embedding.py

# Test the VLLM model
python test_vllm.py
```

### Running the Application

Use the provided script to run the application:

```bash
chmod +x run_server.sh
./run_server.sh
```

Or run it manually:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
cd api && python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Query the Research Assistant
```
POST /api/query
```

Request body:
```json
{
  "query": "What are the key concepts in machine learning?",
  "user_id": "user123"  // Optional, to filter papers by user
}
```

### Add a Paper
```
POST /api/papers
```

Request body:
```json
{
  "paper_id": "paper123",
  "title": "Understanding Machine Learning",
  "abstract": "This paper provides an overview...",
  "keywords": ["machine learning", "AI"],
  "user_id": "user123"
}
```

### Delete a Paper
```
DELETE /api/papers/{paper_id}
```

## Fallback Behavior

If the VLLM or embedding services are not available, the system will fall back to:
- For VLLM: Return a simple summary of the retrieved papers
- For embeddings: Will raise an error as no fallback embedding method is implemented

## Troubleshooting

1. "Unauthorized" errors: Make sure the API keys are correctly configured.

2. Connection errors: Check network connectivity to the remote services.

3. If both embedding and VLLM models fail, ensure the servers are running and accessible.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| VLLM_API_URL | URL for the VLLM service | http://172.16.0.116:8082/v1 |
| VLLM_API_KEY | API key for the VLLM service | hf-Abc13579@ |
| EMBEDDING_API_URL | URL for the embedding service | http://172.16.0.116:8090/v1 |
| EMBEDDING_API_KEY | API key for the embedding service | hf-Abc13579@ |
| QDRANT_HOST | Hostname for Qdrant vector DB | localhost |
| QDRANT_PORT | Port for Qdrant vector DB | 6333 |
| TOP_K | Number of papers to retrieve | 3 | 