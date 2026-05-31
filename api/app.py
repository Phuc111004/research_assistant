import os
import traceback
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from api.research_assistant import ResearchAssistant
from api.config import settings
from vectordb.qdrant_client import QdrantVectorDB
from api.vllm_client import VLLMClient
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize vector_db with error handling
try:
    vector_db = QdrantVectorDB()
    print("Vector database initialized successfully")
except Exception as e:
    print(f"Error initializing vector database: {e}")
    print("Research Assistant will not be able to store or retrieve papers")
    vector_db = None

# Initialize VLLM client with graceful fallback
try:
    vllm_client = VLLMClient()
    print("VLLM client initialized successfully")
except Exception as e:
    print(f"Error initializing VLLM client: {e}")
    print("Research Assistant will run in fallback mode without VLLM")
    vllm_client = None

# Ensure top_k is at least 3
top_k = max(settings.top_k, 3)
print(f"Research Assistant will fetch at least {top_k} papers per query")

# Initialize the Research Assistant
try:
    assistant = ResearchAssistant(
        vector_db=vector_db,
        vllm_client=vllm_client,
        top_k=top_k
    )
    print("Research Assistant initialized successfully")
except Exception as e:
    print(f"Error initializing Research Assistant: {e}")
    raise RuntimeError(f"Failed to initialize Research Assistant: {str(e)}")

# Create FastAPI app
app = FastAPI(
    title="Research Assistant API",
    description="API for querying research papers and getting AI-generated answers",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define whether the app starts in fallback mode
FALLBACK_MODE = vllm_client is None
if FALLBACK_MODE:
    print("Starting Research Assistant in fallback mode with enhanced summaries")
else:
    print("Starting Research Assistant with full LLM capabilities")

# Request/Response Models
class PaperSchema(BaseModel):
    paper_id: str
    title: str
    abstract: str
    keywords: List[str]
    user_id: str
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    papers: List[Dict[str, Any]]
    using_fallback: bool = Field(False, description="Whether fallback mode was used (no VLLM API)")

# Routes
@app.post("/papers", status_code=201)
async def add_paper(paper: PaperSchema):
    """Add a paper to the database"""
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database is not available")
    
    result = assistant.add_paper(
        paper_id=paper.paper_id,
        title=paper.title,
        abstract=paper.abstract,
        keywords=paper.keywords,
        user_id=paper.user_id,
        metadata=paper.metadata
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to add paper")
    
    return {"message": "Paper added successfully", "paper_id": paper.paper_id}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, req: Request):
    """Query the research assistant"""
    if vector_db is None:
        # Return a simple response when vector database is unavailable
        return {
            "query": request.query,
            "answer": "I'm sorry, but the vector database is currently unavailable. Please try again later.",
            "papers": [],
            "using_fallback": True
        }
    
    try:
        result = assistant.query(
            query_text=request.query,
            user_id=request.user_id
        )
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing query: {str(e)}")
        print(f"Error trace: {error_trace}")
        return {
            "query": request.query,
            "answer": f"I encountered an error while processing your query: {str(e)}. The server logs have more details.",
            "papers": [],
            "using_fallback": True
        }

@app.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """Delete a paper from the database"""
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database is not available")
    
    result = assistant.delete_paper(paper_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    return {"message": "Paper deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "vector_db": "available" if vector_db is not None else "unavailable",
        "vllm": "available" if vllm_client is not None else "unavailable",
        "fallback_mode": FALLBACK_MODE
    }
    return status 