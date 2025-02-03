from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST"],
    allow_headers=["*"],
)

# Use a stronger embedding model for better accuracy
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Improved model
)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./vector_db")

# Function to reset collection before adding new documents
def reset_collection():
    global collection
    try:
        client.delete_collection(name="documents")
        print("Collection reset successfully!")
    except Exception as e:
        print(f"Error resetting collection: {e}")
    
    collection = client.create_collection(
        name="documents",
        embedding_function=sentence_transformer_ef
    )
    print("Created new collection: documents")

# Define request schema
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

# Function to add documents to ChromaDB after full reset
def add_documents(docs: List[str]):
    reset_collection()  # Ensure fresh collection
    doc_ids = [str(i) for i in range(len(docs))]
    collection.add(documents=docs, ids=doc_ids)
    print(f"Added {len(docs)} new documents to the database.")

# Function to search for similar documents
def search_similar(query: str, n_results: int = 3):
    print(f"Searching for query: {query}")
    d = collection.query(query_texts=[query], n_results=n_results)
    
    if "documents" not in d or not d["documents"] or not d["documents"][0]:
        print("No relevant matches found!")
        return ["No relevant matches"]
    
    print(f"Found matches: {d['documents'][0]}")  # Debugging
    return [text for text in d["documents"][0]]

# API endpoint to compute document similarity
@app.post("/similarity")
async def compute_similarity(request: SimilarityRequest):
    try:
        # Add documents to ChromaDB
        add_documents(request.docs)

        # Search for similar documents
        top_matches = search_similar(request.query)

        return {"matches": top_matches}
    except Exception as e:
        print(f"Error: {e}")  # Debugging info
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Run the server with: python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
