# ==============================================================================
# SECTION 1: Imports and Setup
# This section imports required modules and initializes the FastAPI app.
# ==============================================================================
from pydantic import BaseModel
import os
import logging
import shutil
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from components.faiss_db import (
    ensure_folder_exists,
    load_and_split_documents,
    embeddings,
)
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# ==============================================================================
# SECTION 2: Logging Configuration
# This section configures logging for the middleware.
# ==============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "vectordb_api.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
    encoding="utf-8"
)


app = FastAPI(title="Vector DB API")

# ==============================================================================
# SECTION 2: Data Models
# This section defines Pydantic models for API requests.
# ==============================================================================

class VectorDBRequest(BaseModel):
    vector_db_path: str

class AddDocumentRequest(BaseModel):
    vector_db_path: str
    document_name: str
    document_path: str

class DeleteDocumentRequest(BaseModel):
    vector_db_path: str
    document_name: str

# ==============================================================================
# SECTION 3: API Root and Health Check
# This section provides basic endpoints for the API.
# ==============================================================================

@app.get("/")
async def root()-> dict:
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the Vector DB API!"}

@app.get("/health")
async def health_check() -> dict:
    logging.info("Health check endpoint accessed.")
    return {"status": "ok", "message": "Vector DB API is running."}

# ==============================================================================
# SECTION 4: Vector DB API Endpoints
# This section provides CRUD endpoints for vector DB management.
# ==============================================================================

@app.delete("/api/vectordb/delete")
def delete_vector_db(req: VectorDBRequest) -> dict:
    """Delete the entire vector DB folder."""
    if not os.path.exists(req.vector_db_path):
        logging.error(f"Vector DB path not found: {req.vector_db_path}")
        raise HTTPException(status_code=404, detail="Vector DB not found.")
    
    shutil.rmtree(req.vector_db_path)
    logging.info(f"Deleted vector DB at path: {req.vector_db_path}")
    return {"status": "deleted", "vector_db_path": req.vector_db_path}

@app.post("/api/vectordb/add")
def add_document_to_vector_db(req: AddDocumentRequest) -> dict:
    """
    Add a new document to the vector DB (incremental, does not re-embed all).
    """
    ensure_folder_exists(req.vector_db_path)
    # Load and split the new document
    docs = load_and_split_documents([req.document_name], [os.path.dirname(req.document_path)])
    if not docs:
        raise HTTPException(status_code=400, detail="No supported documents found.")
    # Add to chunk index
    chunk_index_path = os.path.join(req.vector_db_path, "chunk_index")
    if os.path.exists(os.path.join(chunk_index_path, "index.faiss")):
        chunk_index = FAISS.load_local(
            folder_path=req.vector_db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            index_name="chunk_index"
        )
        chunk_index.add_documents(docs)
        chunk_index.save_local(folder_path=req.vector_db_path, index_name="chunk_index")
    else:
        chunk_index = FAISS.from_documents(documents=docs, embedding=embeddings)
        chunk_index.save_local(folder_path=req.vector_db_path, index_name="chunk_index")
    return {"status": "added", "document_name": req.document_name}

@app.get("/api/vectordb/list")
def list_documents_in_vector_db(vector_db_path: str) -> List[str]:
    """
    List all document names in the vector DB (based on chunk_index metadata).
    """
    if not os.path.exists(os.path.join(vector_db_path, "chunk_index.faiss")):
        logging.info(HTTPException(status_code=404, detail="Vector DB not found."))
        doc_names=[]
    else :
        chunk_index = FAISS.load_local(
            folder_path=vector_db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            index_name="chunk_index"
        )
        # Extract document names from metadata
        doc_names = set()
        for doc in chunk_index.docstore._dict.values():
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                doc_names.add(doc.metadata["source"])
            
    return list(doc_names)

@app.delete("/api/vectordb/delete-document")
def delete_document_from_vector_db(req: DeleteDocumentRequest)-> dict:
    """
    Delete a single document from the vector DB (by document name).
    Only removes matching documents from the index, does not rebuild the whole DB.
    """
    chunk_index_path = os.path.join(req.vector_db_path, "chunk_index")
    if not os.path.exists(os.path.join(chunk_index_path, "index.faiss")):
        raise HTTPException(status_code=404, detail="Vector DB not found.")
    chunk_index = FAISS.load_local(
        folder_path=req.vector_db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        index_name="chunk_index"
    )
    # Find docstore keys to delete
    keys_to_delete = [
        k for k, doc in chunk_index.docstore._dict.items()
        if hasattr(doc, "metadata") and doc.metadata.get("source") == req.document_name
    ]
    if not keys_to_delete:
        raise HTTPException(status_code=404, detail="Document not found in vector DB.")
    for k in keys_to_delete:
        del chunk_index.docstore._dict[k]
    chunk_index.save_local(folder_path=req.vector_db_path, index_name="chunk_index")
    return {"status": "deleted", "document_name": req.document_name}

# ==============================================================================
# SECTION 5: End of Vector DB API
# This section marks the end of the vector DB API endpoints.
# You can add more endpoints for advanced vector DB management 
# ==============================================================================