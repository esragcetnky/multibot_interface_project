# ==============================================================================
# SECTION 1: Imports and Setup
# This section imports required modules and initializes the FastAPI app.
# ==============================================================================
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import os
import shutil
from typing import List
from components.faiss_db import (
    ensure_folder_exists,
    load_and_split_documents,
    embeddings,
)
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

app = FastAPI()

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
# SECTION 3: Vector DB API Endpoints
# This section provides CRUD endpoints for vector DB management.
# ==============================================================================

@app.delete("/api/vectordb/delete")
def delete_vector_db(req: VectorDBRequest):
    """Delete the entire vector DB folder."""
    if not os.path.exists(req.vector_db_path):
        raise HTTPException(status_code=404, detail="Vector DB not found.")
    shutil.rmtree(req.vector_db_path)
    return {"status": "deleted", "vector_db_path": req.vector_db_path}

@app.post("/api/vectordb/add")
def add_document_to_vector_db(req: AddDocumentRequest):
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
def list_documents_in_vector_db(vector_db_path: str):
    """
    List all document names in the vector DB (based on chunk_index metadata).
    """
    chunk_index_path = os.path.join(vector_db_path, "chunk_index")
    if not os.path.exists(os.path.join(chunk_index_path, "index.faiss")):
        raise HTTPException(status_code=404, detail="Vector DB not found.")
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
    return {"documents": list(doc_names)}

@app.delete("/api/vectordb/delete-document")
def delete_document_from_vector_db(req: DeleteDocumentRequest):
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
# SECTION 4: Additional CRUD Endpoints
# You can add more endpoints for advanced vector DB management