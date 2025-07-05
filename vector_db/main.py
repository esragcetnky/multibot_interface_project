# ==============================================================================
# SECTION 1: Imports and Setup
# This section imports required modules and initializes the FastAPI app.
# ==============================================================================
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import yaml
from typing import List
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import OpenAI

from vector_db.utils import (
    ensure_folder_exists,
    load_and_split_documents,
    embed_and_save_documents,
    add_documents_to_vector_db,
    delete_documents_from_vector_db,
    get_all_document_names,
    document_exists_in_vector_db,
    clear_vector_db
)
# ==============================================================================
# SECTION 2: Path and Credential Configuration
# This section defines working directories, data folders, and loads credentials.
# ==============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT,  "shared", "credentials.yml")

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "vectordb_api.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
    encoding="utf-8"
)

def load_credentials():
    """Load API credentials from YAML file."""
    with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
        logging.info(f"Loading credentials from {CREDENTIALS_PATH}")
        return yaml.safe_load(f)

creds = load_credentials()
embeddings = OpenAIEmbeddings(api_key=creds["openai_api_key"], model="text-embedding-3-large")

app = FastAPI(title="Vector DB API")

# ==============================================================================
# SECTION 3: Data Models (Unified)
# ==============================================================================

class VectorDBRequest(BaseModel):
    vector_db_path: str

class AddDocumentsRequest(BaseModel):
    vector_db_path: str
    document_names: List[str]
    document_paths: List[str]

class DeleteDocumentsRequest(BaseModel):
    vector_db_path: str
    document_names: List[str]

# ==============================================================================
# SECTION 4: Root Endpoints
# This section defines the root endpoint for the Vector DB API.
# ==============================================================================
@app.get("/")
async def root() -> dict:
    """
    Root endpoint to verify if the Vector DB API is running.
    Returns:
        dict: Welcome message.
    """
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the Vector DB API!"}

# ==============================================================================
# SECTION 5: Health Check Endpoint
# This section defines the health check endpoint for the Vector DB API.
# ==============================================================================
@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint to verify if the Vector DB API is running.
    Returns:
        dict: Status message indicating the API is running.
    """
    logging.info("Health check endpoint accessed.")
    return {"status": "ok", "message": "Vector DB API is running."}

# ==============================================================================
# SECTION 6: Vector DB Creation and Update Endpoint
# This section defines the endpoint to create or update the vector DB.
# It loads, splits, and embeds all supported files in the specified folder.
# ==============================================================================
@app.post("/api/vectordb/create-or-update")
def create_or_update_vector_db(vector_db_path: str, data_upload_folder: str) -> dict:
    """
    Loads, splits, embeds all supported files in data_upload_folder and updates/creates the FAISS vector DB.
    If the vector DB already exists, it will update the existing index with new documents.
    Args:
        vector_db_path (str): Path to the vector DB folder.
        data_upload_folder (str): Path to the folder containing documents to be added.
    Returns:
        dict: Status message indicating whether the vector DB was created or updated, and number of chunks added.
    """
    ensure_folder_exists(vector_db_path)
    # List all files in the data_upload_folder
    document_names = []
    document_paths = []
    for fname in os.listdir(data_upload_folder):
        fpath = os.path.join(data_upload_folder, fname)
        if os.path.isfile(fpath):
            document_names.append(fname)
            document_paths.append(data_upload_folder)
    if not document_names:
        return {"status": "no_documents_found"}
    docs = load_and_split_documents(document_names, document_paths)
    if not docs:
        return {"status": "no_supported_documents_found"}
    embed_and_save_documents(vector_db_path=vector_db_path, chunk_docs=docs)
    return {"status": "vector_db_updated", "num_chunks": len(docs)}

# ==============================================================================
# SECTION 7: Unified Add Documents Endpoint
# ==============================================================================

@app.post("/api/vectordb/add")
def add_documents_to_vector_db(req: AddDocumentsRequest) -> dict:
    """
    Add one or more documents to the vector DB (incremental, does not re-embed all).
    Skips documents that already exist. Adds to both chunk_index and summary_index.
    """
    ensure_folder_exists(req.vector_db_path)
    existing_docs = get_all_document_names(req.vector_db_path)
    docs_to_add = []
    summary_docs_to_add = []
    added = []
    skipped = []
    client = OpenAI(api_key=creds['openai_api_key'])
    summarizer = load_summarize_chain(client, chain_type="map_reduce")
    for doc_name, doc_path in zip(req.document_names, req.document_paths):
        if doc_name in existing_docs:
            skipped.append(doc_name)
            continue
        docs = load_and_split_documents([doc_name], [doc_path])
        docs_to_add.extend(docs)
        added.append(doc_name)
        for doc in docs:
            try:
                summary_result = summarizer.invoke([doc])
                summary_text = summary_result["output_text"] if isinstance(summary_result, dict) else str(summary_result)
                summary_docs_to_add.append(Document(page_content=summary_text, metadata=doc.metadata))
            except Exception as e:
                logging.error(f"Failed to summarize {doc.metadata.get('source', '')}: {e}")
    if docs_to_add:
        add_documents_to_vector_db(req.vector_db_path, docs_to_add, "chunk_index")
    if summary_docs_to_add:
        add_documents_to_vector_db(req.vector_db_path, summary_docs_to_add, "summary_index")
    return {"status": "added", "added": added, "skipped": skipped}

# ==============================================================================
# SECTION 8: Unified Delete Documents Endpoint
# ==============================================================================

@app.delete("/api/vectordb/delete")
def delete_documents_from_vector_db_api(req: DeleteDocumentsRequest) -> dict:
    """
    Delete one or more documents from the vector DB (by document names) in both chunk_index and summary_index.
    """
    result = delete_documents_from_vector_db(req.vector_db_path, req.document_names)
    return result

# ==============================================================================
# SECTION 9: Vector DB Clear Endpoint
# This section defines an endpoint to clear all documents from the vector DB.
# The DB files remain, but all documents are removed.
# ==============================================================================
@app.post("/api/vectordb/clear")
def clear_vector_db_api(req: VectorDBRequest) -> dict:
    """
    Clear all documents from both chunk_index and summary_index in the vector DB.
    The DB files remain, but all documents are removed.
    Args:
        req (VectorDBRequest): Request containing the vector DB path.
    Returns:
        dict: Status message indicating the vector DB has been cleared.
    """
    return clear_vector_db(req.vector_db_path)

# ==============================================================================
# SECTION 10: List All Documents Endpoint
# This section defines an endpoint to list all document names in the vector DB.
# ==============================================================================

@app.get("/api/vectordb/list")
def list_documents_in_vector_db(vector_db_path: str) -> dict:
    """
    List all document names in the vector DB (based on chunk_index and summary_index metadata).
    Args:
        vector_db_path (str): Path to the vector DB folder.
    Returns:
        dict: List of unique document names found in the vector DB.
    """
    doc_names = get_all_document_names(vector_db_path)
    return {"documents": doc_names}

# ==============================================================================
# SECTION 11: End of File
# This is the end of the Vector DB API implementation.
# ==============================================================================