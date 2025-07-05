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

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader,
    UnstructuredHTMLLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import OpenAI

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
# SECTION 3: Data Models
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

class AddMultipleDocumentsRequest(BaseModel):
    vector_db_path: str
    document_names: List[str]
    document_paths: List[str]

class DeleteMultipleDocumentsRequest(BaseModel):
    vector_db_path: str
    document_names: List[str]

# ==============================================================================
# SECTION 4: Utility Functions (from faiss_db.py and helpers)
# ==============================================================================

def check_faiss_files_exist(folder_path) -> bool:
    """
    Checks if both 'index.faiss' and 'index.pkl' exist in the given folder.
    Returns:
        True if both files exist, False otherwise.
    """
    faiss_file = os.path.join(folder_path, "index.faiss")
    pkl_file = os.path.join(folder_path, "index.pkl")
    return os.path.isfile(faiss_file) and os.path.isfile(pkl_file)

def ensure_folder_exists(folder_path):
    """
    Check if a folder exists, and create it if it does not.
    Args:
        folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_and_split_documents(document_name: List[str], document_path: List[str]) -> List[Document]:
    """
    Loads all supported files from data_folder, splits them into chunks,
    and returns a list of LangChain Document objects.
    Supported formats: PDF, Word, Excel, HTML, TXT.
    """
    docs = []
    for i in range(len(document_name)):
        fname = document_name[i]
        fpath = os.path.join(document_path[i], fname)
        print(f"Loading {fname}...")
        logging.info(f"Loading {fname} from {fpath}")
        ext = os.path.splitext(fname)[1].lower()
        logging.info(f"File extension: {ext}")
        if not os.path.isfile(fpath):
            logging.warning(f"File {fpath} does not exist. Skipping.")
            continue
        try:
            if ext in [".pdf", ".PDF"]:
                loader = PyPDFLoader(fpath)
            elif ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(fpath)
            elif ext in [".xls", ".xlsx"]:
                loader = UnstructuredExcelLoader(fpath)
            elif ext == ".html":
                loader = UnstructuredHTMLLoader(fpath)
            elif ext == ".txt":
                loader = TextLoader(fpath, encoding="utf-8")
            else:
                continue
            file_docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            )
            logging.info(f"Loaded {len(file_docs)} documents from {fname}")
            docs.extend(splitter.split_documents(file_docs))
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return docs

def embed_and_save_documents(vector_db_path: str, chunk_docs: List[Document]) -> None:
    """
    Embeds the given documents and saves/updates the FAISS vector DB.
    Also creates a summary index for fast retrieval.
    """
    client = OpenAI(api_key=creds['openai_api_key'])
    summarizer = load_summarize_chain(client, chain_type="map_reduce")
    summary_docs = []
    for doc in chunk_docs:
        try:
            summary_result = summarizer.invoke([doc])
            summary_text = summary_result["output_text"] if isinstance(summary_result, dict) else str(summary_result)
            summary_docs.append(Document(page_content=summary_text, metadata=doc.metadata))
        except Exception as e:
            logging.error(f"Failed to summarize {doc.metadata.get('source', '')}: {e}")

    try:
        if check_faiss_files_exist(vector_db_path):
            # Update existing summary and chunk indexes
            summary_index = FAISS.load_local(folder_path=vector_db_path, 
                                             embeddings=embeddings, 
                                             allow_dangerous_deserialization=True,
                                             index_name="summary_index")
            summary_index.add_documents(summary_docs)
            summary_index.save_local("summary_index")
            
            chunk_index = FAISS.load_local(folder_path=vector_db_path, 
                                           embeddings=embeddings, 
                                           allow_dangerous_deserialization=True,     
                                           index_name="chunk_index")
            chunk_index.add_documents(chunk_docs)
            chunk_index.save_local(vector_db_path)
            
            print(f"FAISS DB updated and saved to {vector_db_path}")
            logging.info(f"FAISS DB updated and saved to {vector_db_path}")
        else:
            # Create new summary and chunk indexes
            summary_index = FAISS.from_documents(documents=summary_docs, embedding=embeddings)
            summary_index.save_local(folder_path=vector_db_path, index_name="summary_index")

            chunk_index = FAISS.from_documents(documents=chunk_docs, embedding=embeddings)
            chunk_index.save_local(folder_path=vector_db_path, index_name="chunk_index")
            
            print(f"FAISS DB created and saved to {vector_db_path}")
            logging.info(f"FAISS DB created and saved to {vector_db_path}")
    except Exception as e:
        print(f"Error embedding documents: {e}")
        logging.error(f"Error embedding documents: {e}", exc_info=True)

def get_all_document_names(vector_db_path: str) -> List[str]:
    """
    Returns a list of all document names in the vector DB (based on chunk_index metadata).
    """
    doc_names = set()
    for index_name in ["chunk_index", "summary_index"]:
        index_file = os.path.join(vector_db_path, f"{index_name}.faiss")
        if not os.path.exists(index_file):
            continue
        index = FAISS.load_local(
            folder_path=vector_db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            index_name=index_name
        )
        for doc in index.docstore._dict.values():
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                doc_names.add(doc.metadata["source"])
    return list(doc_names)

def document_exists_in_vector_db(vector_db_path: str, document_name: str) -> bool:
    """
    Checks if a document with the given name exists in the vector DB.
    """
    doc_names = get_all_document_names(vector_db_path)
    return document_name in doc_names

def delete_documents_from_vector_db(vector_db_path: str, document_names: List[str]) -> dict:
    """
    Deletes one or more documents from both chunk_index and summary_index by document name.
    """
    deleted = []
    not_found = []
    for index_name in ["chunk_index", "summary_index"]:
        index_file = os.path.join(vector_db_path, f"{index_name}.faiss")
        if not os.path.exists(index_file):
            continue
        index = FAISS.load_local(
            folder_path=vector_db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            index_name=index_name
        )
        for doc_name in document_names:
            keys_to_delete = [
                k for k, doc in index.docstore._dict.items()
                if hasattr(doc, "metadata") and doc.metadata.get("source") == doc_name
            ]
            if not keys_to_delete:
                not_found.append(doc_name)
                continue
            for k in keys_to_delete:
                del index.docstore._dict[k]
            deleted.append(doc_name)
        index.save_local(folder_path=vector_db_path, index_name=index_name)
    return {"deleted": list(set(deleted)), "not_found": list(set(not_found))}

def add_documents_to_vector_db(vector_db_path: str, docs: List[Document], index_name: str) -> None:
    """
    Adds documents to the specified FAISS index (chunk_index or summary_index).
    """
    index_file = os.path.join(vector_db_path, f"{index_name}.faiss")
    if os.path.exists(index_file):
        index = FAISS.load_local(
            folder_path=vector_db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            index_name=index_name
        )
        index.add_documents(docs)
        index.save_local(folder_path=vector_db_path, index_name=index_name)
    else:
        index = FAISS.from_documents(documents=docs, embedding=embeddings)
        index.save_local(folder_path=vector_db_path, index_name=index_name)

def clear_faiss_index(vector_db_path: str, index_name: str) -> None:
    """
    Clears all documents from the specified FAISS index (chunk_index or summary_index) but keeps the index file.
    """
    index_file = os.path.join(vector_db_path, f"{index_name}.faiss")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail=f"{index_name} not found.")
    index = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        index_name=index_name
    )
    # Remove all docs
    index.docstore._dict.clear()
    index.save_local(folder_path=vector_db_path, index_name=index_name)
    logging.info(f"Cleared all documents from {index_name} at {vector_db_path}")

def clear_vector_db(vector_db_path: str) -> dict:
    """
    Clears all documents from both chunk_index and summary_index in the vector DB.
    The DB files remain, but all documents are removed.
    """
    for index_name in ["chunk_index", "summary_index"]:
        try:
            clear_faiss_index(vector_db_path, index_name)
        except HTTPException as e:
            # If index doesn't exist, skip
            logging.warning(f"{index_name} not found at {vector_db_path}, skipping clear.")
    logging.info(f"Cleared all documents from vector DB at path: {vector_db_path}")
    return {"status": "cleared", "vector_db_path": vector_db_path}

# ==============================================================================
# SECTION 5: API Endpoints
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

@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint to verify if the Vector DB API is running.
    Returns:
        dict: Status message indicating the API is running.
    """
    logging.info("Health check endpoint accessed.")
    return {"status": "ok", "message": "Vector DB API is running."}

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

@app.post("/api/vectordb/add")
def add_document_to_vector_db(req: AddDocumentRequest) -> dict:
    """
    Add a new document to the vector DB (incremental, does not re-embed all).
    Checks if a document with the same name exists before adding.
    Adds to both chunk_index and summary_index.
    Args:
        req (AddDocumentRequest): Request containing vector DB path, document name, and path.
    Returns:
        dict: Status message indicating whether the document was added or already exists.
    """
    ensure_folder_exists(req.vector_db_path)
    if document_exists_in_vector_db(req.vector_db_path, req.document_name):
        return {"status": "already_exists", "document_name": req.document_name}
    docs = load_and_split_documents([req.document_name], [req.document_path])
    if not docs:
        raise HTTPException(status_code=400, detail="No supported documents found.")
    # Add to chunk_index
    add_documents_to_vector_db(req.vector_db_path, docs, "chunk_index")
    # Summarize and add to summary_index
    client = OpenAI(api_key=creds['openai_api_key'])
    summarizer = load_summarize_chain(client, chain_type="map_reduce")
    summary_docs = []
    for doc in docs:
        try:
            summary_result = summarizer.invoke([doc])
            summary_text = summary_result["output_text"] if isinstance(summary_result, dict) else str(summary_result)
            summary_docs.append(Document(page_content=summary_text, metadata=doc.metadata))
        except Exception as e:
            logging.error(f"Failed to summarize {doc.metadata.get('source', '')}: {e}")
    add_documents_to_vector_db(req.vector_db_path, summary_docs, "summary_index")
    return {"status": "added", "document_name": req.document_name}

@app.post("/api/vectordb/add-multiple")
def add_multiple_documents_to_vector_db(req: AddMultipleDocumentsRequest) -> dict:
    """
    Add multiple documents to the vector DB (incremental, does not re-embed all).
    Skips documents that already exist. Adds to both chunk_index and summary_index.
    Args:
        req (AddMultipleDocumentsRequest): Request containing vector DB path, document names, and paths.
    Returns:
        dict: Status message indicating which documents were added or skipped.
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

@app.delete("/api/vectordb/delete-document")
def delete_document_from_vector_db(req: DeleteDocumentRequest) -> dict:
    """
    Delete a single document from the vector DB (by document name) in both chunk_index and summary_index.
    Args:
        req (DeleteDocumentRequest): Request containing vector DB path and document name.
    Returns:
        dict: Status message indicating the document was deleted or not found.
    """
    if not document_exists_in_vector_db(req.vector_db_path, req.document_name):
        raise HTTPException(status_code=404, detail="Document not found in vector DB.")
    result = delete_documents_from_vector_db(req.vector_db_path, [req.document_name])
    return result

@app.delete("/api/vectordb/delete-multiple")
def delete_multiple_documents_from_vector_db(req: DeleteMultipleDocumentsRequest) -> dict:
    """
    Delete multiple documents from the vector DB (by document names) in both chunk_index and summary_index.
    Args:
        req (DeleteMultipleDocumentsRequest): Request containing vector DB path and document names.
    Returns:
        dict: Status message indicating which documents were deleted or not found.
    """
    result = delete_documents_from_vector_db(req.vector_db_path, req.document_names)
    return result

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
# SECTION 6: End of Vector DB API
# You can add more endpoints for advanced vector DB management here.
# ==============================================================================