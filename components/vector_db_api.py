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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
from langchain.chains import ConversationalRetrievalChain

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
    chunk_index_path = os.path.join(vector_db_path, "chunk_index")
    faiss_file = os.path.join(vector_db_path, "chunk_index.faiss")
    if not os.path.exists(faiss_file):
        logging.info(f"Vector DB not found at {faiss_file}")
        return []
    chunk_index = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        index_name="chunk_index"
    )
    doc_names = set()
    for doc in chunk_index.docstore._dict.values():
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
    Deletes one or more documents from the vector DB by document name.
    """
    faiss_file = os.path.join(vector_db_path, "chunk_index.faiss")
    if not os.path.exists(faiss_file):
        raise HTTPException(status_code=404, detail="Vector DB not found.")
    chunk_index = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        index_name="chunk_index"
    )
    deleted = []
    not_found = []
    for doc_name in document_names:
        keys_to_delete = [
            k for k, doc in chunk_index.docstore._dict.items()
            if hasattr(doc, "metadata") and doc.metadata.get("source") == doc_name
        ]
        if not keys_to_delete:
            not_found.append(doc_name)
            continue
        for k in keys_to_delete:
            del chunk_index.docstore._dict[k]
        deleted.append(doc_name)
    chunk_index.save_local(folder_path=vector_db_path, index_name="chunk_index")
    return {"deleted": deleted, "not_found": not_found}

def clear_vector_db(vector_db_path: str) -> dict:
    """
    Deletes the entire vector DB folder.
    """
    if not os.path.exists(vector_db_path):
        raise HTTPException(status_code=404, detail="Vector DB not found.")
    shutil.rmtree(vector_db_path)
    logging.info(f"Deleted vector DB at path: {vector_db_path}")
    return {"status": "deleted", "vector_db_path": vector_db_path}

# ==============================================================================
# SECTION 5: API Endpoints
# ==============================================================================

@app.get("/")
async def root() -> dict:
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the Vector DB API!"}

@app.get("/health")
async def health_check() -> dict:
    logging.info("Health check endpoint accessed.")
    return {"status": "ok", "message": "Vector DB API is running."}

@app.post("/api/vectordb/create-or-update")
def create_or_update_vector_db(vector_db_path: str, data_upload_folder: str) -> dict:
    """
    Loads, splits, embeds all supported files in data_upload_folder and updates/creates the FAISS vector DB.
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
    """
    ensure_folder_exists(req.vector_db_path)
    # Check if document already exists
    if document_exists_in_vector_db(req.vector_db_path, req.document_name):
        return {"status": "already_exists", "document_name": req.document_name}
    docs = load_and_split_documents([req.document_name], [req.document_path])
    if not docs:
        raise HTTPException(status_code=400, detail="No supported documents found.")
    faiss_file = os.path.join(req.vector_db_path, "chunk_index.faiss")
    if os.path.exists(faiss_file):
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

@app.post("/api/vectordb/add-multiple")
def add_multiple_documents_to_vector_db(req: AddMultipleDocumentsRequest) -> dict:
    """
    Add multiple documents to the vector DB (incremental, does not re-embed all).
    Skips documents that already exist.
    """
    ensure_folder_exists(req.vector_db_path)
    existing_docs = get_all_document_names(req.vector_db_path)
    docs_to_add = []
    added = []
    skipped = []
    for doc_name, doc_path in zip(req.document_names, req.document_paths):
        if doc_name in existing_docs:
            skipped.append(doc_name)
            continue
        docs = load_and_split_documents([doc_name], [doc_path])
        docs_to_add.extend(docs)
        added.append(doc_name)
    if not docs_to_add:
        return {"status": "no_new_documents_added", "skipped": skipped}
    faiss_file = os.path.join(req.vector_db_path, "chunk_index.faiss")
    if os.path.exists(faiss_file):
        chunk_index = FAISS.load_local(
            folder_path=req.vector_db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            index_name="chunk_index"
        )
        chunk_index.add_documents(docs_to_add)
        chunk_index.save_local(folder_path=req.vector_db_path, index_name="chunk_index")
    else:
        chunk_index = FAISS.from_documents(documents=docs_to_add, embedding=embeddings)
        chunk_index.save_local(folder_path=req.vector_db_path, index_name="chunk_index")
    return {"status": "added", "added": added, "skipped": skipped}

@app.delete("/api/vectordb/delete-document")
def delete_document_from_vector_db(req: DeleteDocumentRequest) -> dict:
    """
    Delete a single document from the vector DB (by document name).
    Only removes matching documents from the index, does not rebuild the whole DB.
    """
    if not document_exists_in_vector_db(req.vector_db_path, req.document_name):
        raise HTTPException(status_code=404, detail="Document not found in vector DB.")
    result = delete_documents_from_vector_db(req.vector_db_path, [req.document_name])
    return result

@app.delete("/api/vectordb/delete-multiple")
def delete_multiple_documents_from_vector_db(req: DeleteMultipleDocumentsRequest) -> dict:
    """
    Delete multiple documents from the vector DB (by document names).
    """
    result = delete_documents_from_vector_db(req.vector_db_path, req.document_names)
    return result

@app.delete("/api/vectordb/delete")
def delete_vector_db(req: VectorDBRequest) -> dict:
    """
    Delete the entire vector DB folder.
    """
    return clear_vector_db(req.vector_db_path)

@app.get("/api/vectordb/list")
def list_documents_in_vector_db(vector_db_path: str) -> List[str]:
    """
    List all document names in the vector DB (based on chunk_index metadata).
    """
    return get_all_document_names(vector_db_path)

@app.post("/api/vectordb/clear")
def clear_vector_db_api(req: VectorDBRequest) -> dict:
    """
    Clear the entire vector DB folder (alias for delete).
    """
    return clear_vector_db(req.vector_db_path)

# ==============================================================================
# SECTION 6: End of Vector DB API
# You can add more endpoints for advanced vector DB management here.
# ==============================================================================