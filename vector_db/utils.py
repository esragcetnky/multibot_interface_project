# ==============================================================================
# SECTION 1: Imports and Setup
# This section imports required modules and sets up paths and credentials.
# ==============================================================================
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import yaml
from typing import List
import logging
from fastapi import HTTPException

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
LOG_FILE = os.path.join(LOGS_DIR, "faiss_db.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Set to INFO to capture info logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    encoding="utf-8",
    force=True
)


def load_credentials():
    """Load API credentials from YAML file."""
    with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
        logging.info(f"Loading credentials from {CREDENTIALS_PATH}")
        return yaml.safe_load(f)
    
creds = load_credentials()
embeddings = OpenAIEmbeddings(api_key=creds["openai_api_key"],model="text-embedding-3-large")

# ==============================================================================
# SECTION 3: Utility Functions
# This section provides helper functions for folder and file management.
# ==============================================================================

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
    Args:
        vector_db_path (str): Path to the vector DB folder.
        document_name (str): Name of the document to check. 
    Returns:
        bool: True if the document exists, False otherwise.
    """
    doc_names = get_all_document_names(vector_db_path)
    return document_name in doc_names

def delete_documents_from_vector_db(vector_db_path: str, document_names: List[str]) -> dict:
    """
    Deletes one or more documents from both chunk_index and summary_index by document name.
    Args:
        vector_db_path (str): Path to the vector DB folder.
        document_names (List[str]): List of document names to delete.
    Returns:
        dict: A dictionary with two keys:
            - "deleted": List of document names that were successfully deleted.
            - "not_found": List of document names that were not found in the vector DB.
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
    If the index does not exist, it creates a new one.
    Args:
        vector_db_path (str): Path to the vector DB folder.
        docs (List[Document]): List of LangChain Document objects to add.
        index_name (str): Name of the index to add documents to ("chunk_index" or "summary_index").
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
    Args:
        vector_db_path (str): Path to the vector DB folder.
        index_name (str): Name of the index to clear ("chunk_index" or "summary_index").    
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
    Args:
        vector_db_path (str): Path to the vector DB folder. 
    Returns:
        dict: A dictionary with a status message indicating the vector DB has been cleared.
    """
    for index_name in ["chunk_index", "summary_index"]:
        try:
            clear_faiss_index(vector_db_path, index_name)
        except HTTPException as e:
            # If index doesn't exist, skip
            logging.warning(f"{index_name} not found at {vector_db_path}, skipping clear.")
    logging.info(f"Cleared all documents from vector DB at path: {vector_db_path}")
    return {"status": "cleared", "vector_db_path": vector_db_path}
