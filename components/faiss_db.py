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


# LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
# LOG_FILE = os.path.join(LOGS_DIR, "faiss_db.log")

# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.INFO,  # Set to INFO to capture info logs
#     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
#     encoding="utf-8"
# )


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

def check_faiss_files_exist(folder_path) -> bool:
    """
    Checks if both 'index.faiss' and 'index.pkl' exist in the given folder.
    Returns:
        True if both files exist, False otherwise.
    """
    faiss_file = os.path.join(folder_path, "index.faiss")
    pkl_file = os.path.join(folder_path, "index.pkl")
    return os.path.isfile(faiss_file) and os.path.isfile(pkl_file)

# ==============================================================================
# SECTION 4: Document Loading and Splitting
# This section loads documents from disk and splits them into chunks.
# ==============================================================================

def load_and_split_documents(document_name: str, document_path : str) -> List:
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
        

# ==============================================================================
# SECTION 5: Embedding and Vector DB Management
# This section handles embedding documents and saving/updating the FAISS vector DB.
# ==============================================================================

def embed_and_save_documents(vector_db_path: str, chunk_docs: List) -> None:
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


# ==============================================================================
# SECTION 6: Folder Management
# This section provides functions to ensure folders exist and to clear a folder.
# ==============================================================================

def ensure_folder_exists(folder_path):
    """
    Check if a folder exists, and create it if it does not.
    Args:
        folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# ==============================================================================
# SECTION 7: Main Update Function
# This section provides a high-level function to update or create the vector DB.
# ==============================================================================

def update_or_create_vector_db(vectorstores_dir : str, document_name:str, document_path: str) -> str:
    """
    Loads, splits, embeds all supported files in data_folder and updates/creates the FAISS vector DB.
    Returns a status message.
    """
    ensure_folder_exists(vectorstores_dir)
    ensure_folder_exists(document_path[0])
    docs = load_and_split_documents(document_name=document_name, 
                                    document_path=document_path)
    if not docs:
        logging.warning("No supported documents found to embed.")
        return "No supported documents found."
    
    embed_and_save_documents(vector_db_path=vectorstores_dir, 
                             chunk_docs=docs)
    logging.info(f"Vector DB updated with {len(docs)} document chunks.")
    return f"Vector DB updated with {len(docs)} document chunks."

# ==============================================================================
# SECTION 8: Get Combined Context
# This section defines a combined retriever that uses both summary and chunk indexes.
# ==============================================================================

class CombinedRetriever:
    def __init__(self, summary_retriever, chunk_retriever):
        self.summary_retriever = summary_retriever
        self.chunk_retriever = chunk_retriever

    def get_relevant_documents(self, query: str):
        docs = self.summary_retriever.get_relevant_documents(query)
        docs += self.chunk_retriever.get_relevant_documents(query)
        unique = {doc.page_content: doc for doc in docs}
        return list(unique.values())

def get_combined_context(query: str, vector_db_path: str, k=3) -> CombinedRetriever:
    """
    Searches the FAISS vector DB for the most similar documents to the query.
    Returns a list of document page contents.
    """
    chunk_index = FAISS.load_local(folder_path=vector_db_path, 
                                   embeddings=embeddings, 
                                   allow_dangerous_deserialization=True,
                                   index_name="chunk_index")
    summary_index = FAISS.load_local(folder_path=vector_db_path, 
                                     embeddings=embeddings, 
                                     allow_dangerous_deserialization=True,
                                     index_name="summary_index")
    # === Step 5: RetrievalQA using both summary and chunk index ===
    summary_retriever = summary_index.as_retriever(search_kwargs={"k": 3})
    chunk_retriever = chunk_index.as_retriever(search_kwargs={"k": 6})

    combined_retriever = CombinedRetriever(summary_retriever, chunk_retriever)
    return combined_retriever