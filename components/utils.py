# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the Streamlit app.
# ==============================================================================
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import uuid
from datetime import datetime
import requests
import os
import logging
import subprocess
import sys
import time
import socket

# Set up logging
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "streamlit_app.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Change to logging.INFO if you want info logs in file
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True, # Force the logging configuration to be applied
    encoding="utf-8"
)

# ==============================================================================
# SECTION 2: Session and Chat History Utilities
# This section provides functions for session ID generation and chat history management.
# ==============================================================================

def generate_session_id() -> str:
    """
    Generate a unique session ID using current timestamp and a random UUID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    unique_id = str(uuid.uuid4())[:8]
    session_id = f"{timestamp}_{unique_id}"
    logging.info(f"Generated session_id: {session_id}")
    return session_id

def initialize_chat_history() -> list:
    """
    Initialize an empty chat history list.
    """
    logging.info("Initialized empty chat history")
    return []

def append_to_chat_history(chat_history, role, content) -> list:
    """
    Append a message to the chat history.

    Args:
        chat_history (list): Current chat history.
        role (str): 'user' or 'assistant'.
        content (str): Message content.

    Returns:
        list: Updated chat history.
    """
    chat_history.append({"role": role, "content": content})
    logging.info(f"Appended to chat history: role={role}, content={content}")
    return chat_history

# ==============================================================================
# SECTION 3: Middleware Process Management
# This section contains functions to find a free port, check port usage, and start the middleware if needed.
# ==============================================================================

def find_free_port():
    """
    Find a free port on localhost.
    Returns:
        int: An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def is_port_in_use(port):
    """
    Check if a given port is already in use.
    Args:
        port (int): Port number to check.
    Returns:
        bool: True if port is in use, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_middleware_if_needed():
    """
    Start the FastAPI middleware as a subprocess on a free port if not already running.
    Returns:
        int: The port number the middleware is running on.
    """
    port = find_free_port()
    if not is_port_in_use(port):
        # Start the middleware FastAPI app as a subprocess
        cmd = [
            sys.executable, "-m", "uvicorn",
            "middleware.main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "info"
        ]
        subprocess.Popen(cmd)
        # Wait for the server to start
        for _ in range(20):
            if is_port_in_use(port):
                break
            time.sleep(0.5)
    return port

# ==============================================================================
# SECTION 4: Middleware Communication
# This section provides the function to send a query to the middleware's /api/chat endpoint.
# ==============================================================================

def send_query_to_middleware(
    middleware_url,
    bot_name,
    query,
    user_name,
    session_id,
    access_key,
    chat_history,
    content_type="",
    document_name = [],
    document_path = [],     
    top_p=1.0,
    temperature=0.7,   
    personalai_prompt = "",    
    assistant_id= "",    
    thread_id = "",
    message_file_id = "",
    model_name="gpt-4o-mini",
):
    """
    Send a chat/query payload to the middleware's /api/chat endpoint.

    Args:
        middleware_url (str): URL of the middleware server.
        bot_name (str): Name of the bot to route to.
        query (str): User's query.
        user_name (str): Name of the user.
        session_id (str): Session identifier.
        access_key (str): Access key for authentication.
        chat_history (list): List of previous chat messages.
        ... (other optional parameters)

    Returns:
        dict: Middleware or bot response, or error message.
    """
    url = f"{middleware_url}/api/chat"

    payload = {
        "bot_name": bot_name,
        "query_input": {
            "query": query,
            "user_name": user_name,            
            "session_id": session_id,            
            "access_key": access_key,            
            "chat_history": chat_history,            
            "content_type": content_type,
            "document_name": document_name,            
            "document_path": document_path,      
            "top_p": top_p,       
            "temperature": temperature,            
            "personalai_prompt": personalai_prompt,            
            "assistant_id": assistant_id,            
            "thread_id": thread_id,            
            "message_file_id": message_file_id,
            "model_name": model_name
        }
    }
    logging.info(f"Sending query to middleware: {payload}")
    try:
        response = requests.post(url, json=payload, timeout=30, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        logging.info(f"Received response from middleware: {response.text}")
        return response.json()
    except Exception as e:
        logging.error("Error in send_query_to_middleware: %s", str(e), exc_info=True)
        return {"error": str(e)}

# ==============================================================================
# SECTION 5: Chat History Preparation
# This section provides a function to prepare chat history for API consumption.
# ==============================================================================

def prepare_chat_history_for_api(chat_history):
    """
    Prepare chat history for API consumption.
    Args:
        chat_history (list): List of chat messages, each a dict with 'role' and 'content'.
    Returns:
        list: Prepared chat history with each message as a dict with 'role' and 'content'.
    """
    prepared = []
    for message in chat_history:
        # Defensive: ensure only dicts with role and content keys
        if isinstance(message, dict):
            role = message.get("role", "") # Default to empty string if role is missing
            content = message.get("content", "") # Default to empty string if content is missing
            # Convert content to string to avoid serialization issues
            prepared.append({"role": role, "content": str(content)}) # Ensure content is a string
    logging.info(f"Prepared chat history for API: {prepared}")
    if not prepared:
        logging.warning("Prepared chat history is empty. Returning an empty list.")
        return []
    return prepared

# ==============================================================================
# SECTION 6: File Upload Handling
# This section provides functions to handle file uploads, including saving uploaded files to the server.
# ==============================================================================

def save_uploaded_file(uploaded_file, session_id, bot_name="ask_me_anything"):
    """
    Save the uploaded file to the data/uploads directory and return its path and original name.
    """
    # Data directory structure and ensure it exists
    uploads_dir = os.path.join("data", "uploads", bot_name)
    os.makedirs(uploads_dir, exist_ok=True)

    # Generate a safe file name and save the file
    safe_name = f"{session_id}_{uploaded_file.name}"
    # Ensure the file name is unique by appending session_id and timestamp
    file_path = os.path.join(uploads_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return uploads_dir, safe_name

# ==============================================================================
# SECTION 7: Folder Management
# This section provides functions to check if FAISS files exist and to clear a folder.
# ==============================================================================

def clear_folder(folder_path):
    """Delete all files in the given folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)