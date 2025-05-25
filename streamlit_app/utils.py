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
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "streamlit_app.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Change to logging.INFO if you want info logs in file
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True  # Force the logging configuration to be applied
)

def generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    session_id = f"{timestamp}_{unique_id}"
    logging.info(f"Generated session_id: {session_id}")
    return session_id

def initialize_chat_history() -> list:
    logging.info("Initialized empty chat history")
    return []

def append_to_chat_history(chat_history, role, content) -> list:
    chat_history.append({"role": role, "content": content})
    logging.info(f"Appended to chat history: role={role}, content={content}")
    return chat_history


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_middleware_if_needed():
    port = find_free_port() # Or use find_free_port() if you want a random port
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

def send_query_to_middleware(
    middleware_url,
    bot_name,
    query,
    user_name,
    session_id,
    access_key,
    chat_history,
    content_type="",
    document_name = "",
    document= "",     
    top_p=1.0,
    temperature=0.7,   
    personalai_prompt = "",    
    assistant_id= "",    
    thread_id = "",
    message_file_id = "",
    model_name="gpt-4o-mini",
):
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
            "document": document,      
            "top_p": top_p,       
            "temperature": temperature,            
            "personalai_prompt": personalai_prompt,            
            "assistant_id": assistant_id,            
            "thread_id": thread_id,            
            "message_file_id": message_file_id,
            "model_name": model_name
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        logging.info(f"Received response from middleware: {response.text}")
        return response.json()
    except Exception as e:
        logging.error("Error in send_query_to_middleware: %s", str(e), exc_info=True)
        return {"error": str(e)}
    


