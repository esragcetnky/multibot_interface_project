# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the Compare Files bot.
# ==============================================================================
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import yaml
import logging
import base64
from openai import OpenAI
from vector_db.utils import update_or_create_vector_db, get_combined_context

# ==============================================================================
# SECTION 2: Logging Configuration
# This section configures logging for the Compare Files bot.
# ==============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Vectorstore path for storing indexed documents
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, "data", "vectorstores", "compare_files")
# Data uploads directory for storing uploaded documents
DATA_UPLOADS_PATH = os.path.join(PROJECT_ROOT, "data", "uploads", "compare_files")
# Load credentials from YAML file
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "shared", "credentials.yml")

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "compare_files.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Set to INFO to capture info logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    encoding="utf-8",
    force=True
)

# ==============================================================================
# SECTION 3: Credentials and OpenAI Client Initialization
# This section loads API credentials and initializes the OpenAI client.
# ==============================================================================

logging.info(f"Loading credentials from {CREDENTIALS_PATH}")
with open(CREDENTIALS_PATH, "r") as file:
    credentials = yaml.safe_load(file)
logging.info("Credentials loaded successfully.")

# Initialize OpenAI client with API key
client = OpenAI(api_key=credentials["openai_api_key"])
logging.info("OpenAI client initialized.")

# ==============================================================================
# SECTION 4: Compare Files Service
# This section defines the main function for answering questions using OpenAI API.
# ==============================================================================

def compare_files_service(
    query: str,
    user_name: str = "",
    session_id: str = "",
    access_key: str = "",
    chat_history: list = [],
    content_type: str = "",
    document_name: list = [],
    document_path: list = [],
    top_p: float = 1.0,
    temperature: float = 0.7,
    personalai_prompt: str = "",
    assistant_id: str = "",
    thread_id: str = "",
    message_file_id: str = "",
    model_name: str = "gpt-4o-mini"
) -> str:
    """
    Main service function for the Compare Files bot.
    Handles document indexing, context retrieval, and OpenAI API interaction.

    Args:
        query (str): The user's question or prompt.
        user_name (str): The name or identifier of the user.
        session_id (str): The session identifier for tracking conversation.
        access_key (str): Access key for authentication or authorization.
        chat_history (list): List of previous chat messages.
        content_type (str): MIME type of the uploaded document.
        document_name (list): Name of the uploaded document file.
        document_path (list): Path to the uploaded document file.
        top_p (float): Nucleus sampling parameter for OpenAI completion.
        temperature (float): Sampling temperature for OpenAI completion.
        personalai_prompt (str): Custom prompt for personal AI context.
        assistant_id (str): Identifier for a specific assistant (if used).
        thread_id (str): Identifier for a conversation thread (if used).
        message_file_id (str): Identifier for a message file (if used).
        model_name (str): The name of the OpenAI model to use.

    Returns:
        str: The response from the OpenAI API or an error message.
    """
    logging.info(f"compare_files_service called with query='{query}', model_name='{model_name}', temperature={temperature}, top_p={top_p}")

    print(f"{document_name=}, {document_path=}")


    if document_name or document_path:
        # Update or create vector database with the new document
        update_or_create_vector_db(VECTORSTORE_PATH, document_name, document_path)
        logging.info(f"Updating vectorstore with document: {document_name} at {document_path}")

    # 2. Retrieve relevant context from vectorstore
    context = ""
    try:
        context = get_combined_context(query=query, vector_db_path=VECTORSTORE_PATH)
        logging.info(f"Retrieved context for query: {context}")
    except Exception as e:
        logging.warning(f"Could not retrieve context: {e}")

    # 3. Compose the system prompt with context if available
    system_prompt = (
        "You are a helpful assistant that will compare files and answer questions based on their contents. "
        "You have access to the contents of the uploaded documents and can provide insights based on them. "
        "Your task is to compare the contents of the uploaded documents and provide insights or answers based on the user's query. "
        "If you do not have enough information, you should indicate that."
        "If user does not provide any documents, you should inform them that you cannot answer without documents."
        "If user ask something unrelated to the documents, you should inform them that you can only answer questions related to the uploaded documents."
    )
    
    if document_name or document_path:
        system_prompt += f"\n\nYou have access to the following document(s):\n {', '.join(document_name)}\n"

    if context:
        system_prompt += f"\n\nRelevant context:\n{context}"

    logging.info(f"System prompt: {system_prompt}")
    logging.info(f"Chat history: {chat_history}")

    try:
        logging.info("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                *chat_history,
                {"role": "user", "content": query}
            ],
            temperature=temperature,
            top_p=top_p
        )
        logging.info("Received response from OpenAI API.")
        result = response.choices[0].message.content.strip()
        logging.info(f"OpenAI API result: {result}")
        return result
    except Exception as e:
        logging.error("Error calling OpenAI API: %s", str(e), exc_info=True)
        return f"Error calling OpenAI API: {str(e)}"
