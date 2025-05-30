# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the Ask Me Anything service.
# ==============================================================================
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import yaml
import logging
import base64
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from components.faiss_db import (
    check_faiss_files_exist,
    update_or_create_vector_db
)

# ==============================================================================
# SECTION 2: Logging Configuration
# This section configures logging for the Ask Me Anything bot.
# ==============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Vectorstore path for storing indexed documents
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, "data", "vectorstores", "ask_me_anything")
# Load credentials from YAML file
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "shared", "credentials.yml")
# 
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "ask_me_anything.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Set to INFO to capture info logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    encoding="utf-8"
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
# SECTION 4: Ask Me Anything Service
# This section defines the main function for answering questions using OpenAI API.
# ==============================================================================

def ask_me_anything_service(
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
    Main service function for the Ask Me Anything bot.
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
    logging.info(f"ask_me_anything_service called with query='{query}', model_name='{model_name}', temperature={temperature}, top_p={top_p}")

    print(f"{document_name=}, {document_path=}")

    # for document in document_path:
    #     check_faiss_files_exist(VECTORSTORE_PATH)
    #     update_or_create_vector_db(
    #         vectorstore_path=VECTORSTORE_PATH,
    #         data_folder=document_path,
    #         document_name=document_name)
        

    # 2. Retrieve relevant context from vectorstore
    context = ""
    try:
        # context = retrieve_relevant_context(query)
        logging.info(f"Retrieved context for query: {context}")
    except Exception as e:
        logging.warning(f"Could not retrieve context: {e}")

    # 3. Compose the system prompt with context if available
    system_prompt = (
        "You are a helpful assistant that can answer any question accurately and concisely. "
        "Your responses should be informative and relevant to the user's query. "
        "If you don't know the answer, it's okay to say so. "
        "You should always strive to provide the best possible answer based on the information available."
    )
    if context:
        system_prompt += f"\n\nRelevant context:\n{context}"

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
