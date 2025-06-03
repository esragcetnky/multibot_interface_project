# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the Grammar Helper service.
# ==============================================================================
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import os
import logging
import base64
from openai import OpenAI

# ==============================================================================
# SECTION 2: Logging Configuration
# This section configures logging for the Grammar Helper bot.
# ==============================================================================

# Vectorstore path for storing indexed documents
VECTORSTORE_PATH = "data/vectorstores/grammar_helper"

# Set up logging directory and file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "grammar_helper.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Set to INFO to capture info logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    encoding="utf-8",
    force=True  # Force the logging configuration to be applied
)

# ==============================================================================
# SECTION 3: Credentials and OpenAI Client Initialization
# This section loads API credentials and initializes the OpenAI client.
# ==============================================================================

# Load credentials from YAML file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "shared", "credentials.yml")

logging.info(f"Loading credentials from {CREDENTIALS_PATH}")
with open(CREDENTIALS_PATH, "r") as file:
    credentials = yaml.safe_load(file)
logging.info("Credentials loaded successfully.")

# Initialize OpenAI client with API key
client = OpenAI(api_key=credentials["openai_api_key"])
logging.info("OpenAI client initialized.")

# ==============================================================================
# SECTION 5: Grammar Correction Service (RAG-enabled)
# This section defines the main function for answering questions using OpenAI API and vectorstore context.
# ==============================================================================

def grammar_correction_service(
    query: str,
    user_name: str = "",
    session_id: str = "",
    access_key: str = "",
    chat_history: list = [],
    content_type: str = "",
    document_name: str = "",
    document: str = "",
    top_p: float = 1.0,
    temperature: float = 0.7,
    personalai_prompt: str = "",
    assistant_id: str = "",
    thread_id: str = "",
    message_file_id: str = "",
    model_name: str = "gpt-4o-mini"
) -> str:
    """
    Grammar correction service that uses OpenAI's chat completion API to answer questions.
    Handles document indexing, context retrieval, and OpenAI API interaction.

    Args:
        query (str): The user's question.
        user_name (str): The name or identifier of the user.
        session_id (str): The session identifier for tracking conversation.
        access_key (str): Access key for authentication or authorization.
        chat_history (list): List of previous chat messages.
        content_type (str): MIME type of the uploaded document.
        document_name (str): Name of the uploaded document file.
        document (str): Base64-encoded content of the uploaded document.
        top_p (float): Nucleus sampling parameter.
        temperature (float): Sampling temperature.
        personalai_prompt (str): Custom prompt for personal AI context.
        assistant_id (str): Identifier for a specific assistant (if used).
        thread_id (str): Identifier for a conversation thread (if used).
        message_file_id (str): Identifier for a message file (if used).
        model_name (str): The OpenAI model to use.

    Returns:
        str: The answer from the OpenAI API or an error message.
    """
    logging.info(f"grammar_correction_service called with query='{query}', model_name='{model_name}', temperature={temperature}, top_p={top_p}")


    # 3. Compose the system prompt with context if available
    system_prompt = (
        "You are a helpful assistant that provides grammar corrections and suggestions. "
        "Your task is to analyze the user's query and provide a corrected version if necessary. "
        "If the query is already grammatically correct, simply return it unchanged. "
        "Only respond with the corrected text, without any additional explanations or comments."
        "If user wants something else, please clarify you are a grammar correction bot and cannot provide other services."
    )

    try:
        logging.info("Sending request to OpenAI API...")
        # Send the request to OpenAI's chat completion endpoint
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                *chat_history,  # Include chat history if available
                {"role": "user", "content": query}
            ],
            temperature=temperature,
            top_p=top_p
        )
        logging.info("Received response from OpenAI API.")
        # Extract and return the answer from the response
        result = response.choices[0].message.content.strip()
        logging.info(f"OpenAI API result: {result}")
        return result
    except Exception as e:
        logging.error("Error calling OpenAI API: %s", str(e), exc_info=True)
        return f"Error calling OpenAI API: {str(e)}"
