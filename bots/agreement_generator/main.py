# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the Agreement Generator bot.
# ==============================================================================
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from .service import agreement_generator_service
from typing import List, Optional
from fastapi.responses import JSONResponse

# ==============================================================================
# SECTION 2: Logging Configuration
# This section configures logging for the Agreement Generator bot.
# ==============================================================================

# Set up logging directory and file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "agreement_generator.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Set to INFO to capture info logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    encoding="utf-8",
    force=True
)

# ==============================================================================
# SECTION 3: FastAPI App Initialization
# This section initializes the FastAPI app and sets up the router for the Agreement Generator bot.
# ==============================================================================
app = FastAPI()

# ==============================================================================
# SECTION 4: Pydantic Models
# This section defines the Pydantic models for request and response validation.
# ==============================================================================
class ChatMessage(BaseModel):
    """
    Model for a single chat message in the chat history.
    """
    role: str
    content: str

class QueryInput(BaseModel):
    """
    Model for the input to the Agreement Generator bot.
    """
    query: str
    user_name: Optional[str] = ""
    session_id: Optional[str] = ""
    access_key: Optional[str] = ""
    chat_history: Optional[List[dict]] = []
    content_type: Optional[str] = ""
    top_p: Optional[float] = 1.0
    temperature: Optional[float] = 0.7
    personalai_prompt: Optional[str] = ""
    assistant_id: Optional[str] = ""
    thread_id: Optional[str] = ""
    message_file_id: Optional[str] = ""
    model_name: Optional[str] = "gpt-4o-mini"

# ==============================================================================
# SECTION 5: Health Check Endpoints
# This section defines the root and health check endpoints for the Agreement Generator bot.
# ==============================================================================
@app.get("/")
async def root():
    """
    Root endpoint to verify if the Agreement Generator bot is running.
    Returns:
        dict: Welcome message.
    """
    return {"message": "Welcome to the Agreement Generator bot. Use /docs for API documentation."}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the Agreement Generator bot is running.
    Returns:
        dict: Status message.
    """
    return {"status": "ok", "message": "Agreement Generator is running."}

# ==============================================================================
# SECTION 6: Middleware and Exception Handlers
# This section logs all incoming requests and handles global exceptions.
# ==============================================================================

@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    """
    Middleware to log all incoming HTTP requests.
    """
    logging.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to log and return errors as JSON responses.
    """
    logging.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# ==============================================================================
# SECTION 7: Bot Endpoint
# This section defines the main /api/v1 endpoint for the Agreement Generator bot.
# ==============================================================================
@app.post("/api/v1")
async def handle_query(query_input: QueryInput):
    """
    Handle incoming Agreement Generator requests.

    Args:
        query_input (QueryInput): The input payload for the bot, containing:
            - query (str): The user's question or prompt.
            - user_name (str, optional): The name or identifier of the user.
            - session_id (str, optional): The session identifier for tracking conversation.
            - access_key (str, optional): Access key for authentication or authorization.
            - chat_history (list of dict, optional): List of previous chat messages (each with 'role' and 'content').
            - content_type (str, optional): MIME type of the uploaded document (e.g., 'application/pdf').
            - top_p (float, optional): Nucleus sampling parameter for OpenAI completion.
            - temperature (float, optional): Sampling temperature for OpenAI completion.
            - personalai_prompt (str, optional): Custom prompt for personal AI context.
            - assistant_id (str, optional): Identifier for a specific assistant (if used).
            - thread_id (str, optional): Identifier for a conversation thread (if used).
            - message_file_id (str, optional): Identifier for a message file (if used).
            - model_name (str, optional): The name of the OpenAI model to use (e.g., 'gpt-4o-mini').

    Returns:
        dict: The response containing the generated agreement or error details.
    """
    logging.info("Agreement Generator endpoint /api/v1 HIT")
    logging.info(f"Bot received QueryInput:")
    logging.info(f"Query: {query_input.query}")
    logging.info(f"User: {query_input.user_name}, Session: {query_input.session_id}")
    logging.info(f"Model: {query_input.model_name}, Temp: {query_input.temperature}, Top_p: {query_input.top_p}")
    logging.info(f"Chat history: {query_input.chat_history}")
    try:
        # Call the main service function with all relevant parameters.
        # Each parameter is passed through from the QueryInput model.
        result = agreement_generator_service(
            query=query_input.query,                   # User's question or prompt
            user_name=query_input.user_name,           # User identifier
            session_id=query_input.session_id,         # Session identifier
            access_key=query_input.access_key,         # Access/auth key
            chat_history=query_input.chat_history,     # Previous chat messages
            content_type=query_input.content_type,     # Uploaded document MIME type
            top_p=query_input.top_p,                   # Nucleus sampling parameter
            temperature=query_input.temperature,       # Sampling temperature
            personalai_prompt=query_input.personalai_prompt, # Custom personal AI prompt
            assistant_id=query_input.assistant_id,     # Assistant identifier
            thread_id=query_input.thread_id,           # Thread identifier
            message_file_id=query_input.message_file_id, # Message file identifier
            model_name=query_input.model_name,         # OpenAI model name
        )
        logging.info(f"agreement_generator result: {result}")
        return {"response": result}
    except Exception as e:
        logging.error("Internal bot error: %s", str(e), exc_info=True)
        return {"error": "Internal bot error", "details": str(e)}


