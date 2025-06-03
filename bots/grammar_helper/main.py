# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the Grammar Helper bot.
# ==============================================================================
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from .service import grammar_correction_service
from typing import List, Optional, Dict, Any
from fastapi.requests import Request
from fastapi.responses import JSONResponse

# ==============================================================================
# SECTION 2: Logging Configuration
# This section configures logging for the Ask Me Anything bot.
# ==============================================================================

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
# SECTION 3: FastAPI App Initialization
# This section initializes the FastAPI app and sets up the router for the Ask Me Anything bot.
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
    Model for the input to the grammar helper bot.
    """
    query: str
    user_name: Optional[str] = ""
    session_id: Optional[str] = ""
    access_key: Optional[str] = ""
    chat_history: Optional[List[dict]] = []
    content_type: Optional[str] = ""
    document_name: Optional[str] = ""
    document: Optional[str] = ""
    top_p: Optional[float] = 1.0
    temperature: Optional[float] = 0.7
    personalai_prompt: Optional[str] = ""
    assistant_id: Optional[str] = ""
    thread_id: Optional[str] = ""
    message_file_id: Optional[str] = ""
    model_name: Optional[str] = "gpt-4o-mini"

# ==============================================================================
# SECTION 5: Health Check Endpoints
# This section defines the root and health check endpoints for the Ask Me Anything bot.
# ==============================================================================
@app.get("/")
async def root():
    """
    Root endpoint to verify if the middleware is running.
    Returns:
        dict: Welcome message.
    """
    return {"message": "Welcome to the Ask Me Anything bot. Use /docs for API documentation."}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the middleware is running.
    Returns:
        dict: Status message.
    """
    return {"status": "ok", "message": "Ask me anything is running."}

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
# This section defines the main /api/v1 endpoint for grammar correction.
# ==============================================================================

@app.post("/api/v1")
async def handle_query(query_input: QueryInput):
    """
    Handle incoming grammar correction requests.

    Args:
        query_input (QueryInput): The input payload for grammar correction.

    Returns:
        dict: The response containing the corrected text or error details.
    """
    logging.info("Bot endpoint /api/v1 HIT")
    logging.info(f"Bot received QueryInput:")
    logging.info(f"Query: {query_input.query}")
    logging.info(f"User: {query_input.user_name}, Session: {query_input.session_id}")
    logging.info(f"Model: {query_input.model_name}, Temp: {query_input.temperature}, Top_p: {query_input.top_p}")
    logging.info(f"Chat history: {query_input.chat_history}")
    try:
        logging.info("Calling grammar_correction_service...")
        # Call the grammar correction service with the provided parameters
        result = grammar_correction_service(
            query=query_input.query,
            user_name=query_input.user_name,
            session_id=query_input.session_id,
            access_key=query_input.access_key,
            chat_history=query_input.chat_history,
            content_type=query_input.content_type,
            document_name=query_input.document_name,
            document=query_input.document,
            top_p=query_input.top_p,
            temperature=query_input.temperature,
            personalai_prompt=query_input.personalai_prompt,
            assistant_id=query_input.assistant_id,
            thread_id=query_input.thread_id,
            message_file_id=query_input.message_file_id,
            model_name=query_input.model_name,
        )
        logging.info(f"grammar_correction_service result: {result}")
        return {"response": result}
    except Exception as e:
        logging.error("Internal bot error: %s", str(e), exc_info=True)
        return {"error": "Internal bot error", "details": str(e)}
    
