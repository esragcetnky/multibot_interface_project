# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the Ask Me Anything bot.
# ==============================================================================

import os
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from .service import ask_me_anything_service
from typing import List, Optional, Dict, Any
from fastapi.requests import Request
from fastapi.responses import JSONResponse

# Set up logging directory and file
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "ask_me_anything.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Set to INFO to capture info logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

# ==============================================================================
# SECTION 2: FastAPI App and Pydantic Models
# This section initializes the FastAPI app and defines request/response models.
# ==============================================================================

app = FastAPI()

class ChatMessage(BaseModel):
    """
    Model for a single chat message in the chat history.
    """
    role: str
    content: str

class QueryInput(BaseModel):
    """
    Model for the input to the Ask Me Anything bot.
    """
    query: str
    user_name: Optional[str] = ""
    session_id: Optional[str] = ""
    access_key: Optional[str] = ""
    chat_history: Optional[List[ChatMessage]] = []
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
# SECTION 3: Middleware and Exception Handlers
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
# SECTION 4: Bot Endpoint
# This section defines the main /api/v1 endpoint for the Ask Me Anything bot.
# ==============================================================================

@app.post("/api/v1")
async def handle_query(query_input: QueryInput):
    """
    Handle incoming Ask Me Anything requests.

    Args:
        query_input (QueryInput): The input payload for the bot.

    Returns:
        dict: The response containing the answer or error details.
    """
    logging.info("Bot endpoint /api/v1 HIT")
    logging.info(f"Bot received QueryInput:")
    logging.info(f"Query: {query_input.query}")
    logging.info(f"User: {query_input.user_name}, Session: {query_input.session_id}")
    logging.info(f"Model: {query_input.model_name}, Temp: {query_input.temperature}, Top_p: {query_input.top_p}")
    logging.info(f"Chat history: {query_input.chat_history}")
    try:
        logging.info("Calling ask_me_anything service...")
        # Call the ask_me_anything service with the provided parameters
        result = ask_me_anything_service(
            query=query_input.query,
            model_name=query_input.model_name,
            temperature=query_input.temperature,
            top_p=query_input.top_p
        )
        logging.info(f"ask_me_anything result: {result}")
        return {"response": result}
    except Exception as e:
        logging.error("Internal bot error: %s", str(e), exc_info=True)
        return {"error": "Internal bot error", "details": str(e)}