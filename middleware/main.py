# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the middleware.
# ==============================================================================

import os
import logging
import asyncio
import subprocess
import socket
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import threading
import traceback
from fastapi.requests import Request
from fastapi.responses import JSONResponse

# Set up logging directory and file
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "middleware.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True  # Force the logging configuration to be applied
)

# ==============================================================================
# SECTION 2: FastAPI App and Middleware Setup
# This section initializes the FastAPI app and configures CORS.
# ==============================================================================

app = FastAPI(title="Multi-Bot Middleware API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# SECTION 3: Bot Management Utilities
# This section defines bot commands, process management, and startup logic.
# ==============================================================================

# List of bots with their start commands (update paths as needed)
BOTS = {
    "ask_me_anything": "uvicorn bots.ask_me_anything.main:app --host 0.0.0.0 --port {port} --reload",
    "grammar_helper": "uvicorn bots.grammar_helper.main:app --host 0.0.0.0 --port {port} --reload",
    # "compare_files": "uvicorn bots.compare_files.main:app --host 0.0.0.0 --port {port} --reload",
    # "create_new_law_agreements": "uvicorn bots.create_new_law_agreements.main:app --host 0.0.0.0 --port {port} --reload",
}

# Store bot subprocesses and their ports
bot_processes = {}
BOT_API_URLS = {}

def find_free_port():
    """
    Find a free port on localhost.
    Returns:
        int: An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
        logging.info(f"Found free port: {port}")
        return port

def log_subprocess_output(pipe):
    """
    Log output from a subprocess (bot) to the middleware log.
    Args:
        pipe: The stdout or stderr pipe from the subprocess.
    """
    for line in iter(pipe.readline, b''):
        decoded_line = line.decode().rstrip()
        logging.info(f"[BOT OUTPUT] {decoded_line}")
        print(f"[BOT OUTPUT] {decoded_line}")

async def start_bot(bot_name):
    """
    Start a bot subprocess if not already running.
    Args:
        bot_name (str): Name of the bot to start.
    """
    if bot_name in bot_processes:
        logging.info(f"Bot '{bot_name}' is already running.")
        return

    port = find_free_port()
    cmd = BOTS[bot_name].format(port=port)
    logging.info(f"Launching bot '{bot_name}' with command: {cmd}")

    try:
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1
        )
        logging.info(f"Started subprocess for bot '{bot_name}' on port {port}. PID: {process.pid}")
    except Exception as e:
        logging.error(f"Failed to start bot '{bot_name}': {e}", exc_info=True)
        raise

    # Start threads to log stdout and stderr from the bot process
    threading.Thread(target=log_subprocess_output, args=(process.stdout,), daemon=True).start()
    threading.Thread(target=log_subprocess_output, args=(process.stderr,), daemon=True).start()

    bot_processes[bot_name] = process
    BOT_API_URLS[bot_name] = f"http://localhost:{port}/api/v1"
    logging.info(f"Bot '{bot_name}' API URL set to {BOT_API_URLS[bot_name]}")

async def start_all_bots():
    """
    Start all bots defined in the BOTS dictionary.
    """
    tasks = []
    logging.info("Starting all bots...")
    for bot_name in BOTS.keys():
        logging.info(f"Scheduling startup for bot: {bot_name}")
        tasks.append(start_bot(bot_name))
    await asyncio.gather(*tasks)
    logging.info("All bots have been started.")

# ==============================================================================
# SECTION 4: Middleware and Exception Handlers
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
# SECTION 5: FastAPI Startup Event
# This section starts all bots when the middleware starts up.
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event to start all bots.
    """
    logging.info("FastAPI startup event triggered. Starting all bots...")
    await start_all_bots()
    await asyncio.sleep(2)
    logging.info("Startup event completed. Bots should be ready.")

# ==============================================================================
# SECTION 6: Pydantic Models
# This section defines the request models for the middleware API.
# ==============================================================================

class QueryInput(BaseModel):
    """
    Model for the input to each bot.
    """
    query: str
    user_name: str
    session_id: str
    access_key: str
    chat_history: list
    content_type: str = ""
    document_name: str = ""
    document: str = ""
    top_p: float = 1.0
    temperature: float = 0.7
    personalai_prompt: str = ""
    assistant_id: str = ""
    thread_id: str = ""
    message_file_id: str = ""
    model_name: str = "gpt-4o-mini"

class BotRequest(BaseModel):
    """
    Model for the middleware /api/chat endpoint.
    """
    bot_name: str
    query_input: QueryInput

# ==============================================================================
# SECTION 7: API Endpoints
# This section defines the main /api/chat and /health endpoints.
# ==============================================================================

@app.post("/api/chat")
async def route_to_bot(request: BotRequest) -> dict:
    """
    Route incoming chat requests to the appropriate bot.
    Args:
        request (BotRequest): The incoming request containing bot name and query input.
    Returns:
        dict: The response from the bot.
    """
    logging.info("route_to_bot called")

    try:
        # Log the received payload as JSON
        logging.info(f"Received payload: {request.json()}")
    except Exception as e:
        logging.error(f"Exception occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    logging.info("Middleware received request")
    logging.info(f"Query Input Type: {type(request.query_input)}")
    logging.info(f"Query Input Content: {request.query_input.model_dump()}")

    # Normalize bot name to match keys in BOT_API_URLS
    bot_name = request.bot_name.lower().replace(" ", "_")

    if bot_name not in BOT_API_URLS:
        logging.error(f"Unsupported bot: {bot_name}")
        raise HTTPException(status_code=400, detail=f"Unsupported bot: {bot_name}")

    url = BOT_API_URLS[bot_name]

    async with httpx.AsyncClient() as client:
        # Debug: Check if bot is alive
        try:
            healthcheck = await client.get(url.replace("/api/v1", "/health"))
            logging.info(f"Healthcheck: {healthcheck.status_code} {healthcheck.text}")
        except Exception as e:
            logging.error(f"Healthcheck failed for {bot_name}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Healthcheck failed for {bot_name}: {str(e)}")

        try:
            logging.info(f"Sending to: {url}")
            logging.info("Payload received!")
            
            # Send the query input as JSON to the bot's API
            response = await client.post(
                url,
                json=request.query_input.model_dump(),
                timeout=30
            )
            response.raise_for_status()
            logging.info(f"Bot raw response: {response.text}")
            return response.json()
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Bot API request failed: {e.response.text}")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the middleware is running.
    Returns:
        dict: Status message.
    """
    return {"status": "ok", "message": "Middleware is running."}