# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the middleware.
# ==============================================================================

# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
import asyncio
import subprocess
import socket
import threading
import json
import traceback

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ==============================================================================
# SECTION 2: Logging Configuration
# This section configures logging for the middleware.
# ==============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "middleware.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
    encoding="utf-8"
)

# ==============================================================================
# SECTION 3: FastAPI App and Middleware Setup
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
# SECTION 4: Bot Management Configuration
# ==============================================================================

BOTS = {
    "ask_me_anything": "uvicorn bots.ask_me_anything.main:app --host 0.0.0.0 --port {port}",
    "grammar_helper": "uvicorn bots.grammar_helper.main:app --host 0.0.0.0 --port {port}",
    "compare_files": "uvicorn bots.compare_files.main:app --host 0.0.0.0 --port {port}",
    "agreement_generator": "uvicorn bots.agreement_generator.main:app --host 0.0.0.0 --port {port}",
    # Add more bots here as needed.
}

bot_processes = {}
BOT_API_URLS = {}

# ==============================================================================
# SECTION 5: Utility Functions
# ==============================================================================

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
        logging.info(f"Found free port: {port}")
        return port

def log_subprocess_output(pipe):
    """Log output from a subprocess (bot) to the middleware log."""
    for line in iter(pipe.readline, b''):
        decoded_line = line.decode().rstrip()
        logging.info(f"[BOT OUTPUT] {decoded_line}")
        print(f"[BOT OUTPUT] {decoded_line}")

async def start_bot(bot_name):
    """Start a bot subprocess if not already running."""
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

    threading.Thread(target=log_subprocess_output, args=(process.stdout,), daemon=True).start()
    threading.Thread(target=log_subprocess_output, args=(process.stderr,), daemon=True).start()

    bot_processes[bot_name] = process
    BOT_API_URLS[bot_name] = f"http://localhost:{port}/api/v1"
    logging.info(f"Bot '{bot_name}' API URL set to {BOT_API_URLS[bot_name]}")

async def start_all_bots():
    """Start all bots defined in the BOTS dictionary."""
    tasks = []
    logging.info("Starting all bots...")
    for bot_name in BOTS:
        logging.info(f"Scheduling startup for bot: {bot_name}")
        tasks.append(start_bot(bot_name))
    await asyncio.gather(*tasks)
    logging.info("All bots have been started.")

# ==============================================================================
# SECTION 6: Middleware and Exception Handlers
# ==============================================================================

@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    return await call_next(request)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

# ==============================================================================
# SECTION 7: FastAPI Startup Event
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    logging.info("FastAPI startup event triggered. Starting all bots...")
    await start_all_bots()
    await asyncio.sleep(2)
    logging.info("Startup event completed. Bots should be ready.")

# ==============================================================================
# SECTION 8: Pydantic Models
# ==============================================================================

class QueryInput(BaseModel):
    query: str
    user_name: str
    session_id: str
    access_key: str
    chat_history: list = []
    content_type: str = ""
    top_p: float = 1.0
    temperature: float = 0.7
    personalai_prompt: str = ""
    assistant_id: str = ""
    thread_id: str = ""
    message_file_id: str = ""
    model_name: str = "gpt-4o-mini"

class BotRequest(BaseModel):
    bot_name: str
    query_input: QueryInput

# ==============================================================================
# SECTION 9: API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-Bot Middleware API. Use /docs for API documentation."}

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Middleware is running."}

@app.post("/api/chat")
async def route_to_bot(request: BotRequest) -> dict:
    logging.info("route_to_bot called")

    try:
        logging.info(f"Received payload: {request.json()}")
    except Exception as e:
        logging.error(f"Exception occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    logging.info("Middleware received request")
    logging.info(f"Query Input Type: {type(request.query_input)}")
    logging.info(f"Query Input Content: {request.query_input.model_dump()}")

    bot_name = request.bot_name.lower().replace(" ", "_")

    if bot_name not in BOT_API_URLS:
        logging.error(f"Unsupported bot: {bot_name}")
        raise HTTPException(status_code=400, detail=f"Unsupported bot: {bot_name}")

    url = BOT_API_URLS[bot_name]

    async with httpx.AsyncClient() as client:
        try:
            healthcheck = await client.get(url.replace("/api/v1", "/health"))
            logging.info(f"Healthcheck: {healthcheck.status_code} {healthcheck.text}")
        except Exception as e:
            logging.error(f"Healthcheck failed for {bot_name}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Healthcheck failed for {bot_name}: {str(e)}")

        try:
            logging.info(f"Sending to: {url}")
            logging.info("Payload received!")

            # Forward all fields, including document and document_path
            response = await client.post(
                url,
                json=request.query_input.model_dump(),
                timeout=180,
                headers={"Content-Type": "application/json"}
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


