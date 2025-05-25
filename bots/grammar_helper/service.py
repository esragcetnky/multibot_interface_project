# ==============================================================================
# SECTION 1: Imports and Logging Setup
# This section imports required modules and sets up logging for the Ask Me Anything service.
# ==============================================================================
import yaml
import os
import logging
from openai import OpenAI

# Set up logging directory and file
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "grammar_helper.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Set to INFO to capture info logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

# ==============================================================================
# SECTION 2: Credentials and OpenAI Client Initialization
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
# SECTION 3: Grammar Correction Service
# This section defines the main function for answering questions using OpenAI API.
# ==============================================================================

def grammar_correction_service(query: str, model_name: str = "gpt-4o", temperature: float = 0.7, top_p: float = 1.0) -> str:
    """
    Grammar correction service that uses OpenAI's chat completion API to answer questions.
    This function sends a user's query to the OpenAI API and returns the response.

    Args:
        query (str): The user's question.
        model_name (str): The OpenAI model to use.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The answer from the OpenAI API or an error message.
    """
    logging.info(f"grammar_correction_service called with query='{query}', model_name='{model_name}', temperature={temperature}, top_p={top_p}")

    # Define the system prompt for the assistant
    system_prompt = (
        "You are a helpful assistant that provides grammar corrections and suggestions. "
        "Your task is to analyze the user's query and provide a corrected version if necessary. "
        "If the query is already grammatically correct, simply return it unchanged."
        "Only respond with the corrected text, without any additional explanations or comments."
    )

    try:
        logging.info("Sending request to OpenAI API...")
        # Send the request to OpenAI's chat completion endpoint
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
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
