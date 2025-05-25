import yaml
import os
import logging
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Set up logging
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "grammar_helper.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Set to INFO to capture info logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

# Load credentials
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "shared", "credentials.yml")

logging.info(f"Loading credentials from {CREDENTIALS_PATH}")
with open(CREDENTIALS_PATH, "r") as file:
    credentials = yaml.safe_load(file)
logging.info("Credentials loaded successfully.")

client = OpenAI(api_key=credentials["openai_api_key"])
logging.info("OpenAI client initialized.")

def grammar_correction_service(query: str, model_name: str = "gpt-4o", temperature: float = 0.7, top_p: float = 1.0) -> str:
    """
    Correct grammar using OpenAI v1+ API client.
    """
    logging.info(f"grammar_correction_service called with query='{query}', model_name='{model_name}', temperature={temperature}, top_p={top_p}")

    system_prompt = (
        "You are a helpful assistant that corrects grammar, punctuation, and style errors. "
        "Return only the corrected version of the text, without explanations."
    )

    try:
        logging.info("Sending request to OpenAI API...")
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
        result = response.choices[0].message.content.strip()
        logging.info(f"OpenAI API result: {result}")
        return result
    except Exception as e:
        logging.error("Error calling OpenAI API: %s", str(e), exc_info=True)
        return f"Error calling OpenAI API: {str(e)}"
