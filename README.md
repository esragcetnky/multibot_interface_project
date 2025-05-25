# Multi-Bot Platform

A modular AI platform featuring multiple bots accessed via a Streamlit frontend and FastAPI backend middleware. Users can choose different bots such as `askmeanything` and `grammar_helper`, with history management and customizable parameters like temperature.

---

## Project Architecture

```
multi_bot_platform/
│
├── env/                        # Python virtual environment folder
│
├── streamlit_app/
│   ├── app.py                  # Main Streamlit frontend interface
│   └── utils.py                # Optional helper functions
│
├── middleware/
│   └── main.py                 # FastAPI middleware routing requests to bots
│
├── bots/
│   ├── askmeanything/
│   │   ├── main.py             # FastAPI app for AskMeAnything bot
│   │   └── service.py          # Logic for calling LLM and processing
│   │
│   ├── grammar_helper/
│   │   ├── main.py             # FastAPI app for GrammarHelper bot
│   │   └── service.py          # Logic for calling LLM and processing
│   │
│   └── shared/
│       ├── credentials.yml     # API keys and configuration for all bots
│       └── llm_utils.py        # Shared utility functions for LLM calls
│
├── logs/
│   └── graylog_config.json     # Graylog logging configuration (optional)
│
├── requirements.txt            # Python dependencies for entire project
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

---

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://your-repo-url.git
   cd multi_bot_platform
   ```

2. **Create and activate Python virtual environment**

   ```bash
   python3 -m venv env
   source env/bin/activate       # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure credentials**

   Add your API keys and any other sensitive configuration in:

   ```
   bots/shared/credentials.yml
   ```

5. **Run FastAPI Middleware**

   From the project root:

   ```bash
   uvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Run Streamlit Interface**

   In another terminal (with the environment activated):

   ```bash
   streamlit run streamlit_app/app.py
   ```

---

## Features

- **Multiple bots** accessible via a sidebar selector.
- **Session management** with auto-generated session IDs.
- **Chatbot-like UI** with persistent chat history per bot.
- **Temperature and other LLM parameters** adjustable in sidebar.
- **Middleware** to route requests from frontend to appropriate bot API.
- **Centralized logging** with optional Graylog integration.

---

## Notes

- The virtual environment folder `env/` is included in `.gitignore` and should not be committed.
- Logs are stored in the `logs/` folder.
- Credentials and sensitive info must be kept private.

---

## Future Enhancements

- Add more bots under `bots/`
- Improve error handling and retry mechanisms
- Enhance UI/UX with richer chat features and file uploads
- Containerize with Docker for easier deployment

---

## License

Specify your license here.

---

## Contact

For questions or contributions, please open an issue or contact the maintainer.
