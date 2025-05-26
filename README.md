# Multi-Bot Interface Platform

A modular AI platform featuring multiple bots accessed via a Streamlit frontend and FastAPI backend middleware. Users can choose different bots such as `Ask Me Anything`, `Grammar Helper`, `Compare Files`, and `Agreement Generator`, with history management and customizable parameters like temperature.

---
![alt text](data/image.png)
## Project Architecture

```
multibot_interface_project/
│
├── env/                        # Python virtual environment folder
│
├── streamlit_app/
│   ├── app.py                  # Main Streamlit frontend interface
│   └── utils.py                # Helper functions for Streamlit app
│
├── middleware/
│   └── main.py                 # FastAPI middleware routing requests to bots
│
├── bots/
│   ├── ask_me_anything/
│   │   ├── main.py             # FastAPI app for Ask Me Anything bot
│   │   └── service.py          # Logic for calling LLM and processing
│   │
│   ├── grammar_helper/
│   │   ├── main.py             # FastAPI app for Grammar Helper bot
│   │   └── service.py          # Logic for calling LLM and processing
│   │
│   ├── compare_files/
│   │   ├── main.py             # FastAPI app for Compare Files bot
│   │   └── service.py          # Logic for file comparison
│   │
│   └── agreement_generator/
│       ├── main.py             # FastAPI app for Agreement Generator bot
│       └── service.py          # Logic for agreement generation
│
├── shared/
│   └── credentials.yml         # API keys and configuration for all bots
│
├── logs/
│   ├── grammar_helper.log      # Logs for Grammar Helper bot
│   ├── middleware.log          # Logs for Middleware
│   └── streamlit_app.log       # Logs for Streamlit app
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
   cd multibot_interface_project
   ```

2. **Create and activate Python virtual environment**

   ```bash
   python3 -m venv env
   # On Linux/Mac:
   source env/bin/activate
   # On Windows:
   env\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure credentials**

   Add your API keys and any other sensitive configuration in:

   ```
   shared/credentials.yml
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

- **Multiple bots** accessible via a sidebar selector:
  - Ask Me Anything
  - Grammar Helper
  - Compare Files
  - Agreement Generator
- **Session management** with auto-generated session IDs.
- **Chatbot-like UI** with persistent chat history per bot.
- **Temperature and other LLM parameters** adjustable in sidebar.
- **Middleware** to route requests from frontend to appropriate bot API.
- **Centralized logging** in the `logs/` folder for each component.

---

## Notes

- The virtual environment folder `env/` is included in `.gitignore` and should not be committed.
- Logs are stored in the `logs/` folder:
  - `middleware.log` for middleware events and errors
  - `streamlit_app.log` for frontend events and errors
  - `grammar_helper.log` for Grammar Helper bot logs
- Credentials and sensitive info must be kept private in `shared/credentials.yml`.

---

## Future Enhancements

- Implement logic for all bots in their respective `service.py` and `main.py` files.
- Add more bots under `bots/`
- Improve error handling and retry mechanisms
- Enhance UI/UX with richer chat features and file uploads
- Containerize with Docker for easier deployment

---

## License

Specify your license here.

---

## Contact

For any questions or suggestions, feel free to reach out:

  - Email: esragcetinkaya@gmail.com
  - Linkedin : [esragcetinkaya](https://www.linkedin.com/in/esra-gul-cetinkaya/?locale=en_US)
