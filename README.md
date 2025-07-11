# Multi-Bot Interface Platform

A modular AI platform featuring multiple bots accessed via a Streamlit frontend and FastAPI backend middleware. Users can choose different bots such as `Ask Me Anything`, `Grammar Helper`, `Compare Files`, and `Agreement Generator`, with history management, document upload, retrieval-augmented generation (RAG), and customizable parameters like temperature.

---

![Platform Architecture](data/image.png)

## Project Structure

```
multibot_interface_project/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── bots/
│   ├── agreement_generator/
│   │   ├── main.py
│   │   └── service.py
│   ├── ask_me_anything/
│   │   ├── main.py
│   │   └── service.py
│   ├── compare_files/
│   │   ├── main.py
│   │   └── service.py
│   └── grammar_helper/
│       ├── main.py
│       └── service.py
│
├── vector_db/
│   ├── main.py
│   └── utils.py
│
├── data/
│   ├── image.png
│   ├── uploads/
│   └── vectorstores/
│
├── env/
│   ├── pyvenv.cfg
│   ├── etc/
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── share/
│
├── logs/
│   ├── ask_me_anything.log
│   ├── compare_files.log
│   ├── faiss_db.log
│   ├── grammar_helper.log
│   ├── middleware.log
│   └── streamlit_app.log
│
├── middleware/
│   └── main.py
│
├── shared/
│   └── credentials.yml
│
└── streamlit_app/
    ├── main.py
    └── utils.py
```

---

## Setup Instructions

1. **Clone the repository**

   ```sh
   git clone https://your-repo-url.git
   cd multibot_interface_project
   ```

2. **Create and activate Python virtual environment**

   ```sh
   python3 -m venv env
   # On Linux/Mac:
   source env/bin/activate
   # On Windows:
   env\Scripts\activate
   ```

3. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Configure credentials**

   Add your API keys and any other sensitive configuration in:

   ```
   shared/credentials.yml
   ```

   Example:
   ```yaml
   openai_api_key: "sk-..."
   tavily_api_key: "tavily-..."
   ```

5. **Run FastAPI Middleware**

   From the project root:

   ```sh
   uvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Run Vector DB API**

   In another terminal (with the environment activated):

   ```sh
   uvicorn vector_db.main:app --reload --host 0.0.0.0 --port 5000
   ```

7. **Run Streamlit Interface**

   In another terminal (with the environment activated):

   ```sh
   streamlit run streamlit_app/main.py
   ```

---

## Features

- **Multiple bots** accessible via a sidebar selector:
  - Ask Me Anything (RAG, file upload, OpenAI)
  - Grammar Helper (RAG, file upload, OpenAI)
  - Compare Files (file comparison)
  - Agreement Generator (contract/summary generation)
- **Session management** with auto-generated session IDs.
- **Chatbot-like UI** with persistent chat history per bot.
- **File/document upload** with support for text, PDF, DOCX, etc.
- **Retrieval-Augmented Generation (RAG)** using FAISS vector DBs.
- **Web search integration** via Tavily API.
- **Temperature and other LLM parameters** adjustable in sidebar.
- **Middleware** to route requests from frontend to appropriate bot API.
- **Centralized logging** in the `logs/` folder for each component.
- **Vector DB microservice** for CRUD operations on document vector stores.

---

## Notes

- The virtual environment folder `env/` is included in `.gitignore` and should not be committed.
- Logs are stored in the `logs/` folder:
  - `middleware.log` for middleware events and errors
  - `streamlit_app.log` for frontend events and errors
  - `ask_me_anything.log`, `grammar_helper.log`, `compare_files.log`, `faiss_db.log` for bot/component logs
- Credentials and sensitive info must be kept private in `shared/credentials.yml`.
- Uploaded files are stored in `data/uploads/` and indexed for retrieval.
- Vector DBs are stored in `data/vectorstores/`.
- Vector DB API code is in `vector_db/` and provides endpoints for document management.

---

## Future Enhancements

- Implement more advanced logic for all bots in their respective `service.py` and `main.py` files.
- Add more bots under `bots/`
- Improve error handling and retry mechanisms
- Enhance UI/UX with richer chat features and file uploads
- Containerize with Docker for easier deployment
- Add user authentication and access control

---

## License

Specify your license here.

---

## Contact

For any questions or suggestions, feel free to reach out:

  - Email: esragcetinkaya@gmail.com
  - Linkedin: [esragcetinkaya](https://www.linkedin.com/in/esra-gul-cetinkaya/?locale=en_US)
