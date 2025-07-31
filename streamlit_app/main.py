# ==============================================================================
# SECTION 1: Imports and Logging Setup
# ==============================================================================
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import logging
import base64
import requests

from streamlit_app.utils import (
    generate_session_id,
    initialize_chat_history,
    append_to_chat_history,
    send_query_to_middleware,
    start_middleware_if_needed,
    prepare_chat_history_for_api,
    save_uploaded_file,
    clear_folder,
    start_vector_db_api_if_needed
)

# ==============================================================================
# SECTION 2: Page Setup
# ==============================================================================
st.set_page_config(page_title="Multi-Bot Chat", page_icon="🤖")
title = ("🤖 Multi-Bot Chat Interface")

# ==============================================================================
# SECTION 3: Caching Middleware and Vector DB URLs
# ==============================================================================
@st.cache_resource
def get_middleware_url():
    port = start_middleware_if_needed()
    return f"http://localhost:{port}"

@st.cache_resource
def get_vector_db_api_url():
    port = start_vector_db_api_if_needed()
    return f"http://localhost:{port}"

middleware_api_port_num = get_middleware_url()
vectordb_api_port_num = get_vector_db_api_url()

# ==============================================================================
# SECTION 4: Setup Paths and Logging
# ==============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
VECTORSTORES_DIR = os.path.join(DATA_DIR, "vectorstores")

for d in [LOGS_DIR, DATA_DIR, UPLOADS_DIR, VECTORSTORES_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "streamlit_app.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
    encoding="utf-8"
)

# ==============================================================================
# SECTION 5: Session State Initialization
# ==============================================================================
def init_session():
    defaults = {
        "clicked": False,
        "session_id": generate_session_id(),
        "chat_history": initialize_chat_history(),
        "user_name": "test_user",
        "bot_name": "Ask Me Anything",
        "temperature": 0.7,
        "top_p": 0.9,
        "model_name": "gpt-4.1",
        "uploaded_document_path": [],
        "uploaded_document_name": [],
        "last_selected_bot": "Ask Me Anything"
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

init_session()

st.header(title)


# ==============================================================================
# SECTION 7: Chat History
# ==============================================================================
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


@st.dialog("Upload Files")
def upload_file():
    """Function to handle file upload dialog."""    
    uploaded_files = st.file_uploader("Please Upload First Document", accept_multiple_files=True)
    with st.spinner("Wait for it...", show_time=True):
        for uploaded_file in uploaded_files:
            file_path, file_name = save_uploaded_file(
                uploaded_file,
                st.session_state.session_id, 
                st.session_state.bot_name.lower().replace(" ", "_"),
            )
            # Prepare lists for the unified API
            payload = {
                "vector_db_path": vector_db_path,
                "document_names": [file_name],
                "document_paths": [file_path]
            }
            response = requests.post(
                f"{vectordb_api_port_num}/api/vectordb/add",
                json=payload
            )
            st.session_state.uploaded_document_name.append(file_name)
            st.session_state.uploaded_document_path.append(file_path)
            # Refresh document list
            response = requests.get(
                f"{vectordb_api_port_num}/api/vectordb/list",
                params={"vector_db_path": vector_db_path}
            )
            st.session_state.vector_db_documents_list = response.json().get("documents", [])



# ==============================================================================
# SECTION 8: Sidebar Controls
# ==============================================================================
with st.sidebar:
    st.sidebar.info(f"Session ID: \n{st.session_state.session_id}")

    selected_bot = st.selectbox(
        "Choose a Bot",
        ["Ask Me Anything", "Grammar Helper", "Compare Files", "Agreement Generator"],
        key="bot_name"
    )

    # Clear chat if bot changes
    if selected_bot != st.session_state.last_selected_bot:
        st.session_state.chat_history = initialize_chat_history()
        st.session_state.session_id = generate_session_id()
        st.session_state.last_selected_bot = selected_bot

    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.top_p, 0.05)
    st.session_state.model_name = st.selectbox(
        "Choose OpenAI Model",
        ["gpt-4.1", "gpt-4o-mini", "gpt-4o", "text-davinci-003"],
        index=0
    )

    vector_db_path = os.path.join(DATA_DIR, "vectorstores", st.session_state.bot_name.lower().replace(" ", "_"))
    st.info(vector_db_path)

    try:
        response = requests.get(
            f"{vectordb_api_port_num}/api/vectordb/list",
            params={"vector_db_path": vector_db_path}
        )
        st.session_state.vector_db_documents_list = response.json().get("documents", [])
    except Exception as e:
        logging.exception("Error listing documents in Vector DB")
        st.error(f"Error listing documents: {e}")
        st.session_state.vector_db_documents_list = []

    selected_docs = st.multiselect("Documents", st.session_state.vector_db_documents_list)

    if st.button("Add New Document", use_container_width=True):
        st.session_state.clicked = True
        upload_file()


    if st.button("Delete Selected Document", use_container_width=True):
        if selected_docs:
            try:
                payload = {
                    "vector_db_path": vector_db_path,
                    "document_names": selected_docs  # This is already a list
                }
                response = requests.delete(
                    f"{vectordb_api_port_num}/api/vectordb/delete",
                    json=payload
                )
                st.success("Selected documents deleted.")
                # Refresh document list
                response = requests.get(
                    f"{vectordb_api_port_num}/api/vectordb/list",
                    params={"vector_db_path": vector_db_path}
                )
                st.session_state.vector_db_documents_list = response.json().get("documents", [])
            except Exception as e:
                logging.exception("Error deleting documents")
                st.error(f"Delete error: {e}")

    if st.button("Clear Vector DB", use_container_width=True):
        try:
            response = requests.post(
                f"{vectordb_api_port_num}/api/vectordb/clear",
                json={"vector_db_path": vector_db_path}
            )
            if response.status_code == 200:
                st.success("Vector DB cleared successfully.")
                # Refresh document list
                response = requests.get(
                    f"{vectordb_api_port_num}/api/vectordb/list",
                    params={"vector_db_path": vector_db_path}
                )
                st.session_state.vector_db_documents_list = response.json().get("documents", [])
        except Exception as e:
            logging.exception("Error clearing Vector DB")
            st.error(f"Error clearing DB: {e}")

# ==============================================================================
# SECTION 9: Chat Input Handling
# ==============================================================================
if user_prompt := st.chat_input("Type your message..."):
    try:
        st.session_state.chat_history = append_to_chat_history(
            st.session_state.chat_history, "user", user_prompt
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)

        chat_history_for_api = prepare_chat_history_for_api(st.session_state.chat_history)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_query_to_middleware(
                    middleware_url=middleware_api_port_num,
                    bot_name=st.session_state.bot_name,
                    query=user_prompt,
                    user_name=st.session_state.user_name,
                    session_id=st.session_state.session_id,
                    access_key="some_access_key",
                    chat_history=chat_history_for_api,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    model_name=st.session_state.model_name,
                    content_type="",
                    personalai_prompt="",
                    assistant_id="",
                    thread_id="",
                    message_file_id=""
                )
                assistant_reply = response.get("response", "No response from bot.") if "error" not in response else f"API Error: {response['error']}"
                st.markdown(assistant_reply)

        st.session_state.chat_history = append_to_chat_history(
            st.session_state.chat_history, "assistant", assistant_reply
        )

    except Exception as e:
        logging.exception("Streamlit app error")
        st.error(f"An unexpected error occurred: {e}")