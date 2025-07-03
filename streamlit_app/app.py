# ==============================================================================
# SECTION 1: Imports and Logging Setup
# ==============================================================================
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import logging
from components.utils import (
    generate_session_id,
    initialize_chat_history,
    append_to_chat_history,
    send_query_to_middleware,
    start_middleware_if_needed,
    prepare_chat_history_for_api,
    save_uploaded_file,
    clear_folder
)
import base64


# ==============================================================================
# SECTION 2: Page Setup
# ==============================================================================

st.set_page_config(page_title="Multi-Bot Chat", page_icon="🤖")
title = ("🤖 Multi-Bot Chat Interface")

@st.cache_resource
def get_middleware_url():
    port = start_middleware_if_needed()
    return f"http://localhost:{port}"
    

# ===============================================================================
# SECTION 3: Middleware URL and Logging Setup
# ===============================================================================
middleware_url = get_middleware_url()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Clear all log files in the logs directory
for fname in os.listdir(LOGS_DIR):
    if fname.endswith(".log"):
        with open(os.path.join(LOGS_DIR, fname), "w", encoding="utf-8") as f:
            pass  # Truncate the file

LOG_FILE = os.path.join(LOGS_DIR, "streamlit_app.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
    encoding="utf-8"
)


# ==============================================================================
# SECTION 7: File Upload Section
# ==============================================================================
if "clicked" not in st.session_state:
    st.session_state.clicked = False

def toggle_clicked():
    if st.session_state.clicked is True:
        st.session_state.clicked = False
    else:
        st.session_state.clicked = True

col1, col2 = st.columns([4,1], gap="large", vertical_alignment="bottom")
with col1:
    st.header(title)
with col2:
    if st.session_state.clicked is True:
        st.button("Close Files", on_click=toggle_clicked)
    else:
        st.button("Upload Files", on_click=toggle_clicked)

st.session_state.uploaded_document_path = []
st.session_state.uploaded_document_name = []

if st.session_state.clicked is True:
    uploaded_files = st.file_uploader("Please Upload First Document", accept_multiple_files=True)
    for uploaded_file in uploaded_files:        # Save file and get path and name using the utility function
        file_path, file_name = save_uploaded_file(
            uploaded_file,
            st.session_state.session_id, 
            st.session_state.bot_name.lower().replace(" ", "_"),
        )
        st.session_state.uploaded_document_name.append(file_name)
        st.session_state.uploaded_document_path.append(file_path)

# ==============================================================================
# SECTION 4: Session State Initialization
# ==============================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = generate_session_id()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = initialize_chat_history()
if "user_name" not in st.session_state:
    st.session_state.user_name = "test_user"


# ==============================================================================
# SECTION 5: Display Chat History
# ==============================================================================

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==============================================================================
# SECTION 6: Sidebar Form - Bot Selection & LLM Settings
# ==============================================================================

# Track last selected bot to detect change
if "last_bot_name" not in st.session_state:
    st.session_state.last_bot_name = "Ask Me Anything"

st.sidebar.title("Select Bot & Settings")
with st.sidebar.form("settings_form"):
    st.info(f"Session ID: \n{st.session_state.session_id if 'session_id' in st.session_state else 'Not set'}")
    selected_bot = st.selectbox(
        "Choose a Bot",
        ["Ask Me Anything", "Grammar Helper", "Compare Files", "Agreement Generator"],
        key="bot_name"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05, key="temperature")
    top_p = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05, key="top_p")
    model = st.selectbox(
        "Choose OpenAI Model",
        ["gpt-4.1","gpt-4o-mini", "gpt-4o", "text-davinci-003"],
        index=0,
        key="model_name"
    )
    # Save values in session_state (if not already)
    st.session_state.setdefault("bot_name", "Ask Me Anything")
    st.session_state.setdefault("temperature", 0.7)
    st.session_state.setdefault("top_p", 0.9)
    st.session_state.setdefault("model_name", "gpt-4o-mini")
    
    submit_settings = st.form_submit_button("Apply Settings")

    # If bot has changed, reset chat history
    if submit_settings:
        if selected_bot != st.session_state.last_bot_name:
            st.session_state.chat_history = initialize_chat_history()
            st.session_state.last_bot_name = selected_bot
        st.balloons()

st.sidebar.title("Vector DB Settings")
with st.sidebar.form("VectorDB"):
    st.info("Vector DB Settings")




# ==============================================================================
# SECTION 8: Handle New Message
# ==============================================================================

if user_prompt := st.chat_input("Type your message..."):
    try:
        # Append and display user message
        st.session_state.chat_history = append_to_chat_history(
            st.session_state.chat_history, "user", user_prompt
        )
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # Prepare chat history for API
        chat_history_for_api = prepare_chat_history_for_api(
            st.session_state.chat_history
        )
    
        # Show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_query_to_middleware(
                    middleware_url=middleware_url,
                    bot_name=st.session_state["bot_name"],
                    query=user_prompt,
                    user_name=st.session_state["user_name"],
                    session_id=st.session_state["session_id"],
                    access_key="some_access_key",
                    chat_history=chat_history_for_api,
                    temperature=st.session_state["temperature"],
                    top_p=st.session_state["top_p"],
                    model_name=st.session_state["model_name"],
                    content_type="",
                    document_name=st.session_state.uploaded_document_name,
                    document_path=st.session_state.uploaded_document_path, 
                    personalai_prompt="",
                    assistant_id="",
                    thread_id="",
                    message_file_id=""
                )
                if "error" in response:
                    assistant_reply = f"API Error: {response['error']}"
                else:
                    assistant_reply = response.get("response", "No response from bot.")
                st.markdown(assistant_reply)

        # Append assistant message
        st.session_state.chat_history = append_to_chat_history(
            st.session_state.chat_history, "assistant", assistant_reply
        )

    except Exception as e:
        logging.exception("Streamlit app error")
        st.error(f"An unexpected error occurred: {e}")












