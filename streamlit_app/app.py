import streamlit as st
import os
import logging
from utils import generate_session_id, initialize_chat_history, append_to_chat_history, send_query_to_middleware, start_middleware_if_needed

middleware_port = start_middleware_if_needed()
middleware_url = f"http://localhost:{middleware_port}"

# Set up logging
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "streamlit_app.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True 
)

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="Multi-Bot Chat", page_icon="🤖")
st.title("🤖 Multi-Bot Chat Interface")

# -------------------------
# Sidebar: Bot Selection & Settings
# -------------------------
st.sidebar.title("Select Bot & Settings")

bot_name = st.sidebar.selectbox("Choose a Bot", ["Ask Me Anything", "Grammar Helper", "Compare Files", "Agreement Generator"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
model_name = st.sidebar.selectbox(
    "Choose OpenAI Model",
    [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "text-davinci-003",
    ],
    index=2  # default to gpt-3.5-turbo
)

# -------------------------
# Session State Initialization
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = generate_session_id()
    logging.info(f"Session ID initialized: {st.session_state.session_id}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = initialize_chat_history()
    logging.info("Chat history initialized.")

if "user_name" not in st.session_state:
    st.session_state.user_name = "test_user"  # Replace with real auth later
    logging.info(f"User name initialized: {st.session_state.user_name}")

# -------------------------
# Display Previous Messages
# -------------------------
for message in st.session_state.chat_history:
    logging.info(f"Displaying chat message: {message}")
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# User Input (New Message)
# -------------------------
if user_prompt := st.chat_input("Type your message..."):
    try:
        logging.info(f"User input received: {user_prompt}")
        # Add user message to history
        st.session_state.chat_history = append_to_chat_history(
            st.session_state.chat_history, "user", user_prompt
        )
        logging.info(f"User message appended to chat history: {user_prompt}")

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_prompt)
        logging.info("User message displayed in chat.")

        # Call middleware API
        access_key = "some_access_key"  # Replace with secure value if needed
        logging.info(
            f"Sending request to middleware: bot_name={bot_name}, user_name={st.session_state.user_name}, "
            f"session_id={st.session_state.session_id}, temperature={temperature}, top_p={top_p}, model_name={model_name}"
        )
        response = send_query_to_middleware(
            middleware_url=middleware_url,
            bot_name=bot_name,
            query=user_prompt,
            user_name=st.session_state.user_name,
            session_id=st.session_state.session_id,
            access_key=access_key,
            chat_history=st.session_state.chat_history,
            temperature=temperature,
            top_p=top_p,
            model_name=model_name,
            content_type="",
            document_name = "",
            document= "",    
            personalai_prompt = "",    
            assistant_id= "",    
            thread_id = "",
            message_file_id = ""
        )
        logging.info(f"Middleware response: {response}")

        # Display assistant response
        if "error" in response:
            assistant_reply = f" API Error: {response['error']}"
            logging.info(f"Assistant reply is an error: {assistant_reply}")
        else:
            assistant_reply = response.get("response", "No response from bot.")
            logging.info(f"Assistant reply received: {assistant_reply}")

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.markdown(assistant_reply)
        logging.info("Assistant reply displayed in chat.")

        st.session_state.chat_history = append_to_chat_history(
            st.session_state.chat_history, "assistant", assistant_reply
        )
        logging.info(f"Assistant reply appended to chat history: {assistant_reply}")
    except Exception as e:
        logging.error("Streamlit app error: %s", str(e), exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
