import time
import requests
import streamlit as st
import v4.constants as const
import uuid
from v4.backend.retrieval_chains import get_source_info

# Set page configuration
st.set_page_config(page_title="Python Documentation Chatbot", layout="wide")

# Add header
st.markdown(
    "<h1 style='text-align: center;'>Python Documentation Chatbot</h1>",
    unsafe_allow_html=True,
)

# Initialize session list if not present
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session = None


# Function to create a new chat session
def create_new_session():
    session_id = str(uuid.uuid4())[:8]  # Generate a short unique session ID
    st.session_state.sessions[session_id] = {
        "messages": [],
        "sources": [],
        "show_sources": False,
    }
    st.session_state.active_session = session_id  # Set new session as active


# Function to delete a chat session
def delete_session(session_id):
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]
        # If the deleted session was active, set another session as active
        if st.session_state.active_session == session_id:
            if st.session_state.sessions:  # If there are remaining sessions
                st.session_state.active_session = next(iter(st.session_state.sessions))
            else:
                create_new_session()  # Ensure at least one session exists
        st.rerun()  # Force immediate UI update in latest Streamlit


# Ensure there is at least one session
if not st.session_state.sessions:
    create_new_session()

# Sidebar with session controls
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        """
        This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions about important Python libraries!
        """
    )

    # Button to create a new session
    if st.button("New Chat Session"):
        create_new_session()

    # Display all sessions with delete option
    st.markdown("### Active Sessions")
    to_delete = None  # Track which session to delete
    for session_id in list(st.session_state.sessions.keys()):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"Session {session_id}", key=f"switch_{session_id}"):
                st.session_state.active_session = session_id
        with col2:
            if st.button("❌", key=f"delete_{session_id}"):
                to_delete = session_id

    # Perform deletion after iteration to avoid modifying dict size during loop
    if to_delete:
        delete_session(to_delete)

    # Display the active session ID
    st.markdown(f"**Active Session ID:** `{st.session_state.active_session}`")

    # Button to clear conversation for the active session
    if st.button("Clear Conversation"):
        st.session_state.sessions[st.session_state.active_session] = {
            "messages": [],
            "sources": [],
            "show_sources": False,
        }

# Retrieve the active session
active_session = st.session_state.active_session
session_data = st.session_state.sessions[active_session]

# Display chat messages from history
for message in session_data["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about Python documentation..."):
    session_data["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        payload = {"query": prompt, "session_id": str(st.session_state.active_session)}
        print(payload)
        response = requests.post(const.FASTAPI_URL_QUERY, json=payload)
        response_data = response.json()
        response_answer = response_data["response"]["answer"]
        try:
            session_data["sources"] = get_source_info(
                response_data["response"]["context"],
                response_data["response"].get("input", ""),
                response_data["response"].get("answer", ""),
            )
        except:
            session_data["sources"] = []

    with st.chat_message("assistant"):
        try:
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.empty()
            message_placeholder.write(response_answer)
            session_data["messages"].append(
                {"role": "assistant", "content": response_answer}
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")
            session_data["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}",
                }
            )

# Toggle "Source" button
if session_data["sources"]:
    if st.button("Show Sources"):
        session_data["show_sources"] = not session_data["show_sources"]

    if session_data["show_sources"]:
        st.markdown("### Sources")
        for source in session_data["sources"]:
            st.markdown(
                f"**ID:** {source['id']}  \n"
                f"**Title:** {source['title']}  \n"
                f"**Source:** {source['source']}  \n"
                f"**Text Segment:** {source['segment']}  \n",
                unsafe_allow_html=True,
            )
