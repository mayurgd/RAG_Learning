import time
import uuid
import json
import sqlite3
import requests
import streamlit as st

from langchain_community.chat_message_histories import SQLChatMessageHistory

import v7.constants as const
from v7.backend.retrieval_chains import get_source_info

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


# Connect to SQLite database and fetch existing sessions
def load_sessions():
    try:
        conn = sqlite3.connect(const.CHAT_DB_LOC)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM message_store")
        session_ids = [row[0] for row in cursor.fetchall()]
    except:
        return
    for session_id in session_ids:
        cursor.execute(
            f"SELECT message FROM message_store WHERE session_id = '{session_id}'"
        )
        messages = [json.loads(row[0]) for row in cursor.fetchall()]
        formatted_messages = [
            {
                "role": "user" if msg["type"] == "human" else "assistant",
                "content": msg["data"]["content"],
            }
            for msg in messages
        ]
        st.session_state.sessions[session_id] = {
            "messages": formatted_messages,
            "sources": [],
            "show_sources": False,
        }
    if session_ids:
        st.session_state.active_session = session_ids[0]

    conn.close()


# Load sessions from DB
if not st.session_state.sessions:
    load_sessions()


# Function to create a new chat session
def create_new_session():
    session_id = str(uuid.uuid4())[:8]
    st.session_state.sessions[session_id] = {
        "messages": [],
        "sources": [],
        "show_sources": False,
    }
    st.session_state.active_session = session_id


# Function to delete a chat session
def delete_session(session_id):
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]
        if st.session_state.active_session == session_id:
            st.session_state.active_session = next(
                iter(st.session_state.sessions), None
            )
            if not st.session_state.active_session:
                create_new_session()
        # create sync sql message history by connection_string
        message_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{const.CHAT_DB_LOC}",
        )
        message_history.clear()

        st.rerun()


# Sidebar with session controls
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions about important Python libraries!"
    )

    if st.button("New Chat Session"):
        create_new_session()

    st.markdown("### Active Sessions")
    to_delete = None
    for session_id in list(st.session_state.sessions.keys()):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"Session {session_id}", key=f"switch_{session_id}"):
                st.session_state.active_session = session_id
        with col2:
            if st.button("❌", key=f"delete_{session_id}"):
                to_delete = session_id

    if to_delete:
        delete_session(to_delete)

    st.markdown(f"**Active Session ID:** `{st.session_state.active_session}`")

# Retrieve the active session
active_session = st.session_state.active_session
session_data = st.session_state.sessions.get(
    active_session, {"messages": [], "sources": [], "show_sources": False}
)

# Display chat messages
for message in session_data["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about Python documentation..."):
    session_data["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.spinner("Thinking..."):
        payload = {"query": prompt, "session_id": active_session}
        print(payload)
        response = requests.post(const.GENERATE_RESPONSE_URL, json=payload)
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
                time.sleep(0.01)
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
                f"**ID:** {source['id']}  \n**Title:** {source['title']}  \n**Source:** {source['source']}  \n**Text Segment:** {source['segment']}  \n",
                unsafe_allow_html=True,
            )
