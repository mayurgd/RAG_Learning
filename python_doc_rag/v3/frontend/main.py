import time
import requests
import streamlit as st
import v3.constants as const
from v3.backend.retrieval_chains import get_source_info

# Set page configuration
st.set_page_config(page_title="Python Documentation Chatbot", layout="wide")
# Add header
st.markdown(
    "<h1 style='text-align: center;'>Python Documentation Chatbot</h1>",
    unsafe_allow_html=True,
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
# Initialize session state for sources if it doesn't exist
if "sources" not in st.session_state:
    st.session_state.sources = []
# Initialize session state for source visibility
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False  # Default to hidden

# Sidebar with information
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        """
        This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions about important python libraries!
        """
    )

    # Add a button to clear chat history
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.show_sources = False  # Hide sources on reset
# Display chat messages from history first
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Handle user input
if prompt := st.chat_input("Ask about Python documentation..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat container
    with st.chat_message("user"):
        st.write(prompt)

    # Add a spinner while getting response
    with st.spinner("Thinking..."):
        payload = {"query": prompt}
        # Get response from retrieval chain
        response = requests.post(const.FASTAPI_URL_QUERY, json=payload)
        response_data = response.json()
        response_answer = response_data["response"]["answer"]
        # Store sources in session state
        st.session_state.sources = get_source_info(
            response_data["response"]["context"],
            response_data["response"].get("rewritten_input", ""),
            response_data["response"].get("answer", ""),
        )

    # Display assistant response
    with st.chat_message("assistant"):
        try:
            # Simulate stream of response with milliseconds
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.empty()
            message_placeholder.write(response_answer)
            # Add assistant response to chat history AFTER displaying it
            st.session_state.messages.append(
                {"role": "assistant", "content": response_answer}
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}",
                }
            )


# **Newly Added: Toggle "Source" button**
if st.session_state.sources:
    if st.button("Show Sources"):
        st.session_state.show_sources = (
            not st.session_state.show_sources
        )  # Toggle visibility

    # Display sources only if show_sources is True
    if st.session_state.show_sources:
        st.markdown("### Sources")
        for source in st.session_state.sources:
            st.markdown(
                f"**ID:** {source['id']}  \n"
                f"**Title:** {source['title']}  \n"
                f"**Source:** {source['source']}  \n"
                f"**Text Segment:** {source['segment']}  \n",
                unsafe_allow_html=True,
            )
