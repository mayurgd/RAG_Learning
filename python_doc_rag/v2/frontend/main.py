import time
import requests
import streamlit as st
import v2.constants as const

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
        if "messages" in st.session_state:
            st.session_state.messages = []  # Changed from del to empty list

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
        response_answer = response_data["response"]

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
            message_placeholder.markdown(full_response)

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
