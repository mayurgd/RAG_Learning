import os
import requests
import streamlit as st

from zipfile import ZipFile
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

_ = load_dotenv(find_dotenv())


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def configure_retriever():
    # Read documents
    docs = TextLoader(
        "python_3_13_changes.txt",
        encoding="utf8",
    ).load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    vectordb = Chroma.from_documents(
        splits, GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    return retriever


@st.cache_data
def create_prompt():
    template = """
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        Question: {question}
        You assist user queries based on : {context}
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        input_variables=["question", "context"], template="{question}"
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt_template


def generate_response(retriever, query):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)


st.title("Python Document Chat")

retriever = configure_retriever()
chat_prompt_template = create_prompt()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = generate_response(retriever, prompt)
        full_response += response
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
