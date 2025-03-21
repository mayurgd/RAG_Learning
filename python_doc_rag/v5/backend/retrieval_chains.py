import os
from typing import List
import v5.constants as const
from v5.logger import loggers_utils
from pydantic import BaseModel, Field
from v5.backend.llm import get_chat_model
from langchain_core.prompts import PromptTemplate
from v5.backend.vector_store import process_vector_store, VectorDbBM25Retriever
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

logger = loggers_utils(__name__)

os.makedirs(const.CHAT_DB_LOC.split("/")[0], exist_ok=True)


def retrieval_chain_with_session_memory():
    """Create a retrieval-augmented generation (RAG) chain with session memory."""
    logger.info("Initializing retrieval chain with session memory.")

    llm = get_chat_model()
    logger.info("Loaded chat model.")

    vector_store = process_vector_store()
    retriever = VectorDbBM25Retriever(vector_store, k=5)
    logger.info("Initialized retriever from vector store.")

    # Contextualize question based on chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. If the question is already self-contained, return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    logger.info("Created contextualization prompt template.")

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    logger.info("Initialized history-aware retriever.")

    # Answer question using retrieved context
    qa_system_prompt = (
        "You are an assistant for answering questions about Python libraries. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "If you don't have enough context, say that you don't know. "
        "Keep the answers concise."
        "\n\n"
        "Context: {context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    logger.info("Created QA prompt template.")

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    logger.info("Created question-answering chain.")

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    logger.info("Created retrieval chain.")

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{const.CHAT_DB_LOC}",
            table_name="message_store",
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    logger.info("Initialized conversational RAG chain.")
    return conversational_rag_chain


def get_source_info(docs_to_use, question, answer):
    """Extracts relevant document segments used to generate an answer."""
    logger.info("Initializing source information extraction.")

    llm = get_chat_model()

    # Data model
    class HighlightDocuments(BaseModel):
        """Return the specific part of a document used for answering the question."""

        id: List[str] = Field(
            ..., description="List of id of docs used to answer the question"
        )

        title: List[str] = Field(
            ..., description="List of titles used to answer the question"
        )

        source: List[str] = Field(
            ..., description="List of sources used to answer the question"
        )

        segment: List[str] = Field(
            ...,
            description="List of direct segments from used documents that answer the question",
        )

    # Parser
    parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

    # Prompt
    system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
    1. A question.
    2. A generated answer based on the question.
    3. A set of documents that were referenced in generating the answer.

    Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
    generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text 
    in the provided documents.

    Ensure that:
    - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
    - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
    - (Important) If you didn't use the specific document, don't mention it.

    Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{answer}</answer>

    <format_instruction>
    {format_instructions}
    </format_instruction>
    """

    prompt = PromptTemplate(
        template=system,
        input_variables=["documents", "question", "answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Chain
    doc_lookup = prompt | llm | parser

    def format_docs(docs):
        logger.info("Formatting documents for processing.")
        return "\n".join(
            f"<doc{i+1}>:\nTitle:{doc['metadata']['title']}\nSource:{doc['metadata']['source']}\nContent:{doc['page_content']}\n</doc{i+1}>\n"
            for i, doc in enumerate(docs)
        )

    # Run
    logger.info("Invoking document lookup.")
    lookup_response = doc_lookup.invoke(
        {
            "documents": format_docs(docs_to_use),
            "question": question,
            "answer": answer,
        }
    )
    logger.info("Received lookup response.")

    result = []
    for id, title, source, segment in zip(
        lookup_response.id,
        lookup_response.title,
        lookup_response.source,
        lookup_response.segment,
    ):
        result.append({"id": id, "title": title, "source": source, "segment": segment})

    logger.info("Extracted relevant document segments.")
    return result
