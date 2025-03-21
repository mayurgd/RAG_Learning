from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from v4.backend.llm import get_chat_model
from v4.backend.vector_store import process_vector_store

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def retrieval_chain_with_session_memory():
    llm = get_chat_model()
    retriever = process_vector_store()

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

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question using retrieved context
    qa_system_prompt = (
        "You are an assistant for answering questions about Python libraries. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
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

    # Chain to generate final response
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create final retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


def get_source_info(docs_to_use, question, answer):

    llm = get_chat_model()

    # Data model
    class HighlightDocuments(BaseModel):
        """Return the specific part of a document used for answering the question."""

        id: List[str] = Field(
            ..., description="List of id of docs used to answers the question"
        )

        title: List[str] = Field(
            ..., description="List of titles used to answers the question"
        )

        source: List[str] = Field(
            ..., description="List of sources used to answers the question"
        )

        segment: List[str] = Field(
            ...,
            description="List of direct segements from used documents that answers the question",
        )

    # parser
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
    - (Important) If you didn't used the specific document don't mention it.

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
        return "\n".join(
            f"<doc{i+1}>:\nTitle:{doc['metadata']['title']}\nSource:{doc['metadata']['source']}\nContent:{doc['page_content']}\n</doc{i+1}>\n"
            for i, doc in enumerate(docs)
        )

    # Run
    lookup_response = doc_lookup.invoke(
        {
            "documents": format_docs(docs_to_use),
            "question": question,
            "answer": answer,
        }
    )

    result = []
    for id, title, source, segment in zip(
        lookup_response.id,
        lookup_response.title,
        lookup_response.source,
        lookup_response.segment,
    ):
        result.append({"id": id, "title": title, "source": source, "segment": segment})
    return result
