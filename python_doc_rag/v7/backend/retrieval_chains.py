import httpx
import asyncio
import requests

from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, AIMessage

import v7.constants as const
from v7.logger import loggers_utils
from v7.backend.llm import get_chat_model
from v7.backend.file_handler import create_directories_for_path
from v7.backend.models import MultiQueries, LLMResposne

logger = loggers_utils(__name__)

create_directories_for_path(const.CHAT_DB_LOC)


def get_doc_key(doc: Document) -> str:
    """Generates a unique key for a document based on its source and content hash."""
    source = doc["metadata"].get("source", "")
    content_hash = hash(doc["page_content"])
    return f"{source}_{content_hash}"


def reciprocal_rank_fusion(
    search_results_dict: Dict[str, List[Tuple[Document, float]]], k: int = 60
) -> List[Document]:
    """Performs Reciprocal Rank Fusion (RRF) on multiple search results."""

    fused_scores = {}

    for _, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(doc_scores):
            doc_key = get_doc_key(doc)  # Unique identifier for the document

            if doc_key not in fused_scores:
                fused_scores[doc_key] = [0, doc]  # Initialize score and store doc

            fused_scores[doc_key][0] += 1 / (rank + k)

    # Sort by fused score in descending order
    reranked_results = [
        data[1]  # Extract document
        for _, data in sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)
    ]

    return reranked_results


def create_multi_queries(query, session_id):

    chat_history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{const.CHAT_DB_LOC}",
        table_name="message_store",
    )

    formatted_chat_history = [
        (
            {"role": "human", "content": msg.content}
            if isinstance(msg, HumanMessage)
            else {"role": "ai", "content": msg.content}
        )
        for msg in chat_history.get_messages()
    ]

    data = {
        "query": [
            (
                "system",
                "You are an intelligent assistant specialized in refining user queries for better search results. "
                "If the user's question depends on previous chat history, rewrite it into a standalone query that makes sense independently. "
                "If the question is already self-contained, keep it unchanged. "
                "Then, based on the final query, generate three diverse but relevant search queries that maintain the original intent "
                "while covering different aspects, synonyms, and perspectives. "
                "Ensure the queries are well-formed, clear, and optimized for search engines.",
            ),
            ("human", "{input}"),
        ],
        "values": {"input": query},
        "structured_output": MultiQueries(),
        "chat_history": formatted_chat_history,
    }

    response = requests.post(const.INVOKE_LLM_URL, json=data)
    response_data = response.json()

    return response_data["response"]


async def retrieve_docs(query: str):
    """Asynchronous function to retrieve documents based on the query."""
    url = const.RETRIEVE_DOCS_URL
    data = {"query": query}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        return {query: response.json()}  # Return result mapped to query


def get_llm_response(query, docs):
    data = {
        "query": [
            (
                "system",
                "You are an assistant for answering questions about Python libraries. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer or don't have enough, say that you don't know. "
                ">>>> Context:\n\n```\n{context}\n``` \n\n>>>>",
            ),
            ("human", "{input}"),
        ],
        "values": {"input": query, "context": docs},
        "structured_output": LLMResposne(),
    }
    response = requests.post(const.INVOKE_LLM_URL, json=data)
    response_data = response.json()
    return response_data["response"]


async def main(query, session_id, top_n=10):
    fusion_queries = create_multi_queries(query=query, session_id=session_id)

    # Gather all requests asynchronously
    responses = await asyncio.gather(
        *[retrieve_docs(q) for q in fusion_queries.values()]
    )

    # Combine results into a dictionary
    retrieved_docs = {k: v[k]["docs"] for v in responses for k in v}

    reranked_docs = reciprocal_rank_fusion(retrieved_docs)[:top_n]

    return fusion_queries, reranked_docs


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


if __name__ == "__main__":
    fusion_queries, reranked_docs = asyncio.run(
        main(query="How to install it?", session_id="fa834781", top_n=10)
    )
    response = get_llm_response(fusion_queries["recreated_query"], reranked_docs)
