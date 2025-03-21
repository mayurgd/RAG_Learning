from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

import v6.constants as const
from v6.logger import loggers_utils
from v6.backend.llm import get_chat_model
from v6.backend.file_handler import create_directories_for_path

logger = loggers_utils(__name__)

create_directories_for_path(const.CHAT_DB_LOC)


class RecreatedQuery(BaseModel):
    """Model for a single contextualized query."""

    recreated_query: str = Field(description="Recreated contextualized query")


class MultiQueries(BaseModel):
    """Model for multiple search queries."""

    recreated_query: str = Field(
        description="Input query using which LLM creates multiple queries"
    )
    query_1: str
    query_2: str
    query_3: str
    query_4: str


def get_doc_key(doc: Document) -> str:
    """Generates a unique key for a document based on its source and content hash."""
    source = doc.metadata.get("source", "")
    content_hash = hash(doc.page_content)
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


class RAGPipeline:
    """Retrieval-Augmented Generation (RAG) pipeline with query contextualization."""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

        # Define prompts
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given a chat history and the latest user question, formulate a standalone question that can be understood without the chat history.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.query_generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that generates multiple search queries based on a single input query.",
                ),
                (
                    "human",
                    "Generate multiple search queries related to: {recreated_query}",
                ),
                ("human", "OUTPUT 4 queries:"),
            ]
        )

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant for answering questions about Python libraries. Use the following pieces of retrieved context to answer the question. If you don't know the answer or don't have enough, say that you don't know. >>>> Context: {context} >>>>",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create chains
        self.query_creation_chain = (
            self.contextualize_q_prompt
            | self.llm.with_structured_output(RecreatedQuery)
            | self.query_generation_prompt
            | self.llm.with_structured_output(MultiQueries)
        )

        self.rag_chain = self.qa_prompt | self.llm | StrOutputParser()

    def run(self, input_query: str, session_id: str, k: int = 10) -> str:
        """Runs the full RAG pipeline using session_id to fetch chat history and commits new messages."""
        chat_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{const.CHAT_DB_LOC}",
            table_name="message_store",
        )

        # Fetch existing messages
        history_messages = chat_history.messages

        # Generate multiple queries
        multi_queries = self.query_creation_chain.invoke(
            {"input": input_query, "chat_history": history_messages}
        ).model_dump()

        # Retrieve documents using multiple queries
        all_results = {
            query: self.retriever.invoke(query)
            for query in list(multi_queries.values())[1:]
        }
        documents = reciprocal_rank_fusion(all_results)[:k]

        # Generate response
        response = self.rag_chain.invoke(
            {
                "input": multi_queries["recreated_query"],
                "chat_history": history_messages,
                "context": documents,
            }
        )

        # Store the conversation in chat history
        chat_history.add_message(HumanMessage(multi_queries["recreated_query"]))
        chat_history.add_message(AIMessage(response))

        return {
            "input": multi_queries["recreated_query"],
            "answer": response,
            "context": documents,
        }


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
