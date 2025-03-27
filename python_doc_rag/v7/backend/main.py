import uuid
import asyncio

from contextvars import ContextVar
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

import v7.constants as const
from v7.backend.llm import get_chat_model
from v7.backend.rag import generate_response
from v7.logger import setup_logging, loggers_utils
from v7.backend.file_handler import create_directories_for_path
from v7.backend.retrieval_chains import main, get_llm_response
from v7.backend.models import (
    RAGQuery,
    LLMQuery,
    RetrieverQuery,
    create_dynamic_pydantic_model,
)
from v7.backend.process_data import load_and_process_data
from v7.backend.vector_store import (
    create_bm25_index,
    create_vector_store,
    VectorDbBM25Retriever,
)
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

create_directories_for_path(const.LOG_FILE_LOC)

# Create a context variable for correlation ID
correlation_id_ctx_var = ContextVar("correlation_id", default=str(uuid.uuid4()))
# Generate and set a new correlation ID for each log
correlation_id_ctx_var.set(str(uuid.uuid4()))

app = FastAPI()
setup_logging(correlation_id_ctx_var, log_to_file=True, log_file=const.LOG_FILE_LOC)
logger = loggers_utils(__name__)
llm = get_chat_model()

cleaned_texts = load_and_process_data(chunk_size=500, chunk_overlap=200)
vector_store = create_vector_store(
    docs=cleaned_texts,
    # embedding_model=OpenAIEmbeddings(model="text-embedding-3-large")
    embedding_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
)
bm25 = create_bm25_index(docs=cleaned_texts)
k = 10
retriever = VectorDbBM25Retriever(
    vector_store=vector_store,
    bm25=bm25,
    bm25_corpus=cleaned_texts,
    k=k,
    do_bm25_search=True,
)


@app.post("/generate-response/")
async def generate_response(request: RAGQuery):
    logger.info(
        "Received request",
        extra={"session_id": request.session_id, "query": request.query},
    )
    try:

        session_id = request.session_id
        query = request.query

        chat_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{const.CHAT_DB_LOC}",
            table_name="message_store",
        )
        logger.info("Generating response", extra={"session_id": session_id})

        fusion_queries, reranked_docs = await main(
            query=query, session_id=session_id, top_n=k
        )
        print(query)
        print(fusion_queries)
        # Ensure 'recreated_query' exists in fusion_queries to avoid KeyError
        recreated_query = fusion_queries.get("recreated_query", "")

        # Call the synchronous function in a separate thread
        response = await asyncio.to_thread(
            get_llm_response, recreated_query, reranked_docs
        )

        response["context"] = reranked_docs

        logger.info(
            "Response generated successfully",
            extra={"session_id": session_id, "response": response.get("answer", "")},
        )

        # Store the conversation in chat history
        chat_history.add_message(HumanMessage(fusion_queries["recreated_query"]))
        chat_history.add_message(AIMessage(response["answer"]))

        return {
            "query": request.query,
            "response": response,
        }

    except Exception as e:
        logger.error(
            "Error during query processing",
            exc_info=True,
            extra={"session_id": request.session_id},
        )
        raise HTTPException(
            status_code=500, detail="Internal server error. Please try again later."
        )


@app.post("/invoke-llm/")
async def invoke_llm(request: LLMQuery):
    query = request.query
    structured_output = request.structured_output
    chat_history = request.chat_history

    # Reconstruct ChatPromptTemplate
    if chat_history:
        all_messages = (
            [query[0]] + [(msg.role, msg.content) for msg in chat_history] + query[1:]
        )
    else:
        all_messages = query

    chat_prompt = ChatPromptTemplate.from_messages(all_messages)
    formatted_prompt = chat_prompt.format(**request.values)
    # Apply structured output formatting if provided
    if structured_output:
        Model = create_dynamic_pydantic_model("DynamicOutput", structured_output)
        llm_with_structure = llm.with_structured_output(Model)
    else:
        llm_with_structure = llm

    # Invoke LLM
    print("=" * 50)
    try:
        response = llm_with_structure.invoke(formatted_prompt)
    except:
        response = llm.invoke(formatted_prompt)
    print("+" * 50)
    print(response)
    return {
        "query": request.query,
        "formatted_prompt": formatted_prompt,
        "response": response,
    }


@app.post("/retrieve-docs/")
async def retrieve_docs(request: RetrieverQuery):
    query = request.query
    docs = retriever.invoke(query)
    return {"docs": docs}
