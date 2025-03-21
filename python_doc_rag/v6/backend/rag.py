from v6.logger import loggers_utils
from v6.backend.llm import get_chat_model
from v6.backend.retrieval_chains import RAGPipeline
from v6.backend.process_data import load_and_process_data
from v6.backend.vector_store import (
    create_bm25_index,
    create_vector_store,
    VectorDbBM25Retriever,
)

logger = loggers_utils(__name__)

cleaned_texts = load_and_process_data(chunk_size=1000, chunk_overlap=100)
vector_store = create_vector_store(docs=cleaned_texts)
bm25 = create_bm25_index(docs=cleaned_texts)

retriever = VectorDbBM25Retriever(
    vector_store=vector_store,
    bm25=bm25,
    bm25_corpus=cleaned_texts,
    k=10,
    do_bm25_search=True,
)

llm = get_chat_model()

chain = RAGPipeline(llm=llm, retriever=retriever)


def generate_response(query: str, session_id: str = None) -> str:
    """
    Generates a response to a given query using a retrieval-augmented chain.

    Args:
        query (str): The user's input query.
        session_id (str, optional): The session identifier.

    Returns:
        str: The generated response as a string.
    """
    logger.info(f"Generating response for session_id: {session_id}, query: {query}")
    try:
        response = chain.run(input_query=query, session_id=session_id, k=10)
        return response
    except Exception as e:
        raise
