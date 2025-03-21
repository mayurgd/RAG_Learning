from v5.logger import loggers_utils
from v5.backend.retrieval_chains import retrieval_chain_with_session_memory

logger = loggers_utils(__name__)
retriever = retrieval_chain_with_session_memory()


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
        response = retriever.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        return response
    except Exception as e:
        raise
