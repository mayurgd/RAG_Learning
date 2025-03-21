from v4.backend.retrieval_chains import retrieval_chain_with_session_memory

retriever = retrieval_chain_with_session_memory()


def generate_response(query: str, session_id: str = None) -> str:
    """
    Generates a response to a given query using a retrieval-augmented chain.

    Args:
        query (str): The user's input query.

    Returns:
        str: The generated response as a string.
    """
    return retriever.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
