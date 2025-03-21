from v3.backend.retrieval_chains import retrieval_chain_with_memory_and_rewriting

retriever = retrieval_chain_with_memory_and_rewriting()


def generate_response(query: str, chat_history=[]) -> str:
    """
    Generates a response to a given query using a retrieval-augmented chain.

    Args:
        query (str): The user's input query.

    Returns:
        str: The generated response as a string.
    """
    return retriever(query, chat_history)
