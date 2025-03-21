from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI


def get_chat_model() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns a ChatGoogleGenerativeAI model instance.

    The model used is 'gemini-2.0-flash'

    Returns:
        ChatGoogleGenerativeAI: An instance of the Gemini 2.0 Flash chat model.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return llm
