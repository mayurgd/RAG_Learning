from dotenv import load_dotenv
from v5.logger import loggers_utils
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logger = loggers_utils(__name__)


def get_chat_model() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns a ChatGoogleGenerativeAI model instance.

    The model used is 'gemini-2.0-flash'

    Returns:
        ChatGoogleGenerativeAI: An instance of the Gemini 2.0 Flash chat model.
    """
    logger.info("Initializing ChatGoogleGenerativeAI model instance.")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        logger.info("ChatGoogleGenerativeAI model instance created successfully.")
        return llm
    except Exception as e:
        logger.error(
            f"Error initializing ChatGoogleGenerativeAI model: {e}", exc_info=True
        )
        raise
