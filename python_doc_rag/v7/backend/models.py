from pydantic import BaseModel, Field, create_model
from typing import Any, Optional, Union, Dict, List, Tuple, Type


class Message(BaseModel):
    """Represents a message in a conversation."""

    role: str  # "human" or "ai"
    content: str


class LLMQuery(BaseModel):
    """Input model for LLM API request."""

    query: List[Tuple[str, str]]  # List of messages as (role, content)
    values: Dict[str, Any] = Field(default={})  # Variables for prompt formatting
    structured_output: Optional[Dict[str, str]] = (
        None  # Expected structured response format
    )
    chat_history: Optional[List[Message]] = None  # Optional chat history


class RAGQuery(BaseModel):
    session_id: str
    query: str


class RetrieverQuery(BaseModel):
    query: str


def MultiQueries():
    return {
        "recreated_query": "str",
        "query_1": "str",
        "query_2": "str",
        "query_3": "str",
    }


def LLMResposne():
    return {"input": "str", "answer": "str"}


def create_dynamic_pydantic_model(
    name: str, fields: Dict[str, type]
) -> Type[BaseModel]:
    """Dynamically creates a Pydantic model with given fields."""
    return create_model(name, **{key: (value, ...) for key, value in fields.items()})
