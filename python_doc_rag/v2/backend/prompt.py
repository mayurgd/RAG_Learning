from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def create_chat_prompt() -> ChatPromptTemplate:
    """
    Creates and returns a chat prompt template for question-answering tasks.

    Returns:
        ChatPromptTemplate: A structured chat prompt template.
    """
    template = """
    "You are an assistant for answering questions about Python libraries."
    "Use the following pieces of retrieved context to answer the question."
    "If you don't know the answer, say that you don't know."
    "Keep the answers concise."
    "\n\n"
    Conext: {context}
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        input_variables=["question"], template="{question}"
    )

    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat_prompt_template
