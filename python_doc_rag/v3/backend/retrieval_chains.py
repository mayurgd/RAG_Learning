from pydantic import BaseModel, Field
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from v3.backend.llm import get_chat_model
from v3.backend.vector_store import process_vector_store

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


def basic_retrieval_chain():
    """
    Creates and returns a retrieval chain for question-answering tasks..
    """

    llm = get_chat_model()
    retriever = process_vector_store()

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

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain


def retrieval_chain_with_memory():
    llm = get_chat_model()
    retriever = process_vector_store()

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = (
        "You are an assistant for answering questions about Python libraries."
        "Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, say that you don't know."
        "Keep the answers concise."
        "\n\n"
        "Context: {context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Below we use create_stuff_documents_chain to feed all retrieved context # into the LLM. Note that we can also use StuffDocumentsChain and other # instances of BaseCombineDocumentsChain.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def retreival_grader(question, docs):

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM with function call
    llm = get_chat_model()
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    docs_to_use = []

    for doc in docs:
        print(doc.page_content, "\n", "-" * 50)
        res = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        print(res, "\n")
        if res.binary_score == "yes":
            docs_to_use.append(doc)

    return docs_to_use


def get_source_info(docs_to_use, question, answer):

    llm = get_chat_model()

    # Data model
    class HighlightDocuments(BaseModel):
        """Return the specific part of a document used for answering the question."""

        id: List[str] = Field(
            ..., description="List of id of docs used to answers the question"
        )

        title: List[str] = Field(
            ..., description="List of titles used to answers the question"
        )

        source: List[str] = Field(
            ..., description="List of sources used to answers the question"
        )

        segment: List[str] = Field(
            ...,
            description="List of direct segements from used documents that answers the question",
        )

    # parser
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
    - (Important) If you didn't used the specific document don't mention it.

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
        return "\n".join(
            f"<doc{i+1}>:\nTitle:{doc['metadata']['title']}\nSource:{doc['metadata']['source']}\nContent:{doc['page_content']}\n</doc{i+1}>\n"
            for i, doc in enumerate(docs)
        )

    # Run
    lookup_response = doc_lookup.invoke(
        {
            "documents": format_docs(docs_to_use),
            "question": question,
            "answer": answer,
        }
    )

    result = []
    for id, title, source, segment in zip(
        lookup_response.id,
        lookup_response.title,
        lookup_response.source,
        lookup_response.segment,
    ):
        result.append({"id": id, "title": title, "source": source, "segment": segment})
    return result


def retrieval_chain_with_memory_and_rewriting():
    llm = get_chat_model()
    retriever = process_vector_store()

    # Contextualize question based on chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. If the question is already self-contained, return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Query rewriting system prompt
    query_rewrite_template = """You are an AI assistant designed to enhance user queries for better retrieval accuracy.
    Your task is to rewrite the query while maintaining its original intent.

    Rules:
    1. If the query references previous chat history (e.g., uses pronouns like "it," "that," "they," or assumes prior context), rewrite it into a fully self-contained question.
    2. If the query is already clear and specific, rephrase it naturally while keeping its meaning intact.
    3. If the query is too vague to improve, return it unchanged.

    Chat History:
    {chat_history}

    Original Query: {original_query}

    Rewritten Query:"""

    query_rewrite_prompt = ChatPromptTemplate.from_template(query_rewrite_template)

    # Create query rewriting chain
    query_rewriter = query_rewrite_prompt | llm | StrOutputParser()

    # Answer question using retrieved context
    qa_system_prompt = (
        "You are an assistant for answering questions about Python libraries. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answers concise."
        "\n\n"
        "Context: {context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Chain to generate final response
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create final retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def retrieve_with_rewritten_query(original_query, chat_history):
        rewritten_query = query_rewriter.invoke(
            {"chat_history": chat_history, "original_query": original_query}
        )
        print(rewritten_query)
        response = rag_chain.invoke(
            {"chat_history": chat_history, "input": rewritten_query}
        )
        response["rewritten_input"] = rewritten_query
        return response

    return retrieve_with_rewritten_query
