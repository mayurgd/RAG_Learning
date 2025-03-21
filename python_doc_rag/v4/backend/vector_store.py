import os
import faiss
import numpy as np
from langchain.schema import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore

import v4.constants as const


def process_vector_store(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_model: GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    ),
    recreate_vector_store=False,
) -> BaseRetriever:
    """
    Processes documents by chunking them and creating a vector store for retrieval.

    Args:
        chunk_size (int, optional): The size of each text chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between consecutive chunks. Defaults to 100.
        embedding (GoogleGenerativeAIEmbeddings, optional): The embedding model used for vectorization.
            Defaults to GoogleGenerativeAIEmbeddings(model="models/embedding-001").

    Returns:
        BaseRetriever: A retriever object that can be used to fetch relevant documents based on queries.
    """
    if os.path.exists("v4/backend/faiss_index") and not recreate_vector_store:
        vector_store = FAISS.load_local(
            "v4/backend/faiss_index",
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        # Load documents from a web source
        docs = WebBaseLoader(const.lib_info_link).load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(docs)

        # vector_store = FAISS.from_documents(splits, embedding_model)

        # Generate embeddings for documents
        embeddings = embedding_model.embed_documents(
            [doc.page_content for doc in splits]
        )

        # Create FAISS HNSW index
        dimension = len(embeddings[0])  # Get the embedding vector size
        hnsw_index = faiss.IndexHNSWFlat(
            dimension, 32
        )  # 32 controls the HNSW ef_construction parameter

        # Convert embeddings to NumPy array and add them to FAISS
        embeddings_array = np.array(embeddings, dtype=np.float32)
        hnsw_index.add(embeddings_array)  # Adding embeddings to FAISS HNSW

        # Wrap FAISS index with LangChain's FAISS vector store
        vector_store = FAISS(
            embedding_function=embeddings,
            index=hnsw_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        ).from_documents(splits, embedding_model)

        vector_store.save_local("v4/backend/faiss_index")

    # Convert the vector store into a retriever
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    return retriever
