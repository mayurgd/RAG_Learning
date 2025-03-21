import os
import faiss
import pickle
import numpy as np
from typing import List
import v5.constants as const
from rank_bm25 import BM25Okapi
from v5.logger import loggers_utils
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from v5.backend.helpers import replace_t_with_space
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from typing import List, Dict, Any
from pydantic import BaseModel


from typing import Dict, Any
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords

# Load stopwords once
STOPWORDS = set(stopwords.words("english"))
logger = loggers_utils(__name__)


def load_and_process_data(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    docs = WebBaseLoader(const.lib_info_link).load()
    logger.info("Loaded documents from web source.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)
    cleaned_texts = replace_t_with_space(splits)

    logger.info("Split documents into chunks. Total chunks: %d", len(cleaned_texts))
    return cleaned_texts


def save_bm25_index(index: BM25Okapi, filename: str):
    """Save the BM25 index to a file."""
    with open(filename, "wb") as f:
        pickle.dump(index, f)


def load_bm25_index(filename: str) -> BM25Okapi:
    """Load a BM25 index from a file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


class VectorDbWithBM25:
    def __init__(self, vector_db, docs):
        self.__vector_db = vector_db
        self.__bm25_corpus = [(doc.page_content, doc.metadata) for doc in docs]

        tokenized_corpus = [self.preprocess_text(doc[0]) for doc in self.__bm25_corpus]

        if os.path.exists(const.BM25_INDEX_LOC):
            self.__bm25 = load_bm25_index(const.BM25_INDEX_LOC)
        else:
            self.__bm25 = BM25Okapi(tokenized_corpus)
            save_bm25_index(self.__bm25, const.BM25_INDEX_LOC)

    def preprocess_text(self, doc):
        return [word.lower() for word in doc.split() if word.lower() not in STOPWORDS]

    def vector_db_search(self, query: str, k=3) -> Dict[str, Dict[str, Any]]:
        """Performs a vector search and returns results with metadata."""

        # Perform similarity search with scores
        try:
            docs_and_scores = self.__vector_db.similarity_search_with_relevance_scores(
                query=query, k=k
            )
        except Exception as e:
            logger.error(f"Vector DB search failed: {e}")
            return {}

        search_result = {}

        # Ensure each result has metadata
        for doc, score in docs_and_scores:
            if not isinstance(doc, Document):  # Ensure it's a valid LangChain Document
                logger.warning(f"Unexpected document type: {type(doc)}")
                continue

            search_result[doc.page_content] = {
                "score": score,
                "metadata": (
                    doc.metadata if doc.metadata else {}
                ),  # Keep metadata intact
            }

        # Sort results by score (descending)
        return dict(
            sorted(search_result.items(), key=lambda x: x[1]["score"], reverse=True)
        )

    def bm25_search(self, query: str, k=3) -> Dict[str, float]:
        tokenized_query = query.split(" ")
        doc_scores = self.__bm25.get_scores(tokenized_query)
        # Store metadata alongside content
        docs_with_scores = {
            content: {"score": score, "metadata": metadata}
            for (content, metadata), score in zip(self.__bm25_corpus, doc_scores)
        }

        # Sort by score and return top-k results
        sorted_docs_with_scores = sorted(
            docs_with_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )
        return dict(sorted_docs_with_scores[:k])

    def combine_results(
        self,
        vector_db_search_results: Dict[str, Dict[str, Any]],
        bm25_search_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Combines vector search and BM25 search results, normalizing scores correctly."""

        def normalize_dict(input_dict):
            if not input_dict:
                return {}

            epsilon = 0.05
            min_value = min(v["score"] for v in input_dict.values())
            max_value = max(v["score"] for v in input_dict.values())
            a, b = 0.05, 1  # Normalization range

            if max_value == min_value:
                return {
                    k: {"score": b if max_value > 0.5 else a, "metadata": v["metadata"]}
                    for k, v in input_dict.items()
                }

            return {
                k: {
                    "score": a
                    + ((v["score"] - min_value) / (max_value - min_value + epsilon))
                    * (b - a),
                    "metadata": v["metadata"],
                }
                for k, v in input_dict.items()
            }

        # Normalize scores
        norm_vector_db_search_results = normalize_dict(vector_db_search_results)
        norm_bm25_search_results = normalize_dict(bm25_search_results)

        # Combine results by taking the max score
        combined_dict = norm_vector_db_search_results.copy()

        for k, v in norm_bm25_search_results.items():
            if k in combined_dict:
                combined_dict[k]["score"] = max(combined_dict[k]["score"], v["score"])
            else:
                combined_dict[k] = v

        # Ensure sorting is done based on scores correctly
        sorted_docs_with_scores = sorted(
            combined_dict.items(), key=lambda x: x[1]["score"], reverse=True
        )

        return dict(sorted_docs_with_scores)

    def search(self, query: str, k=3, do_bm25_search=True) -> Dict[str, float]:
        vector_db_search_results = self.vector_db_search(query, k=k)

        if do_bm25_search:
            bm25_search_results = self.bm25_search(
                " ".join(self.preprocess_text(query)), k=k
            )
            if bm25_search_results:
                combined_search_results = self.combine_results(
                    vector_db_search_results, bm25_search_results
                )
                sorted_docs_with_scores = sorted(
                    combined_search_results.items(),
                    key=lambda x: x[1]["score"],
                    reverse=True,
                )
                return dict(sorted_docs_with_scores)
        return vector_db_search_results


class VectorDbBM25Retriever(BaseRetriever, BaseModel):
    k: int = 3
    do_bm25_search: bool = True

    def __init__(
        self,
        vector_db_with_bm25: VectorDbWithBM25,
        k: int = 3,
        do_bm25_search: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        object.__setattr__(
            self, "vector_db_with_bm25", vector_db_with_bm25
        )  # Bypass Pydantic validation
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "do_bm25_search", do_bm25_search)

    def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        k = kwargs.get("k", self.k)
        search_results = self.vector_db_with_bm25.search(
            query, k=k, do_bm25_search=self.do_bm25_search
        )

        return [
            Document(page_content=content, metadata=data["metadata"])
            for content, data in search_results.items()
        ]

    async def aget_relevant_documents(
        self, query: str, **kwargs: Any
    ) -> List[Document]:
        """Async version of `get_relevant_documents`."""
        return self.get_relevant_documents(query, **kwargs)

    class Config:
        arbitrary_types_allowed = (
            True  # Allow non-Pydantic objects like `VectorDbWithBM25`
        )


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
    logger.info("Starting process_vector_store function.")
    if os.path.exists(const.VECTOR_INDEX_LOC) and not recreate_vector_store:
        logger.info("Loading existing FAISS index from local storage.")
        vector_store = FAISS.load_local(
            const.VECTOR_INDEX_LOC,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        cleaned_texts = load_and_process_data(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        logger.info(
            "No existing FAISS index found or recreation requested. Processing documents."
        )
        cleaned_texts = load_and_process_data(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        embeddings = embedding_model.embed_documents(
            [doc.page_content for doc in cleaned_texts]
        )
        logger.info("Generated embeddings for documents.")

        dimension = len(embeddings[0])
        hnsw_index = faiss.IndexHNSWFlat(dimension, 32)
        logger.info("Initialized FAISS HNSW index with dimension: %d", dimension)

        embeddings_array = np.array(embeddings, dtype=np.float32)
        hnsw_index.add(embeddings_array)
        logger.info("Added embeddings to FAISS HNSW index.")

        vector_store = FAISS(
            embedding_function=embedding_model,
            index=hnsw_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        ).from_documents(cleaned_texts, embedding_model)
        logger.info("Wrapped FAISS index with LangChain's FAISS vector store.")

        vector_store.save_local(const.VECTOR_INDEX_LOC)
        logger.info("Saved FAISS index locally.")

        logger.info("Creating BM25 index to retrive by keywords")
    vector_db_with_bm25 = VectorDbWithBM25(vector_db=vector_store, docs=cleaned_texts)

    # retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    # logger.info("Converted vector store into retriever.")
    return vector_db_with_bm25
