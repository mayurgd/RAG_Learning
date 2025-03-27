import os
import faiss
import string
import numpy as np

from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from typing import List, Any, Optional
from langchain.schema.runnable import Runnable
from langchain.docstore.document import Document
from pydantic import BaseModel, Field, PrivateAttr
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

import v7.constants as const
from v7.logger import loggers_utils
from v7.backend.utils import time_it
from v7.backend.file_handler import (
    file_exists,
    save_to_pickle,
    load_from_pickle,
)


logger = loggers_utils(__name__)


class VectorDbBM25Retriever(BaseModel, Runnable):
    name: str = Field(default="VectorDbBM25Retriever")  # Required for Runnable
    k: int = 3
    do_bm25_search: bool = True

    # Declare non-Pydantic attributes
    _vector_store: Any = PrivateAttr()
    _bm25: Any = PrivateAttr(default=None)
    _bm25_corpus: List[str] = PrivateAttr(default=None)

    def __init__(
        self,
        vector_store: Any,
        bm25: Any = None,
        bm25_corpus: List[str] = None,
        k: int = 3,
        do_bm25_search: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vector_store = vector_store
        self._bm25 = bm25
        self._bm25_corpus = bm25_corpus
        self.k = k
        self.do_bm25_search = do_bm25_search

    @staticmethod
    def normalize_scores(doc_list):
        if not doc_list:
            return []

        scores = [score for _, score in doc_list]
        min_value, max_value = min(scores), max(scores)
        a, b = 0.05, 1

        if min_value == max_value:
            default_score = b if max_value > 0.5 else a
            return [(doc, default_score) for doc, _ in doc_list]

        return [
            (doc, a + ((score - min_value) / (max_value - min_value)) * (b - a))
            for doc, score in doc_list
        ]

    @staticmethod
    def combine_scores_list(vector_scores, bm25_scores):
        hashmap = {}

        def get_doc_key(doc):
            source = doc.metadata.get("source", "")
            content_hash = hash(doc.page_content)
            return f"{source}_{content_hash}"

        for doc, score in vector_scores:
            key = get_doc_key(doc)
            hashmap[key] = [doc, score]

        for doc, score in bm25_scores:
            key = get_doc_key(doc)
            if key in hashmap:
                hashmap[key][1] = max(score, hashmap[key][1])
            else:
                hashmap[key] = [doc, score]

        return sorted(hashmap.values(), key=lambda x: x[1], reverse=True)

    def search(self, query: str, k: int):
        doc_scores = self._vector_store.similarity_search_with_relevance_scores(
            query=query, k=k
        )
        vector_scores = sorted(
            [doc for doc in doc_scores if doc[1] > 0.3],  # Filter docs with score > 0.5
            key=lambda x: x[1],
            reverse=True,
        )
        vector_scores = self.normalize_scores(vector_scores)

        bm25_scores = []
        if self._bm25 and self.do_bm25_search:
            stop_words = set(stopwords.words("english"))
            tokenized_query = [
                word.lower().translate(str.maketrans("", "", string.punctuation))
                for word in query.split()
                if word.lower().translate(str.maketrans("", "", string.punctuation))
                not in stop_words
            ]
            scores = self._bm25.get_scores(tokenized_query)
            doc_scores = list(zip(self._bm25_corpus, scores))
            doc_scores = sorted(
                [
                    doc for doc in doc_scores if doc[1] > 0.3
                ],  # Filter docs with score > 0.5
                key=lambda x: x[1],
                reverse=True,
            )
            bm25_scores = self.normalize_scores(doc_scores[:k])

        return self.combine_scores_list(
            vector_scores=vector_scores, bm25_scores=bm25_scores
        )

    def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        k = kwargs.get("k", self.k)
        return self.search(query, k=k)

    async def aget_relevant_documents(
        self, query: str, **kwargs: Any
    ) -> List[Document]:
        return self.get_relevant_documents(query)

    def invoke(self, query: str) -> List[Document]:
        """Implements LangChain's Runnable `invoke` method."""
        return self.get_relevant_documents(query)[: self.k]

    def runnable(self) -> "Runnable":
        """Returns itself as a Runnable instance for LangChain pipelines."""
        return self

    model_config = {"arbitrary_types_allowed": True}


@time_it
def create_vector_store(
    docs: List[Document],
    embedding_model=None,
    re_run: bool = False,
    use_hnsw: bool = False,
) -> FAISS:
    """
    Processes documents by chunking them and creating a FAISS-based vector store for retrieval.

    - If an existing FAISS index is found, it loads it.
    - Otherwise, it processes documents, generates embeddings, builds a FAISS index, and saves it.

    Args:
        docs (List[Document]): A list of documents containing `page_content` and `metadata`.
        embedding_model : The embedding model used for vectorization.
        re_run (bool, optional): Whether to recreate the vector store if it already exists. Defaults to False.
        use_hnsw (bool, optional): Whether to use HNSW indexing in FAISS for approximate nearest neighbors. Defaults to False.

    Returns:
        FAISS: A FAISS vector store containing indexed document embeddings.
    """

    logger.info("Starting create_vector_store function.")

    if not file_exists(os.path.join(const.VECTOR_INDEX_LOC, "index.faiss")):
        logger.warning("FAISS index file not found. Creating a new one.")
        re_run = True

    # Load existing FAISS index if it exists and recreation is not requested
    if file_exists(const.VECTOR_INDEX_LOC) and not re_run:
        logger.info("Loading existing FAISS index from local storage.")
        vector_store = FAISS.load_local(
            const.VECTOR_INDEX_LOC,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        logger.info(
            "No existing FAISS index found or recreation requested. Processing documents."
        )
        cleaned_texts = [doc.page_content for doc in docs]

        # Generate document embeddings
        embeddings = embedding_model.embed_documents(cleaned_texts)
        logger.info("Generated embeddings for %d documents.", len(docs))

        # Determine embedding dimensions
        dimension = len(embeddings[0])
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Initialize FAISS index
        if use_hnsw:
            index = faiss.IndexHNSWFlat(dimension, 32)
            logger.info("Initialized FAISS HNSW index with dimension: %d", dimension)
        else:
            index = faiss.IndexFlatL2(dimension)
            logger.info("Initialized FAISS FlatL2 index with dimension: %d", dimension)

        index.add(embeddings_array)
        logger.info("Added embeddings to FAISS index.")

        # Create FAISS vector store
        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        ).from_documents(cleaned_texts, embedding_model)
        logger.info("Wrapped FAISS index with LangChain's FAISS vector store.")

        # Save vector store locally
        vector_store.save_local(const.VECTOR_INDEX_LOC)
        logger.info("Saved FAISS index locally at %s", const.VECTOR_INDEX_LOC)

    return vector_store


@time_it
def create_bm25_index(
    docs: List[Document],
    re_run: bool = False,
) -> BM25Okapi:
    """
    Creates or loads a BM25 index from a given list of documents.

    - If a precomputed BM25 index exists, it loads it.
    - Otherwise, it tokenizes the document contents, builds a new BM25 index, and saves it.

    Args:
        docs (List[Document]): A list of documents containing `page_content` and `metadata`.
        re_run (bool, optional): Whether to recreate the bm25 index if it already exists. Defaults to False.

    Returns:
        BM25Okapi: The BM25 index built from the document contents.
    """
    STOPWORDS = set(stopwords.words("english"))

    def tokenize_text(doc: str) -> List[str]:
        """
        Tokenizes a given text by converting words to lowercase and removing stopwords.

        Args:
            doc (str): The text content to tokenize.

        Returns:
            List[str]: A list of tokenized words without stopwords.
        """
        return [word.lower() for word in doc.split() if word.lower() not in STOPWORDS]

    bm25_corpus = [(doc.page_content, doc.metadata) for doc in docs]

    if file_exists(const.BM25_INDEX_LOC) and not re_run:
        bm25 = load_from_pickle(const.BM25_INDEX_LOC)
    else:
        tokenized_corpus = [tokenize_text(doc[0]) for doc in bm25_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        save_to_pickle(bm25, const.BM25_INDEX_LOC)

    return bm25
