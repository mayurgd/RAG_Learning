import multiprocessing

from typing import List, Tuple
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import v7.constants as const
from v7.logger import loggers_utils
from v7.backend.utils import time_it
from v7.backend.file_handler import file_exists, save_to_pickle, load_from_pickle

logger = loggers_utils(__name__)

# Notes
# 40 percent runtime decrease using multiprocessing


def process_doc(doc_chunk: Tuple[int, int, Document]) -> List[Document]:
    """
    Splits a document into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        doc_chunk (Tuple[int, int, Document]): A tuple containing chunk size, chunk overlap, and the document.

    Returns:
        List[Document]: A list of split document chunks.
    """
    chunk_size, chunk_overlap, doc = doc_chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents([doc])


def replace_t_with_space(list_of_documents: List[Document]) -> List[Document]:
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents (List[Document]): A list of document objects, each with a 'page_content' attribute.

    Returns:
        List[Document]: A new list of documents with tab characters replaced by spaces.
    """
    return [
        (
            Document(
                page_content=doc.page_content.replace("\t", " "), metadata=doc.metadata
            )
            if hasattr(doc, "page_content") and doc.page_content
            else doc
        )
        for doc in list_of_documents
    ]


@time_it
def load_and_process_data(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    re_run: bool = False,
) -> List[Document]:
    """
    Loads documents, splits them into smaller chunks, and cleans up text formatting.

    Args:
        chunk_size (int, optional): Size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 100.
        re_run (bool, optional): Whether to recreate the clean texts data if it already exists. Defaults to False.

    Returns:
        List[Document]: A list of processed and cleaned document chunks.
    """
    if file_exists(const.CLEANED_TEXTS_LOC) and not re_run:
        cleaned_texts = load_from_pickle(const.CLEANED_TEXTS_LOC)

    else:
        docs = WebBaseLoader(const.lib_info_link).load()
        logger.info("Loaded documents from web source.")

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(
                executor.map(
                    process_doc, [(chunk_size, chunk_overlap, doc) for doc in docs]
                )
            )

        splits = [item for sublist in results for item in sublist]

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            cleaned_texts = list(executor.map(replace_t_with_space, [splits]))

        cleaned_texts = cleaned_texts[0]  # Extract list from nested list

        logger.info(
            "Split documents into chunks using ThreadPoolExecutor. Total chunks: %d",
            len(cleaned_texts),
        )

        save_to_pickle(cleaned_texts, const.CLEANED_TEXTS_LOC)

    return cleaned_texts


if __name__ == "__main__":
    chunked_texts = load_and_process_data(chunk_size=500, chunk_overlap=100)
