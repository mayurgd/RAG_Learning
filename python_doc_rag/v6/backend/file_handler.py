import os
import pickle
from typing import Any

from v6.logger import loggers_utils
from v6.backend.utils import time_it

logger = loggers_utils(__name__)


def create_directories_for_path(path: str) -> None:
    """
    Creates necessary directories for a given file or directory path.

    - If the path has a file extension (e.g., ends with ".pkl"), it creates directories up to the parent folder.
    - If the path does not have an extension, it treats it as a directory and creates it.

    Args:
        path (str): The file or directory path.

    Returns:
        None
    """
    # Check if path has an extension (assuming it's a file)
    if "." in os.path.basename(path):
        dir_path = os.path.dirname(path)  # Extract parent directory
    else:
        dir_path = path  # Treat the entire path as a directory

    # Create directories if they don't exist
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directories created: {dir_path}")
    else:
        logger.info(f"Directory already exists: {dir_path}")


def file_exists(filepath: str) -> bool:
    """
    Checks if a file exists at the given path.

    Args:
        filepath (str): The full path of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.exists(filepath)


@time_it
def save_to_pickle(data: Any, filepath: str) -> None:
    """
    Saves the given data to a pickle file.

    Args:
        data (Any): The data to be saved.
        filepath (str): The path of the file to save the data to.

    Returns:
        None
    """
    create_directories_for_path(filepath)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Data successfully saved to {filepath}")


@time_it
def load_from_pickle(filepath: str) -> Any:
    """
    Loads data from a pickle file.

    Args:
        filepath (str): The path of the pickle file to load data from.

    Returns:
        Any: The loaded data.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Data successfully loaded from {filepath}")
    return data
