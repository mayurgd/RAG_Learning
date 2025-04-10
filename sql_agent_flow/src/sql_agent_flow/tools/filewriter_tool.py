import os
import json
from distutils.util import strtobool
from typing import Any, Optional, Type, Union

from crewai.tools import BaseTool
from pydantic import BaseModel


class FileWriterToolInput(BaseModel):
    filename: str  # e.g., 'result.json'
    directory: Optional[str] = "./"
    overwrite: str = "False"
    content: Union[dict, list]  # Expecting structured data


class FileWriterTool(BaseTool):
    name: str = "File Writer Tool"
    description: str = (
        "A tool to write structured content (like dict or list) to a JSON file. "
        "Accepts filename, content (as a dict or list), and optionally a directory path and overwrite flag."
    )
    args_schema: Type[BaseModel] = FileWriterToolInput

    def _run(self, **kwargs: Any) -> str:
        try:
            directory = kwargs.get("directory") or "./"
            filename = kwargs["filename"]
            filepath = os.path.join(directory, filename)

            # Ensure the content is a dict or list
            content = kwargs["content"]
            if not isinstance(content, (dict, list)):
                return "Content must be a dict or list to be written as JSON."

            # Make sure directory exists
            os.makedirs(directory, exist_ok=True)

            # Convert overwrite flag
            overwrite = bool(strtobool(kwargs["overwrite"]))

            # Check if file exists
            if os.path.exists(filepath) and not overwrite:
                return f"File {filepath} already exists and overwrite option was not passed."

            # Write JSON content to file
            mode = "w" if overwrite else "x"
            with open(filepath, mode, encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)

            return f"Content successfully written to {filepath}"

        except FileExistsError:
            return (
                f"File {filepath} already exists and overwrite option was not passed."
            )
        except KeyError as e:
            return f"An error occurred while accessing key: {str(e)}"
        except Exception as e:
            return f"An error occurred while writing to the file: {str(e)}"
