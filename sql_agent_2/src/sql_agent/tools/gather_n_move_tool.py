import os
import shutil
from typing import Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class GatherAndMoveOutputsInput(BaseModel):
    """Input schema for GatherAndMoveOutputsTool."""

    filenames: List[str] = Field(..., description="List of output file names to move")
    output_dir: str = Field(
        default="outputs", description="Target directory to move files to"
    )


class GatherAndMoveOutputsTool(BaseTool):
    name: str = "GatherAndMoveOutputsTool"
    description: str = "Moves agent-generated output files to the 'outputs' directory"
    args_schema: Type[BaseModel] = GatherAndMoveOutputsInput

    def _run(self, filenames: List[str], output_dir: str = "outputs") -> str:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        moved_files = []
        skipped_files = []

        for filename in filenames:
            if os.path.exists(filename):
                dest_path = os.path.join(output_dir, os.path.basename(filename))
                shutil.move(filename, dest_path)
                moved_files.append(filename)
            else:
                skipped_files.append(filename)

        result = []
        if moved_files:
            result.append(f"✅ Moved files: {', '.join(moved_files)} → '{output_dir}'")
        if skipped_files:
            result.append(f"⚠️ Skipped (not found): {', '.join(skipped_files)}")
        return "\n".join(result)
