from dotenv import load_dotenv

load_dotenv()

from typing import List, Dict, Any
from pydantic import BaseModel, Field

# crewai imports
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# tools
from crewai_tools import CodeInterpreterTool
from src.sql_agent_flow.tools.nl2sql_tool import NL2SQLTool
from src.sql_agent_flow.tools.filewriter_tool import FileWriterTool

# Initialize the tool
nl2sql = NL2SQLTool(
    db_uri="sqlite:////Users/mayurgd/Documents/CodingSpace/RAG_Learning/sql_agent_flow/src/sql_agent_flow/sales.db"
)
file_writer_tool = FileWriterTool(directory="output")


from crewai import LLM

llm = LLM(
    model="gpt-4o-mini",
    temperature=0,  # Higher for more creative outputs
    timeout=120,  # Seconds to wait for response
    seed=42,  # For reproducible results
)


class DataGatheringOutput(BaseModel):
    user_query: str = Field(
        ..., description="The original natural language query from the user."
    )
    sql_query: str = Field(
        ..., description="The SQL query generated from the user query."
    )
    results: List[Dict[str, Any]] = Field(
        ...,
        description="List of records retrieved from the database after successful query execution.",
    )
    output_file_path: str = Field(
        ..., description="Path where the output JSON file is saved."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_query": "Get all orders placed in March 2023",
                "sql_query": "SELECT * FROM orders WHERE MONTH(order_date) = 'March' AND YEAR(order_date) = 2023;",
                "results": [
                    {"order_id": 101, "customer": "Alice", "order_date": "2023-03-12"},
                    {"order_id": 102, "customer": "Bob", "order_date": "2023-03-15"},
                ],
                "output_file_path": "output/orders_march_2023.json",
            }
        }


@CrewBase
class DataCrew:
    """Data crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_engineer"],
            verbose=True,
            tools=[nl2sql, file_writer_tool],
            llm=llm,
        )

    @task
    def data_engineering_task(self) -> Task:
        return Task(
            config=self.tasks_config["data_engineering_task"],
            output_pydantic=DataGatheringOutput,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ContentCrew crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
