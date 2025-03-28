import os
from datetime import datetime
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv()
from src.sql_agent.tools.nl2sql_tool import NL2SQLTool
from crewai_tools import FileWriterTool, FileReadTool, CodeInterpreterTool

# Initialize the tool
nl2sql = NL2SQLTool(db_uri="sqlite:///src/sql_agent/sales.db")
file_writer_tool = FileWriterTool()
file_read_tool = FileReadTool()
code_interpreter = CodeInterpreterTool(
    user_dockerfile_path="/Users/mayurgd/Documents/CodingSpace/RAG_Learning/sql_agent_with_forecasting/src/sql_agent/Docker"
)


@CrewBase
class SqlAgent:
    """SqlAgent crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @agent
    def database_developer(self) -> Agent:
        return Agent(
            config=self.agents_config["database_developer"],
            verbose=True,
            tools=[nl2sql, file_writer_tool],
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst"],
            verbose=True,
            tools=[code_interpreter, file_writer_tool],
            output_file=f"report_{self.timestamp}.md",
        )

    @task
    def query_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config["query_creation_task"],
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SqlAgent crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
