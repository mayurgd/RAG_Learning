import os
from datetime import datetime
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv()

import agentops

agentops.init()

from src.sql_agent.tools.nl2sql_tool import NL2SQLTool
from src.sql_agent.tools.gather_n_move_tool import GatherAndMoveOutputsTool
from crewai_tools import FileWriterTool, FileReadTool, CodeInterpreterTool

# Initialize the tool
nl2sql = NL2SQLTool(db_uri="sqlite:///src/sql_agent/sales.db")
gather_n_move = GatherAndMoveOutputsTool(output_dir="outputs")
file_writer_tool = FileWriterTool()
code_interpreter = CodeInterpreterTool(
    user_dockerfile_path="/Users/mayurgd/Documents/CodingSpace/RAG_Learning/sql_agent_2/src/sql_agent/Docker"
)


@CrewBase
class SqlAgent:
    """SqlAgent crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_engineer"],
            verbose=True,
            tools=[nl2sql, file_writer_tool, gather_n_move],
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst"],
            verbose=True,
            tools=[code_interpreter, file_writer_tool, gather_n_move],
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
