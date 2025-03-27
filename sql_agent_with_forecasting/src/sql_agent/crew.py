import os
from datetime import datetime
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv()
from src.sql_agent.tools.nl2sql_tool import NL2SQLTool
from crewai_tools import CodeInterpreterTool

nl2sql = NL2SQLTool(db_uri="sqlite:///src/sql_agent/sales.db")
code_interpreter = CodeInterpreterTool()


@CrewBase
class SqlAgent:
    """SqlAgent crew"""

    # Define paths for saving outputs
    output_dir = "/outputs"

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @agent
    def database_developer(self) -> Agent:
        return Agent(
            config=self.agents_config["database_developer"],
            verbose=True,
            tools=[nl2sql],
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst"],
            verbose=True,
        )

    @agent
    def code_executor(self) -> Agent:
        return Agent(
            config=self.agents_config["code_executor"],
            verbose=True,
            output_files={
                "data": f"{self.output_dir}/analysis_output_{self.timestamp}.csv",
                "visualization": f"{self.output_dir}/visualization_{self.timestamp}.png",
            },
            tools=[code_interpreter],
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writer"],
            verbose=True,
            output_file=f"{self.output_dir}/report_{self.timestamp}.md",
        )

    @task
    def query_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config["query_creation_task"],
        )

    @task
    def analysis_task(self) -> Task:
        return Task(config=self.tasks_config["analysis_task"], allow_delegation=True)

    @task
    def code_execution_task(self) -> Task:
        return Task(
            config=self.tasks_config["code_execution_task"],
            output_files={
                "data": f"{self.output_dir}/analysis_output_{self.timestamp}.csv",
                "visualization": f"{self.output_dir}/visualization_{self.timestamp}.png",
            },
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],
            output_file=f"{self.output_dir}/report_{self.timestamp}.md",
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
