from dotenv import load_dotenv

load_dotenv()


from crewai_tools import FileReadTool
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.sql_agent_flow.tools.filewriter_tool import FileWriterTool
from src.sql_agent_flow.tools.code_interpreter_tool import CodeInterpreterTool

# Initialize the tool
# file_writer_tool = FileWriterTool(directory="output")
code_interpreter = CodeInterpreterTool(
    # user_dockerfile_path="/Users/mayurgd/Documents/CodingSpace/RAG_Learning/sql_agent_2/src/sql_agent/Docker"
    unsafe_mode=True
)
file_read_tool = FileReadTool()
from crewai import LLM

llm = LLM(
    model="gpt-4o-mini",
    temperature=0,  # Higher for more creative outputs
    timeout=120,  # Seconds to wait for response
    seed=42,  # For reproducible results
)


@CrewBase
class AnalysisCrew:
    """Analysis crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst"],
            verbose=True,
            tools=[file_read_tool, code_interpreter],
            llm=llm,
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"],
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
