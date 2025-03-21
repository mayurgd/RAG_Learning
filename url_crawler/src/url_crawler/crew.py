from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import FirecrawlCrawlWebsiteTool
from dotenv import load_dotenv

load_dotenv()
tool = FirecrawlCrawlWebsiteTool(url="https://www.lilly.com/in/")


@CrewBase
class UrlCrawler:
    """UrlCrawler crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def web_scraper(self) -> Agent:
        return Agent(
            config=self.agents_config["web_scraper"],
            verbose=True,
            tools=[tool],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def scrape_task(self) -> Task:
        return Task(
            config=self.tasks_config["scrape_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the UrlCrawler crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
