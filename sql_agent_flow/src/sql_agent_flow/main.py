#!/usr/bin/env python
import os
import json

from crewai import LLM
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from src.sql_agent_flow.tools.nl2sql_tool import NL2SQLTool
from src.sql_agent_flow.crews.data_crew.data_crew import DataCrew
from src.sql_agent_flow.crews.analysis_crew.analysis_crew import AnalysisCrew

# Initialize the NL2SQLTool
nl2sql_tool = NL2SQLTool(
    db_uri="sqlite:////Users/mayurgd/Documents/CodingSpace/RAG_Learning/sql_agent_flow/src/sql_agent_flow/sales.db"
)

# Advanced configuration with detailed parameters
llm = LLM(
    model="gpt-4o-mini",
    temperature=0,  # Higher for more creative outputs
    timeout=120,  # Seconds to wait for response
    seed=42,  # For reproducible results
)


# Define state for the DataFlow
class DataFlowState(BaseModel):
    user_query: str = ""
    sql_query: str = ""
    query_result: dict = {}


# DataFlow implementation
class DataFlow(Flow[DataFlowState]):
    """Flow to generate a SQL query via LLM using table and column metadata, then gather and analyze data"""

    @start()
    def generate_sql_query(self):
        """Generate a SQL query using a direct LLM call with DB schema context"""
        self.state.user_query = "How are the sales pattern for product 'C'"

        print("\n=== Generating SQL Query via LLM ===\n")

        # Prepare DB metadata context
        schema_context = {"tables": nl2sql_tool.tables, "columns": nl2sql_tool.columns}

        # Prepare LLM messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant designed to write SQL queries.",
            },
            {
                "role": "user",
                "content": f"""
                Given the following database schema:
                {json.dumps(schema_context, indent=2)}

                Write a SQL query to: {self.state.user_query}
                Return only the SQL query string. No explanation.
                """,
            },
        ]

        # Call LLM to get SQL query as a string
        response = llm.call(messages=messages)
        self.state.sql_query = response.strip()

        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)

        # Save generated SQL query to file
        with open("output/sql_query.sql", "w") as f:
            f.write(self.state.sql_query)

        print(f"\nGenerated SQL Query:\n{self.state.sql_query}\n")
        return self.state

    @listen(generate_sql_query)
    def gather_data(self, state):
        """Use the generated SQL to gather data using DataCrew"""
        print("Gathering data using generated SQL query...")

        # Kick off the data crew with the generated SQL
        result = (
            DataCrew()
            .crew()
            .kickoff(
                inputs={
                    "user_query": state.user_query,
                    "sql_query": state.sql_query,
                }
            )
        )

        state.query_result = result.raw

        # Save query result to file
        with open("output/query_result.json", "w") as f:
            json.dump(state.query_result, f, indent=2)

        print("Data gathering complete.\n", result.raw)
        return state

    @listen(gather_data)
    def analyze_data(self, state):
        """Analyze the gathered data using AnalysisCrew"""
        print("Analyzing data...")

        output_file_path = json.loads(state.query_result)["output_file_path"]

        if not os.path.isfile(output_file_path):
            with open(output_file_path, "w") as f:
                json.dump(json.loads(state.query_result)["results"], f)

        # Kick off the analysis crew with the gathered data
        result = (
            AnalysisCrew()
            .crew()
            .kickoff(
                inputs={
                    "query": state.user_query,
                    "output_file_path": output_file_path,
                }
            )
        )

        print("Analysis complete.\n", result.raw)
        return state


def kickoff():
    data_flow = DataFlow()
    data_flow.kickoff()


def plot():
    data_flow = DataFlow()
    data_flow.plot()


if __name__ == "__main__":
    kickoff()
