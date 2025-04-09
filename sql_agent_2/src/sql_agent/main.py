#!/usr/bin/env python
import warnings
from fastapi import FastAPI
from src.sql_agent.crew import SqlAgent
from src.sql_agent.models import AgentQuery

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

app = FastAPI()


@app.post("/query-agent/")
async def query_agent(request: AgentQuery):
    input = {"query": request.query}
    try:
        SqlAgent().crew().kickoff(inputs=input)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
