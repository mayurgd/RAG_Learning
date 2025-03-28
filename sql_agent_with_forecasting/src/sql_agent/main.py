#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from src.sql_agent.crew import SqlAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    inputs = {
        "query": "give daily sales graph of product name B, also provide additional analysis data."
    }

    try:
        SqlAgent().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


run()
