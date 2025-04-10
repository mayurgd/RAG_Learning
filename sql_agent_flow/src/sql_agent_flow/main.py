#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from crews.data_crew.data_crew import DataCrew


class DataFlow(Flow):

    @start()
    def gather_data(self):
        print("Gathering Data")
        result = DataCrew().crew().kickoff(inputs={"query": "What is the data about?"})

        print("Data gathered", result.raw)


def kickoff():
    data_flow = DataFlow()

    data_flow.gather_data()


def plot():
    data_flow = DataFlow()
    data_flow.plot()


if __name__ == "__main__":
    kickoff()
