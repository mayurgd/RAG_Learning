from pydantic import BaseModel, Field


class AgentQuery(BaseModel):
    query: str = Field(..., example="Analyze the sales pattern for product name B")
