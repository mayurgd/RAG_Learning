from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from v2.backend.rag import generate_response

app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/generate-response/")
async def query_docs(request: Query):
    print(request)
    try:
        return {"query": request.query, "response": generate_response(request.query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {e}")
