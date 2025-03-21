from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from v4.backend.rag import generate_response

app = FastAPI()


class Query(BaseModel):
    session_id: str
    query: str


@app.post("/generate-response/")
async def query_docs(request: Query):
    print(request)
    try:
        query = request.query
        session_id = request.session_id
        response = generate_response(query=query, session_id=session_id)

        return {
            "query": request.query,
            "response": response,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {e}")
