from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from v3.backend.rag import generate_response

app = FastAPI()
chat_history = []


class Query(BaseModel):
    query: str


@app.post("/generate-response/")
async def query_docs(request: Query):
    print(request)
    try:
        query = request.query
        response = generate_response(query=query, chat_history=chat_history[-5:])

        chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=response["answer"]),
            ]
        )

        return {
            "query": request.query,
            "response": response,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {e}")
