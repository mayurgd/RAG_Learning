import os
import uuid
import v5.constants as const
from pydantic import BaseModel
from contextvars import ContextVar
from fastapi import FastAPI, HTTPException
from v5.backend.rag import generate_response
from v5.logger import setup_logging, loggers_utils

if not os.path.exists(const.LOG_FILE_LOC.rsplit("/", 1)[0]):
    os.makedirs(const.LOG_FILE_LOC.rsplit("/", 1)[0])

# Create a context variable for correlation ID
correlation_id_ctx_var = ContextVar("correlation_id", default=str(uuid.uuid4()))
# Generate and set a new correlation ID for each log
correlation_id_ctx_var.set(str(uuid.uuid4()))


app = FastAPI()
setup_logging(correlation_id_ctx_var, log_to_file=True, log_file=const.LOG_FILE_LOC)
logger = loggers_utils(__name__)


class Query(BaseModel):
    session_id: str
    query: str


@app.post("/generate-response/")
async def query_docs(request: Query):
    logger.info(
        f"Received request: session_id={request.session_id}, query={request.query}"
    )
    try:
        query = request.query
        session_id = request.session_id
        logger.info("Generating Response")
        response = generate_response(query=query, session_id=session_id)
        logger.info(
            f"Response generated successfully for session_id={session_id}, response: {response['answer']}"
        )
        return {
            "query": request.query,
            "response": response,
        }
    except Exception as e:
        logger.error(f"Error during query processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during query: {e}")
