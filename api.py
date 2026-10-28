from fastapi import FastAPI
from pydantic import BaseModel
from main import initial_rag, rag_answer
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_resources = initial_rag()
    yield

app = FastAPI(lifespan=lifespan)

class QuestionRequest(BaseModel):
    question: str
@app.post("/ask")
def ask(request: QuestionRequest):
    bm25, vectorstore, documnts,llm=app.state.rag_resources
    query = request.question
    return{
        'question':query,
        'answer':rag_answer(query,bm25,vectorstore,documnts,llm)
    }