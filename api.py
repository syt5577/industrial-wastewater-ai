from fastapi import FastAPI
from pydantic import BaseModel
from main import initial_rag, get_rag_answer
from contextlib import asynccontextmanager
import pandas as pd
from src.tools import create_pricing_tool
from src.agent import pricing_agent
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_resources = initial_rag()
    limits = {
        "COD_mg_L": 600,
        "NH4_N_mg_L": 30,
        "TP_mg_L": 3.2
    }

    history_df = pd.DataFrame()

    app.state.pricing_tool = create_pricing_tool(
        history_df, limits, 10.0
    )
    print("报价工具初始化成功")
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
        'answer':get_rag_answer(query,bm25,vectorstore,documnts,llm)
    }
@app.post("/price")
def price(request: QuestionRequest):
    question = request.question
    bm25, vectorstore, documnts,llm=app.state.rag_resources
    pricing_tool = app.state.pricing_tool
    answer = pricing_agent(question, llm, pricing_tool)
    return {
        "question": question,
        "answer": answer
    }