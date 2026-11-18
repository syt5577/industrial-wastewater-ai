from fastapi import FastAPI
from pydantic import BaseModel
from main import initial_rag, get_rag_answer
from contextlib import asynccontextmanager
import pandas as pd
from src.tools import create_pricing_tool
from src.agent import pricing_agent
from pathlib import Path
import os
from src.contract_review import (
    ContractReviewResult,
    load_contract_docx,
    load_contract_rules,
    review_contract
)
from fastapi import UploadFile, File, HTTPException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    app.state.rag_resources = initial_rag()
    limits = {
        "COD_mg_L": 600,
        "NH4_N_mg_L": 30,
        "TP_mg_L": 3.2
    }
    contract_llm = ChatOpenAI(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V3.2"
    )

    app.state.contract_llm = contract_llm.with_structured_output(
        ContractReviewResult
    )

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

@app.post("/contract-review")
def contract_review(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail='请上传DOCX格式的合同'
        )
    contract_docx = load_contract_docx(file.file)
    rule_path = Path(__file__).parent / "data/rules/contract_rules.json"
    rules = load_contract_rules(rule_path)
    results = review_contract(contract_docx,rules,app.state.contract_llm)
    return {
        "contract_name": file.filename,
        "total": len(results),
        "results": [
            result.model_dump()
            for result in results
        ]
    }