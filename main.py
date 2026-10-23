from src.loader import load_documents
from src.retrieval import build_bm25,bm25_search,build_faiss,faiss_search
from collections import Counter
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from src.rag import build_context,generate_answer
from langchain_openai import ChatOpenAI
load_dotenv()
api_key = os.getenv("SILICONFLOW_API_KEY")
documents = load_documents("data/processed")

print("Chunk数量：", len(documents))
query="如何签订污水处理合同"
bm25 = build_bm25(documents)
result = bm25_search(
    query=query,
    documents=documents,
    bm25=bm25,
    k=10
)
for rank,item, in enumerate(result,start=1):
    print("BM_25搜索内容：",rank,
          item['document'].metadata['chunk_id'],
          round(item['score'],4))
embedding = OpenAIEmbeddings(api_key=api_key,
                             model="BAAI/bge-m3",
                             base_url="https://api.siliconflow.cn/v1",
                             check_embedding_ctx_length=False
                             )
vectorstore = build_faiss(documents, embedding)
faiss_result = faiss_search(query, vectorstore, k=10)
for rank,item, in enumerate(faiss_result,start=1):
    print("faiss结果：",rank,
          item['document'].metadata['chunk_id'],
          round(item['score'],4))
from src.fusion import rrf_fusion

rrf_result = rrf_fusion(result, faiss_result)

for rank, item in enumerate(rrf_result, start=1):
    print(
        "RRF融合结果：",
        rank,
        item["document"].metadata["chunk_id"],
        round(item["score"], 5)
    )
context = build_context(rrf_result, top_k=5)
llm  = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V4.1-Flash",
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key,
    temperature=0
)
answer = generate_answer(query, context, llm)
print("\n===== RAG 最终回答 =====")
print(answer)