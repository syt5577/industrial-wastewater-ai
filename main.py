from langchain_community.vectorstores import FAISS

from src.loader import load_documents
from src.retrieval import build_bm25,bm25_search,build_faiss,faiss_search
from src.fusion import rrf_fusion
from collections import Counter
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from src.rag import build_context,generate_answer
from langchain_openai import ChatOpenAI
load_dotenv()

from pathlib import Path

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")
def initial_rag():
    path = Path(__file__).resolve().parent/'data'/'processed'
    documnts = load_documents(path)
    bm25 = build_bm25(documnts)
    embedding = OpenAIEmbeddings(api_key=api_key,
                                 model="BAAI/bge-m3",
                                 base_url="https://api.siliconflow.cn/v1",
                                 check_embedding_ctx_length=False
                                 )
    index_path = Path(__file__).resolve().parent/'data'/'faiss_index'
    if (index_path/'index.faiss').exists() and (index_path/'index.pkl').exists():
        print('发现Faiss索引')
        vectorstore = FAISS.load_local(str(index_path),
                                 embeddings=embedding,
                                 allow_dangerous_deserialization=True)

    else:
        print('未发现索引，需重构')
        vectorstore = build_faiss(documnts, embedding)
        vectorstore.save_local(str(index_path))
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        base_url="https://api.siliconflow.cn/v1",
        api_key=api_key,
        temperature=0)
    print(f'FAISS对象类型：{type(vectorstore)}')
    return bm25, vectorstore, documnts,llm
def rag_answer(query,bm25,vectore,documnts,llm):
    bm25_results = bm25_search(query=query, bm25=bm25, documents=documnts, k=10)
    faiss_results = faiss_search(query=query, vectorstore=vectore, k=10)
    rrf_results = rrf_fusion(bm25_results, faiss_results)
    context = build_context(rrf_results, top_k=5)
    response = generate_answer(query, context, llm)
    return response
if __name__ == "__main__":
    bm25, vectorstore, documnts, llm = initial_rag()

    answer = rag_answer(
        "如何签订污水处理合同",
        bm25, vectorstore, documnts, llm
    )
    print(answer)
# api_key = os.getenv("SILICONFLOW_API_KEY")
# documents = load_documents("data/processed")

# print("Chunk数量：", len(documents))
# query="如何签订污水处理合同"
# bm25 = build_bm25(documents)
# result = bm25_search(
#     query=query,
#     documents=documents,
#     bm25=bm25,
#     k=10
# )
# for rank,item, in enumerate(result,start=1):
#     print("BM_25搜索内容：",rank,
#           item['document'].metadata['chunk_id'],
#           round(item['score'],4))
# embedding = OpenAIEmbeddings(api_key=api_key,
#                              model="BAAI/bge-m3",
#                              base_url="https://api.siliconflow.cn/v1",
#                              check_embedding_ctx_length=False
#                              )
# vectorstore = build_faiss(documents, embedding)
# faiss_result = faiss_search(query, vectorstore, k=10)
# for rank,item, in enumerate(faiss_result,start=1):
#     print("faiss结果：",rank,
#           item['document'].metadata['chunk_id'],
#           round(item['score'],4))
# from src.fusion import rrf_fusion

# rrf_result = rrf_fusion(result, faiss_result)

# for rank, item in enumerate(rrf_result, start=1):
#     print(
#         "RRF融合结果：",
#         rank,
#         item["document"].metadata["chunk_id"],
#         round(item["score"], 5)
#     )
# context = build_context(rrf_result, top_k=5)
# print('\n==========实际传给llm的context是========')
# print(context)
# llm  = ChatOpenAI(
#     model="deepseek-ai/DeepSeek-V4.1-Flash",
#     base_url="https://api.siliconflow.cn/v1",
#     api_key=api_key,
#     temperature=0
# )
# answer = generate_answer(query, context, llm)
# print("\n===== RAG 最终回答 =====")
# print(answer)


