import os
from langchain_openai import ChatOpenAI
from src.retrieval import bm25_search,faiss_search
from src.fusion import rrf_fusion
from src.query_processing import rewrite_query

def build_context(rrf_results, top_k=3):
    context_parts = []
    top_doc = rrf_results[:top_k]
    for item in top_doc:
        doc = item['document']
        chunk_id = doc.metadata.get('chunk_id','未知chunk')
        source = doc.metadata.get('source','未知来源')
        content = doc.page_content
        context_parts.append(
            f'来源：{source}|{chunk_id}\n内容：{content}'
        )
    context = '\n\n----------\n\n'.join(context_parts)
    return context
def generate_answer(query, context, llm):

    prompt = f"""
    你是一名工业污水处理业务助手。

    请严格依据以下参考资料回答用户问题。
    如果参考资料不足以回答，请明确说明，不要编造。
    回答时引用对应的来源编号（chunk_id）。

    参考资料：
    {context}

    用户问题：
    {query}
    """
    response = llm.invoke(prompt)
    return response.content
def rag_answer(llm,query,vectorestore,bm25,documents):
    rewritten_query = rewrite_query(query,llm)
    faiss_result=faiss_search(rewritten_query,vectorstore=vectorestore,k=5)
    bm25_result = bm25_search(rewritten_query,bm25,documents,k=10)
    rrf_result = rrf_fusion(bm25_result,faiss_result,top_k=10)
    context = build_context(rrf_result,top_k=5)
    answer = generate_answer(query,context,llm)
    return answer

