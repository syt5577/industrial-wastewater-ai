import jieba
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
def build_bm25(documents):
    tokenizer_corpus = [list(jieba.cut(doc.page_content)) for doc in documents]
    bm25 = BM25Okapi(tokenizer_corpus)
    return bm25
def bm25_search(query, bm25, documents, k=5):
    # 1. 对 query 进行 jieba 分词

    # 2. 使用 bm25.get_scores() 计算分数

    # 3. 根据分数从高到低排序，取前 k 个索引

    # 4. 根据索引从 documents 找回 Document

    # 5. 返回包含 document 和 score 的字典列表
    query_tokens = list(jieba.cut(query))
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]
    result = []
    for i in top_indices:
        result.append(
            {
                "document":documents[i],
                "score":float(scores[i])
            }
        )
    return result
def build_faiss(documents, embeddings):
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    return vectorstore
def faiss_search(query, vectorstore, k=5):
    # 1. 使用 vectorstore 进行相似度检索
    # 2. 将返回的 Document 和 score 统一整理成字典
    # 3. 返回结果列表
    faiss_results = []
    results = vectorstore.similarity_search_with_score(query, k=k)
    for doc,score in results:
        faiss_results.append({
            "document":doc,
            "score":float(score)
        })
    return faiss_results