def rrf_fusion(bm25_results, faiss_results, k=60,top_k = None,bm25_weight=0.8, faiss_weight=1.0):
    rrf_scores = {}
    documents = {}
    for rank,item in enumerate(bm25_results,start=1):
        chunk_id = item["document"].metadata["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0) + bm25_weight/(rank+k)
        documents[chunk_id] = item["document"]
    for rank,item in enumerate(faiss_results,start=1):
        chunk_id = item["document"].metadata["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0) + faiss_weight/(rank+k)
        documents[chunk_id] = item["document"]
    sorted_rrf_scores = sorted(rrf_scores,
                        key= lambda chunk_id: rrf_scores[chunk_id],
                        reverse=True)[:top_k]
    return [
         {"document":documents[chunk_id],
         "score":rrf_scores[chunk_id]}
         for chunk_id in sorted_rrf_scores
      ]












