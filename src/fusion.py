def rrf_fusion(bm25_results, faiss_results, k=60,top_k = None):
    rrf_scores = {}
    documents = {}
    for result in [bm25_results, faiss_results]:
        for rank,item in enumerate(result,start=1):
            chunk_id = item["document"].metadata["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0) + 1/(rank+k)
            documents[chunk_id] = item["document"]
    sorted_rrf_scores = sorted(rrf_scores,
                        key= lambda chunk_id: rrf_scores[chunk_id],
                        reverse=True)[:top_k]
    return [
         {"document":documents[chunk_id],
         "score":rrf_scores[chunk_id]}
         for chunk_id in sorted_rrf_scores
      ]

