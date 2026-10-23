import json
from pathlib import Path

def load_eval_data(file_path):
    path = Path(file_path)
    with open(path,"r",encoding="utf-8") as f:
        eval_data = json.load(f)
    return eval_data
def hit_at_k(retrieved_results, relevant_chunk_ids, k=3):
    # 1. 取前 k 条检索结果

    # 2. 提取这些结果的 chunk_id

    # 3. 判断是否有任意一个 chunk_id
    #    存在于 relevant_chunk_ids 中
    # 4. 命中返回 1，否则返回 0
    top_retrieved_results = retrieved_results[:k]
    top_relevant_chunk_ids = [
        item['document'].metadata['chunk_id']
        for item in top_retrieved_results
    ]
    hit = any(
        chunk_id in relevant_chunk_ids
        for chunk_id in top_relevant_chunk_ids
    )
    return int(hit)

def reciprocal_rank_at_k(retrieved_results, relevant_chunk_ids, k=3):
    top_retrieved_results = retrieved_results[:k]
    top_retrieved_chunk_ids = [
        item['document'].metadata['chunk_id']
        for item in top_retrieved_results
    ]
    for rank, item in enumerate(top_retrieved_chunk_ids,start=1):
        if item in relevant_chunk_ids:
            return 1/rank

    return 0
