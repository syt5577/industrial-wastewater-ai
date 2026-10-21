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

    for item in eval_data:
        id = item["id"]
        question = item['question']
        relevant_chunk_ids = item['relevant_chunk_ids']
        print(f'{id}问题开始\n问题为：{question}')
        fassi_result =
        vector_result =
        fusion_result =
        any(chunk_id in relevant_chunk_ids for chunk_id in retrieved_chunk_ids)
