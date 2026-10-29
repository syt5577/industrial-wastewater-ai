import os

from src.evaluation import load_eval_data,hit_at_k,reciprocal_rank_at_k
from pathlib import Path
from src.loader import load_documents
from src.retrieval import build_bm25,bm25_search,build_faiss,faiss_search
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from src.fusion import rrf_fusion




def evaluate_bm25(eval_data, bm25, documents):
    hits = []
    mrrs = []
    if not eval_data:
        return 0.0,0.0
    for eval_file in eval_data:
        query = eval_file["question"]
        relevant_chunk_ids = eval_file['relevant_chunk_ids']
        retrieved_results = bm25_search(query, bm25,documents,k=3)
        retrieved_chunk_ids = [results['document'].metadata['chunk_id']
                               for results in retrieved_results]
        hit3 = hit_at_k(
            retrieved_results,
            relevant_chunk_ids,
            k=3
        )
        mrr3 = reciprocal_rank_at_k(retrieved_results,
                                    relevant_chunk_ids,
                                    k=3)
        print(f'bm25的具体情况\n\n'
              f'问题是{query}\n'
              f'正确答案是{relevant_chunk_ids}\n'
              f'找到的答案{retrieved_chunk_ids}\n'
              f'hit3是{hit3}\n'
              f'mrr3是{mrr3}\n'
              f'{"*"*20}')
        hits.append(hit3)
        mrrs.append(mrr3)
    sum_hits = sum(hits)/len(hits)
    sum_mrr3 = sum(mrrs)/len(mrrs)
    print(f'bm25总体的hit3是{sum_hits:.2%}'
          f'bm25总体的mrr3是{sum_mrr3:.2%}')
    return sum_hits, sum_mrr3

def faiss_evaluate(eval_data,vectorstore):
    hits = []
    mrrs = []
    if not eval_data:
        return 0.0,0.0
    for eval_file in eval_data:
        query = eval_file["question"]
        relevant_chunk_ids = eval_file['relevant_chunk_ids']
        retrieved_results = faiss_search(query, vectorstore, k=3)
        retrieved_chunk_ids = [results['document'].metadata['chunk_id']
                               for results in retrieved_results]
        hit3 = hit_at_k(
                retrieved_results,
                relevant_chunk_ids,
                k=3
            )
        mrr3 = reciprocal_rank_at_k(retrieved_results,
                                        relevant_chunk_ids,
                                        k=3)
        print(f'faiss的具体情况\n\n'
              f'问题是{query}\n'
                  f'正确答案是{relevant_chunk_ids}\n'
                  f'找到的答案{retrieved_chunk_ids}\n'
                  f'hit3是{hit3}\n'
                  f'mrr3是{mrr3}\n'
                  f'{"*"*20}')
        hits.append(hit3)
        mrrs.append(mrr3)
    sum_hits = sum(hits)/len(hits)
    sum_mrr3 = sum(mrrs)/len(mrrs)
    print(f'faiss总体的hit3是{sum_hits:.2%}'
              f'faiss总体的mrr3是{sum_mrr3:.2%}')
    return sum_hits, sum_mrr3
def evaluate_rrf(eval_data, bm25, documents, vectorstore):
    hits = []
    mrrs = []
    if not eval_data:
        return 0.0,0.0
    for eval_file in eval_data:
        query = eval_file["question"]
        relevant_chunk_ids = eval_file['relevant_chunk_ids']
        bm25_result = bm25_search(query, bm25,documents,k=10)
        faiss_result = faiss_search(query, vectorstore, k=10)
        rrf_result = rrf_fusion(bm25_result,faiss_result,top_k=5)
        retrieved_chunk_ids = [results['document'].metadata['chunk_id']
                               for results in rrf_result]
        hit3 = hit_at_k(
            rrf_result,
            relevant_chunk_ids,
            k=3
        )
        mrr3 = reciprocal_rank_at_k(rrf_result,
                                    relevant_chunk_ids,
                                    k=3)
        hits.append(hit3)
        mrrs.append(mrr3)
        print(f'rrf的具体情况\n\n'
              f'问题是{query}\n'
              f'正确答案是{relevant_chunk_ids}\n'
              f'找到的答案{retrieved_chunk_ids}\n'
              f'hit3是{hit3}\n'
              f'mrr3是{mrr3}\n'
              f'{"*" * 20}')

    sum_hits = sum(hits)/len(hits)
    sum_mrr3 = sum(mrrs)/len(mrrs)
    print(f'rrf总体的hit3是{sum_hits:.2%}'
                 f'rrf总体的mrr3是{sum_mrr3:.2%}')
    return sum_hits, sum_mrr3


def main():
    load_dotenv()
    api_key = os.getenv('SILICONFLOW_API_KEY')
    BASE_DIR = Path(__file__).resolve().parent
    formal_eval_dir = BASE_DIR / "data" / "eval" / "retrieval_eval.json"
    colloquial_eval_path = BASE_DIR / "data" / "eval" / "retrieval_eval_colloquial.json"
    documents_dir = BASE_DIR / "data" / "processed"
    formal_eval_data = load_eval_data(formal_eval_dir)
    colloquial_eval_data = load_eval_data(colloquial_eval_path)
    documents = load_documents(documents_dir)
    bm25 = build_bm25(documents)

    embeddings = OpenAIEmbeddings(base_url="https://api.siliconflow.cn/v1",
                              model='BAAI/bge-m3',
                              api_key=api_key,
                              check_embedding_ctx_length=False)
    vectorstore = build_faiss(documents, embeddings)
    print('正式问法')
    bm25_scores = evaluate_bm25(formal_eval_data, bm25, documents)
    faiss_scores = faiss_evaluate(formal_eval_data, vectorstore)
    rrf_scores = evaluate_rrf(formal_eval_data, bm25, documents, vectorstore)
    print("\n===== 检索效果对比 =====")
    print("BM25:", bm25_scores)
    print("FAISS:", faiss_scores)
    print("RRF:", rrf_scores)
    print('日常问法')
    bm25_scores = evaluate_bm25(colloquial_eval_data, bm25, documents)
    faiss_scores = faiss_evaluate(colloquial_eval_data, vectorstore)
    rrf_scores = evaluate_rrf(colloquial_eval_data, bm25, documents, vectorstore)
    print("\n===== 检索效果对比 =====")
    print("BM25:", bm25_scores)
    print("FAISS:", faiss_scores)
    print("RRF:", rrf_scores)
if __name__ == "__main__":
    main()

