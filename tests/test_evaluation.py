import pytest
from langchain_core.documents import Document
from src.evaluation import hit_at_k, reciprocal_rank_at_k


def test_evaluation_rank_two():
    doc_a = Document(page_content="A", metadata={"chunk_id": "A"})
    doc_b = Document(page_content="B", metadata={"chunk_id": "B"})
    doc_c = Document(page_content="C", metadata={"chunk_id": "C"})

    retrieved_results = [
        {"document": doc_a, "score": 0.9},
        {"document": doc_b, "score": 0.8},
        {"document": doc_c, "score": 0.7}
    ]


    relevant_chunk_ids = ["A", "C"]

    hit = hit_at_k(retrieved_results, relevant_chunk_ids, k=3)
    rr = reciprocal_rank_at_k(retrieved_results, relevant_chunk_ids, k=3)

    print("Hit@3:", hit)
    print("RR@3:", rr)

    assert hit ==1
    assert rr == 1
def test_evaluation_empty():
    retrieved_results = []
    relevant_chunk_ids = ["A"]

    hit = hit_at_k(retrieved_results, relevant_chunk_ids, k=3)
    rr = reciprocal_rank_at_k(retrieved_results, relevant_chunk_ids, k=3)

    assert hit == 0
    assert rr == 0