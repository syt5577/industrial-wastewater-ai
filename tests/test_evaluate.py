import pytest
import evaluate
from langchain_core.documents import Document


def test_evaluate_bm25(monkeypatch):
    doc_a = Document(
        page_content="A",
        metadata={"chunk_id": "A"}
    )

    doc_b = Document(
        page_content="B",
        metadata={"chunk_id": "B"}
    )

    def fake_bm25_search(query, bm25, documents, k=3):
        return [
            {"document": doc_a, "score": 0.9},
            {"document": doc_b, "score": 0.8}
        ]

    monkeypatch.setattr(
        evaluate,
        "bm25_search",
        fake_bm25_search
    )
    eval_data = [
        {
            "question": "问题1",
            "relevant_chunk_ids": ["A"]
        },
        {
            "question": "问题2",
            "relevant_chunk_ids": ["B"]
        }
    ]

    hit, mrr = evaluate.evaluate_bm25(
        eval_data,
        bm25=None,
        documents=None)
    print("Hit@3:", hit)
    print("MRR@3:", mrr)
    assert hit == 1
    assert mrr == 0.75

def test_evaluate_faiss(monkeypatch):
    doc_a = Document(
        page_content="A",
        metadata={"chunk_id": "A"}
    )

    doc_b = Document(
        page_content="B",
        metadata={"chunk_id": "B"}
    )

    def fake_faiss_search(query,vectoresore,k=3):
        return [
            {"document": doc_a, "score": 0.9},
            {"document": doc_b, "score": 0.8}
        ]
    monkeypatch.setattr(
        evaluate,
        "faiss_search",
        fake_faiss_search
    )
    eval_data = [
        {
            "question": "问题1",
            "relevant_chunk_ids": ["B"]
        },
        {
            "question": "问题2",
            "relevant_chunk_ids": ["A"]
        }
    ]

    hit,mrr = evaluate.faiss_evaluate(eval_data,vectorstore=None)
    print("Hit@3:", hit)
    print("MRR@3:", mrr)
    assert hit == 1
    assert mrr == 0.75