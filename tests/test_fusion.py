import pytest
from langchain_core.documents import Document
from src.fusion import rrf_fusion

def test_rrf_single_retriever():
    doc_a = Document(
        page_content="水质检测",
        metadata={"chunk_id": "A"}
    )

    doc_b = Document(
        page_content="资质审核",
        metadata={"chunk_id": "B"}
    )

    bm25_results = [
        {"document": doc_a, "score": 10.0},
        {"document": doc_b, "score": 8.0}
    ]

    faiss_results = []
    result = rrf_fusion(bm25_results, faiss_results)
    assert result[0]['document'].metadata["chunk_id"] == "A"
    assert result[0]['score'] == pytest.approx(1/61)
    assert len(result) == 2
def test_rrf_two_retrievers():
    doc_a = Document(page_content="A", metadata={"chunk_id": "A"})
    doc_b = Document(page_content="B", metadata={"chunk_id": "B"})
    doc_c = Document(page_content="C", metadata={"chunk_id": "C"})
    doc_d = Document(page_content="D", metadata={"chunk_id": "D"})

    bm25_results = [
        {"document": doc_a, "score": 10},
        {"document": doc_b, "score": 8},
        {"document": doc_c, "score": 6}
    ]

    faiss_results = [
        {"document": doc_b, "score": 0.1},
        {"document": doc_a, "score": 0.2},
        {"document": doc_d, "score": 0.3}
    ]

    result = rrf_fusion(bm25_results, faiss_results)

    # 下面的 assert 由你自己写
    assert result[0]['score'] == result[1]['score']
    assert result[2]['score'] == result[3]['score']
    assert result[0]['score'] > result[2]['score']
    assert result[1]['score'] > result[3]['score']
def test_rrf_empty():
    result = rrf_fusion([], [])
    assert result == []