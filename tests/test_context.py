from importlib.metadata import metadata

from langchain_core.documents import Document
from src.rag import build_context

def test_context_top_k():
    doc_a = Document(
        page_content='外委污水接收前需要检测水质',
        metadata = {'chunk_id' :'A'})
    doc_b = Document(
        page_content='需要审核客户资质',
        metadata={'chunk_id':'B'})
    rrf_result = [
        {'document':doc_a,'score':'0.03'},
        {'document':doc_b,'score':'0.02'}]
    context = build_context(rrf_result,top_k=1)
    assert '来源：A' in context
    assert '外委污水接收前需要检测水质' in context
    assert '来源：B' not in context
def test_context_join():
    doc_a = Document(
        page_content="水质检测",
        metadata={"chunk_id": "A"}
    )

    doc_b = Document(
        page_content="资质审核",
        metadata={"chunk_id": "B"}
    )

    results = [
        {"document": doc_a, "score": 0.03},
        {"document": doc_b, "score": 0.02}
    ]

    context = build_context(results)
    assert '来源：A\n内容：水质检测' in context
    assert '来源：B\n内容：资质审核' in context
    assert '\n\n----------\n\n' in context
    print(repr(context))
def test_context_empty():
    results = []

    context = build_context(results, top_k=3)

    assert '' == context