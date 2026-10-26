from src.loader import load_documents
import json


def test_loader(tmp_path):
    records = [
        {'page_content':"外委污水接收前需要检测水质",'metadata':{'chunk_id':"A"}}
    ]
    file_path = tmp_path / "test_documents.json"
    with open(file_path, "w",encoding='utf-8') as f:
        json.dump(records, f,ensure_ascii=False)
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata['chunk_id'] == 'A'
    assert docs[0].page_content =="外委污水接收前需要检测水质"
def test_loader_dedup(tmp_path):
    records = [
        {
            "page_content": "第一条水质检测记录",
            "metadata": {"chunk_id": "A"}
        },
        {
            "page_content": "第二条水质检测记录",
            "metadata": {"chunk_id": "A"}
        },
        {
            "page_content": "客户资质审核",
            "metadata": {"chunk_id": "B"}
        }
    ]
    file_path = tmp_path / "test_dedup_documents.json"
    with open(file_path, "w",encoding='utf-8') as f:
        json.dump(records, f,ensure_ascii=False)
    docs = load_documents(tmp_path)
    assert len(docs) == 2
    assert docs[0].metadata['chunk_id'] == 'A'
    assert docs[1].metadata['chunk_id'] == 'B'
    assert docs[1].page_content == '客户资质审核'