from pathlib import Path
import json
from langchain_core.documents import Document


def load_documents(processed_dir):
    all_documents = []
    processed_dir = Path(processed_dir)
    for file_path in sorted(processed_dir.glob('*_documents.json')):
        with open(file_path,"r",encoding="utf-8") as f:
            records = json.load(f)
        docs = [
            Document(
                page_content = item['page_content'],
                metadata = item['metadata']
            )
        for item in records]
        all_documents.extend(docs)
    unique_docs = []
    seen = set()
    for doc in all_documents:
        chunk_id = doc.metadata['chunk_id']
        if chunk_id not in seen:
            unique_docs.append(doc)
            seen.add(chunk_id)
    return unique_docs


