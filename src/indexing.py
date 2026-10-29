from pathlib import Path
import hashlib
import json
def collect_file_metadata(data_path: Path):
    if not data_path.is_dir():
        raise FileNotFoundError(f"目录不存在: {data_path}")
    file_metadata = []
    for file in sorted(data_path.glob('*_documents.json')):
        file_name = file.name
        file_size = file.stat().st_size
        file_time = file.stat().st_mtime
        file_metadata.append(
        {'file_name': file_name,
        'file_size': file_size,
        'file_time': file_time,}
            )
    return file_metadata
def generate_metadata_hash(metadata):
    metadata_hash_file = json.dumps(metadata)
    metadata_hash = hashlib.sha256(metadata_hash_file.encode('utf-8')).hexdigest()
    return metadata_hash

def save_manifest(index_path:Path,matahash:str):
    manifest_index = index_path/'manifest.json'
    manifest_date = {
        'metadata_hash': matahash
    }
    index_path.mkdir(parents=True, exist_ok=True)
    with open(manifest_index, 'w', encoding='utf-8') as f:
        json.dump(manifest_date,f,ensure_ascii=False, indent=2)

def load_manifest(index_path:Path):
    manifest_index = index_path/'manifest.json'
    if not manifest_index.exists():
        return None
    with open(manifest_index, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('metadata_hash', None)

def is_index_valid(index_path: Path, current_hash: str):
    if not (index_path/'index.faiss').exists() or not (index_path/'index.pkl').exists():
        return False
    saved_hash = load_manifest(index_path)
    return saved_hash == current_hash




if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent/'data'/'processed'
    index_path =Path(__file__).resolve().parent.parent/'data'/'faiss_index'
    json_metadata = collect_file_metadata(data_path)
    current_hash = generate_metadata_hash(json_metadata)
    print(f"当前数据哈希值: {current_hash}")
    saved_hash = load_manifest(index_path)
    if saved_hash:
        print(f"已保存的哈希值: {saved_hash}")
    else:
        print("未找到已保存的哈希值。")
    print(f'索引是否有效: {is_index_valid(index_path, current_hash)}')
