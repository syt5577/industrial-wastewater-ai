import pytest
from src.indexing import collect_file_metadata, generate_metadata_hash, save_manifest, load_manifest, is_index_valid


def test_index_invalid_when_files_missing(tmp_path):
    # Create a temporary directory for testing
    index_path = tmp_path / "faiss_index"
    index_path.mkdir()

    # Save a manifest with a specific hash
    test_hash = "test_hash_value"
    save_manifest(index_path, test_hash)

    # Ensure the index is invalid when files are missing
    assert not is_index_valid(index_path, test_hash)
def test_hash_not_same(tmp_path):
    index_path = tmp_path / "faiss_index"
    index_path.mkdir()

    # Save a manifest with a specific hash
    test_hash = "test_hash_value"
    save_manifest(index_path, test_hash)
    (index_path / "index.faiss").write_text("dummy faiss content")
    (index_path / "index.pkl").write_text("dummy pkl content")

    # Ensure the index is invalid when the hash does not match
    assert not is_index_valid(index_path, "different_hash_value")

def test_index_valid_when_files_and_hash_match(tmp_path):
    index_path = tmp_path / "faiss_index"
    index_path.mkdir()
    
        # Save a manifest with a specific hash
    test_hash = "test_hash_value"
    save_manifest(index_path, test_hash)
    (index_path / "index.faiss").write_text("dummy faiss content")
    (index_path / "index.pkl").write_text("dummy pkl content")
    
    assert is_index_valid(index_path, "test_hash_value")
