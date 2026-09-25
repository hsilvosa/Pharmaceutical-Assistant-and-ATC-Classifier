from src.rag.providers import HashEmbedder
from src.rag.store import LanceStore


def test_lance_store_builds_dense_and_lexical_indexes(fixture_components, tmp_path) -> None:
    chunks, medicines, _, _ = fixture_components
    embedder = HashEmbedder()
    vectors = embedder.encode([chunk.embedding_text for chunk in chunks])
    store = LanceStore.build(tmp_path / "index", chunks, vectors, medicines)
    dense = store.dense_search(embedder.encode(["alergia"])[0], 2, "10001")
    lexical = store.lexical_search("alergia", 2, "10001")
    assert dense[0].chunk.registration_number == "10001"
    assert lexical[0].chunk.section == "4.3"

