from repopilot.chunker import chunk_text, chunk_documents
from repopilot.models import SourceDocument


def test_chunk_text_short_text_stays_one_chunk():
    assert chunk_text("hello", chunk_size=20) == ["hello"]


def test_chunk_text_long_text_chunks_with_overlap():
    text = "abc " * 100
    chunks = chunk_text(text, chunk_size=80, overlap=10)
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)


def test_chunk_documents_adds_metadata():
    docs = [SourceDocument(id="doc1", text="hello " * 100, metadata={"path": "README.md"})]
    chunks = chunk_documents(docs, chunk_size=60, overlap=5)
    assert chunks
    assert chunks[0].metadata["path"] == "README.md"
    assert "chunk_index" in chunks[0].metadata
