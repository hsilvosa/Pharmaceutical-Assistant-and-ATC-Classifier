from src.rag.chunking import RegexOffsetTokenizer, chunk_document


def _chunks(text: str):
    return chunk_document(
        dataset_revision="abc1234",
        registration_number="1",
        medicine_name="Example",
        document_type=1,
        section="4.1",
        title="Indications",
        source_order=1,
        text=text,
        source_url="https://example.test",
        photo_url="",
        tokenizer=RegexOffsetTokenizer(),
        max_tokens=5,
        overlap_tokens=2,
    )


def test_chunk_ids_and_offsets_are_stable() -> None:
    first = _chunks("one two three four five six seven eight")
    second = _chunks("one two three four five six seven eight")
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert first[0].text == "one two three four five"
    assert first[1].text == "four five six seven eight"
    assert first[0].char_end > first[1].char_start


def test_empty_text_is_rejected() -> None:
    assert _chunks("   ") == []


