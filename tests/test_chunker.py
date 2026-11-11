import pytest

from app.search.chunker import split_into_chunks


def words(chunk):
    return chunk.split()


def make_text(n_sentences, words_per=8):
    return " ".join(
        " ".join(f"s{i}w{j}" for j in range(words_per)) + "." for i in range(n_sentences))


def test_empty_and_whitespace():
    assert split_into_chunks("") == [] and split_into_chunks(None) == [] and split_into_chunks("  \n \n ") == []


def test_short_text_is_one_chunk():
    assert split_into_chunks("One short sentence. Another one.") == ["One short sentence. Another one."]


def test_chunks_respect_size_limit():
    chunks = split_into_chunks(make_text(60), size_words=50, overlap_words=10)
    assert len(chunks) > 5 and all(len(words(c)) <= 50 for c in chunks)


def test_every_word_is_covered():
    text = make_text(40)
    chunks = split_into_chunks(text, size_words=40, overlap_words=8)
    covered = set(" ".join(chunks).split())
    assert set(text.split()) <= covered


def test_consecutive_chunks_overlap():
    chunks = split_into_chunks(make_text(60), size_words=50, overlap_words=16)
    for a, b in zip(chunks, chunks[1:]):
        assert words(b)[0] in words(a)[-16:]       # next chunk starts inside the previous chunk's tail
        assert len(set(words(a)) & set(words(b))) >= 1


def test_no_overlap_when_zero():
    chunks = split_into_chunks(make_text(40), size_words=40, overlap_words=0)
    for a, b in zip(chunks, chunks[1:]):
        assert not (set(words(a)) & set(words(b)))


def test_chunks_end_on_sentence_boundaries_when_possible():
    chunks = split_into_chunks(make_text(40), size_words=50, overlap_words=10)
    assert all(c.endswith(".") for c in chunks)


def test_no_redundant_tail_chunk():
    chunks = split_into_chunks(make_text(60), size_words=50, overlap_words=16)
    assert len(set(chunks)) == len(chunks)
    assert set(words(chunks[-1])) - set(words(chunks[-2]))   # the last chunk contains new material


def test_single_giant_sentence_is_split_by_words():
    text = " ".join(f"w{i}" for i in range(450))            # no punctuation at all
    chunks = split_into_chunks(text, size_words=100, overlap_words=20)
    assert len(chunks) >= 5 and all(len(words(c)) <= 100 for c in chunks)
    assert set(text.split()) <= set(" ".join(chunks).split())


def test_long_sentence_still_gets_word_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = split_into_chunks(text, size_words=100, overlap_words=20)
    assert len(chunks) >= 3
    for a, b in zip(chunks, chunks[1:]):
        assert words(b)[0] in words(a)[-20:]       # overlap survives even without sentence boundaries


def test_lines_and_devanagari_danda_split_sentences():
    text = "पहला वाक्य यहाँ है। दूसरा वाक्य यहाँ है।\nतीसरी पंक्ति"
    assert split_into_chunks(text, size_words=5, overlap_words=0) == [
        "पहला वाक्य यहाँ है।", "दूसरा वाक्य यहाँ है।", "तीसरी पंक्ति"]


def test_huge_unspaced_text_does_not_crash():
    chunks = split_into_chunks("x" * 10_000, size_words=50, overlap_words=10)
    assert len(chunks) > 10 and all(len(c) <= 50 * 6 for c in chunks)      # bounded by the character budget
    assert "".join(chunks).count("x") >= 10_000                              # nothing lost


def test_deterministic():
    t = make_text(50)
    assert split_into_chunks(t, size_words=30, overlap_words=5) == split_into_chunks(t, size_words=30, overlap_words=5)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_into_chunks("text", size_words=size, overlap_words=overlap)
