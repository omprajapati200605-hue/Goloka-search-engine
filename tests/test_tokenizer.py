import pytest

from app.search.tokenizer import (
    MAX_QUERY_TERMS, MAX_TOKEN_LENGTH, STOPWORDS, analyze_document, analyze_query, tokenize,
)


def test_lowercases_and_splits_on_punctuation():
    assert tokenize("Hello, World! It's node.js & snake_case.") == [
        "hello", "world", "it", "s", "node", "js", "snake", "case"]


def test_numbers_kept():
    assert tokenize("Python 3.12 released in 2023") == ["python", "3", "12", "released", "in", "2023"]


def test_term_frequency_preserved():
    assert tokenize("go go go") == ["go", "go", "go"]


def test_unicode_normalization_and_accents_kept():
    assert tokenize("Café ﬁne ÜBER") == ["café", "fine", "über"]


def test_devanagari_words_not_fragmented():
    assert tokenize("खोज इंजन, हिन्दी।") == ["खोज", "इंजन", "हिन्दी"]


def test_cjk_and_mixed_scripts():
    assert tokenize("search 検索 engine") == ["search", "検索", "engine"]


def test_empty_and_none():
    assert tokenize("") == [] and tokenize(None) == [] and tokenize("  !!! ... ") == []


def test_overlong_tokens_dropped():
    assert tokenize("ok " + "x" * (MAX_TOKEN_LENGTH + 1) + " fine") == ["ok", "fine"]


def test_stopwords_removed_only_when_asked():
    assert tokenize("the quick fox") == ["the", "quick", "fox"]
    assert tokenize("the quick fox and the hound", remove_stopwords=True) == ["quick", "fox", "hound"]


def test_stopword_list_sanity():
    assert {"the", "and", "of", "to", "is", "a", "t", "s"} <= STOPWORDS
    assert not ({"search", "engine", "python", "c"} & STOPWORDS)


def test_analyze_document_removes_stopwords():
    assert analyze_document("The art of the search") == ["art", "search"]


def test_analyze_query_dedupes_and_keeps_order():
    assert analyze_query("python the python tutorial PYTHON") == ["python", "tutorial"]


def test_stopword_only_query_falls_back_to_raw_terms():
    assert analyze_query("to be or not to be") == ["to", "be", "or", "not"]


@pytest.mark.parametrize("q", ["", None, "   ", "!!!"])
def test_empty_query(q):
    assert analyze_query(q) == []


def test_query_length_capped():
    q = " ".join(f"term{i}" for i in range(500))
    assert len(analyze_query(q)) == MAX_QUERY_TERMS
