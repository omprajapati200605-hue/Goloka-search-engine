from app.search.snippet import ELLIPSIS, build_snippet, truncate

LONG = ("Intro words that have nothing to do with the topic. " * 12 +
        "Tomatoes need plenty of sun and regular water to grow well. " +
        "Trailing filler sentence repeated for length. " * 12)


def test_short_text_returned_whole_without_ellipsis():
    assert build_snippet("Tomatoes love sun.", "tomatoes") == "Tomatoes love sun."


def test_window_is_centred_on_the_match_and_bounded():
    s = build_snippet(LONG, "tomatoes sun", max_chars=120)
    assert "Tomatoes" in s and len(s) <= 120 + 2
    assert s.startswith(ELLIPSIS) and s.endswith(ELLIPSIS)


def test_picks_window_with_most_distinct_terms():
    text = ("alpha " + "filler " * 60 + "alpha beta gamma together here " + "filler " * 60 + "gamma")
    s = build_snippet(text, "alpha beta gamma", max_chars=80)
    assert all(w in s for w in ("alpha", "beta", "gamma"))


def test_matching_is_case_and_accent_insensitive_but_keeps_original_text():
    s = build_snippet("x " * 100 + "Visit the CAFÉ downtown today " + "y " * 100, "café", max_chars=60)
    assert "CAFÉ" in s


def test_whitespace_and_newlines_collapsed():
    assert "\n" not in build_snippet("line one\n\n  line   two\ttab", "two")
    assert build_snippet("a\n\nb   c", "b") == "a b c"


def test_hindi_terms():
    text = "यह एक लंबा पाठ है। " * 20 + "खोज इंजन बहुत उपयोगी है। " + "और पाठ। " * 20
    assert "खोज इंजन" in build_snippet(text, "इंजन", max_chars=80)


def test_no_term_found_uses_fallback_then_document_start():
    assert build_snippet("Some body text about gardens.", "spaceship", fallback="Matched semantic chunk text") == \
        "Matched semantic chunk text"
    assert build_snippet("Some body text about gardens.", "spaceship") == "Some body text about gardens."


def test_fallback_is_truncated():
    s = build_snippet("irrelevant", "zzz", max_chars=40, fallback="word " * 50)
    assert len(s) <= 41 and s.endswith(ELLIPSIS)


def test_empty_inputs():
    assert build_snippet("", "query") == "" and build_snippet(None, "query") == ""
    assert build_snippet("Some text", "") == "Some text" and build_snippet("Some text", "   ") == "Some text"


def test_stopword_only_query_still_finds_text():
    assert "question" in build_snippet("To be or not to be, that is the question.", "the question", max_chars=50)


def test_match_always_inside_snippet_even_with_long_neighbouring_word():
    text = "a" * 70 + " target " + "b" * 200
    assert "target" in build_snippet(text, "target", max_chars=30)


def test_truncate_on_word_boundary():
    assert truncate("one two three four five", 12) == "one two…"
    assert truncate("short", 50) == "short" and truncate("", 10) == ""
