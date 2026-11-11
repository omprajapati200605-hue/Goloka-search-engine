from pathlib import Path

import pytest

from app.extraction import metadata as md
from app.extraction.extractor import compute_content_hash, extract_page, normalize_text

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://example.test/blog/hello-world"


def load(name: str, url: str = URL, **kw):
    return extract_page((FIXTURES / name).read_text(encoding="utf-8"), url, **kw)


@pytest.fixture(scope="module")
def article():
    return load("article_en.html", crawled_at="2026-01-01T00:00:00+00:00")


# ---- main content & clutter removal --------------------------------------------------------
def test_main_content_kept(article):
    for phrase in ["Search engines begin by crawling the web", "inverted index", "Tokenization",
                   "Stop-word removal", "Ranking decides which page appears first", "Indexing"]:
        assert phrase in article.text


@pytest.mark.parametrize("junk", [
    "SCRIPT_SECRET_TEXT", "color: red", "SITE_HEADER_TEXT", "Home", "Contact", "Accept all cookies",
    "HIDDEN_STYLE_TEXT", "HIDDEN_ATTR_TEXT", "NOSCRIPT_TEXT", "RELATED_TEXT",
    "Subscribe to our newsletter", "ASIDE_TEXT", "FOOTER_TEXT", "Privacy",
])
def test_clutter_removed(article, junk):
    assert junk not in article.text


def test_whitespace_normalized_and_source_linebreaks_not_leaked(article):
    assert "\xa0" not in article.text and "  " not in article.text
    assert "follow links from one page to the next." in article.text  # source newline is a space
    assert "inverted index so that queries" in article.text           # inline <strong> doesn't split


def test_headings_and_paragraphs(article):
    assert article.headings == [{"level": 1, "text": "Hello World"}, {"level": 2, "text": "Indexing"}]
    assert any(p.startswith("Search engines begin") for p in article.paragraphs)
    assert all(p == p.strip() and "  " not in p for p in article.paragraphs)


def test_messy_layout_keeps_wrapper_drops_sidebars():
    p = load("messy_layout.html")
    assert "Tomatoes need at least six hours" in p.text and "Mulch keeps moisture" in p.text
    assert "Basil\nSun" in p.text                       # table cells on separate lines
    for junk in ["NAVBAR_TEXT", "SIDEBAR_TEXT", "SHARE_TEXT", "COMMENTS_TEXT", "FOOTERLINKS_TEXT"]:
        assert junk not in p.text
    assert p.title == "Old School Page"


def test_br_creates_line_break():
    p = extract_page("<body><div>one<br>two</div></body>", URL)
    assert p.text == "one\ntwo"


# ---- metadata ------------------------------------------------------------------------------
def test_basic_metadata(article):
    assert article.title == "Hello World: A Guide to Search Engines"
    assert article.meta_description.startswith("A friendly introduction")
    assert article.domain == "example.test"
    assert article.language == "en"                            # "en-US" normalized
    assert article.canonical_url == "https://example.test/blog/hello-world"  # resolved, utm stripped


def test_meta_beats_jsonld_for_date_and_author(article):
    assert article.published_date == "2024-03-15T09:30:00+00:00"
    assert article.author == "Jane Doe"                         # "By " prefix removed


def test_original_url_and_crawl_timestamp_preserved():
    p = load("article_en.html", url="http://example.test/old", final_url="https://example.test/new",
             crawled_at="2026-05-05T05:05:05+00:00")
    assert p.url == "http://example.test/old" and p.final_url == "https://example.test/new"
    assert p.crawled_at == "2026-05-05T05:05:05+00:00" and p.domain == "example.test"


def test_crawled_at_defaults_to_now(article):
    assert extract_page("<p>x</p>", URL).crawled_at.endswith("+00:00")


def test_jsonld_used_when_no_meta():
    html = """<head><script type="application/ld+json">
      {"@graph": [{"@type": "WebSite"}, {"@type": "Article", "datePublished": "2023-07-04",
       "author": [{"name": "Ann"}, {"name": "Bob"}]}]}</script></head><body><p>x</p></body>"""
    p = extract_page(html, URL)
    assert p.published_date == "2023-07-04" and p.author == "Ann, Bob"


def test_invalid_jsonld_ignored():
    p = extract_page('<head><script type="application/ld+json">{oops</script></head><p>Hi</p>', URL)
    assert p.published_date is None and p.text == "Hi"


def test_time_element_only_trusted_inside_article():
    inside = extract_page('<article><time datetime="2022-02-02">x</time><p>Body</p></article>', URL)
    outside = extract_page('<div class="comments"><time datetime="2022-02-02">x</time></div><p>Body</p>', URL)
    assert inside.published_date == "2022-02-02" and outside.published_date is None


@pytest.mark.parametrize("html,expected", [
    ('<meta name="author" content="A. Writer"><p>x</p>', "A. Writer"),
    ('<span itemprop="author"><span itemprop="name">N. Name</span></span><p>x</p>', "N. Name"),
    ('<a rel="author" href="/u">Rel Author</a><p>x</p>', "Rel Author"),
    ('<meta name="author" content="https://facebook.com/someone"><p>x</p>', None),
    ("<p>x</p>", None),
])
def test_author_variants(html, expected):
    assert extract_page(html, URL).author == expected


@pytest.mark.parametrize("value,expected", [
    ("2024-01-05", "2024-01-05"), ("2024-01-05T10:00:00Z", "2024-01-05T10:00:00+00:00"),
    ("March 3, 2021", "2021-03-03"), ("3 March 2021", "2021-03-03"),
    ("Mon, 05 Feb 2024 10:00:00 GMT", "2024-02-05T10:00:00+00:00"),
    ("not a date", None), ("", None), (None, None), ("1850-01-01", None), ("2999-01-01", None),
])
def test_parse_date(value, expected):
    assert md.parse_date(value) == expected


def test_title_fallbacks():
    assert extract_page('<meta property="og:title" content="OG T"><p>x</p>', URL).title == "OG T"
    assert extract_page("<body><h1>Heading Title</h1></body>", URL).title == "Heading Title"
    assert extract_page("<p>x</p>", URL).title == ""


def test_description_fallback_to_og():
    p = extract_page('<meta property="og:description" content="OG desc"><p>x</p>', URL)
    assert p.meta_description == "OG desc"


# ---- language ------------------------------------------------------------------------------
def test_language_detected_from_text_when_undeclared():
    es = ("<body><p>Los motores de búsqueda rastrean la web para descubrir páginas nuevas y "
          "luego las almacenan en un índice para responder rápidamente a las consultas.</p></body>")
    assert extract_page(es, URL).language == "es"


def test_language_not_guessed_from_tiny_text():
    assert extract_page("<p>Hola</p>", URL).language is None


def test_declared_language_wins():
    assert extract_page('<html lang="fr_CA"><body><p>Hello there</p></body></html>', URL).language == "fr"


# ---- links ---------------------------------------------------------------------------------
def test_internal_and_external_links(article):
    assert "https://example.test/blog/ranking" in article.internal_links
    assert "https://example.test/about" in article.internal_links          # nav links kept as links
    assert "https://www.example.test/blog/other" in article.internal_links  # www == same site
    assert article.external_links == ["https://en.wikipedia.org/wiki/Search_engine"]
    assert not any(l.startswith(("mailto", "#")) for l in article.internal_links + article.external_links)
    assert len(set(article.internal_links)) == len(article.internal_links)


def test_links_use_base_tag_and_skip_self():
    html = '<base href="https://other.test/dir/"><a href="x">x</a><a href="https://example.test/blog/hello-world#s">me</a>'
    p = extract_page(html, URL)
    assert p.external_links == ["https://other.test/dir/x"] and p.internal_links == []


# ---- content hash --------------------------------------------------------------------------
def test_hash_stable_across_whitespace_and_reparse():
    a = extract_page("<title>T</title><p>Hello   world</p>", URL)
    b = extract_page("<title>T</title>\n\n<p>Hello\n world</p>", URL)
    assert a.content_hash == b.content_hash and len(a.content_hash) == 64


def test_hash_changes_with_text_or_title():
    base = extract_page("<title>T</title><p>Hello world</p>", URL).content_hash
    assert extract_page("<title>T</title><p>Hello there</p>", URL).content_hash != base
    assert extract_page("<title>U</title><p>Hello world</p>", URL).content_hash != base


def test_hash_ignores_url_and_crawl_time():
    a = extract_page("<p>Same</p>", "https://a.test/1", crawled_at="2020-01-01T00:00:00+00:00")
    b = extract_page("<p>Same</p>", "https://b.test/2", crawled_at="2025-01-01T00:00:00+00:00")
    assert a.content_hash == b.content_hash


def test_compute_content_hash_deterministic():
    assert compute_content_hash("t", "x  y") == compute_content_hash("t", " x y ")


# ---- robustness ----------------------------------------------------------------------------
@pytest.mark.parametrize("html", ["", "   ", "<html></html>", "<<<>>>", "<p>unclosed <b>tags <i>everywhere",
                                  "plain text, no tags at all"])
def test_malformed_or_empty_html_never_raises(html):
    p = extract_page(html, URL)
    assert isinstance(p.text, str) and len(p.content_hash) == 64


def test_empty_page_flags():
    p = extract_page("<html><body><script>x()</script></body></html>", URL)
    assert not p.has_content and p.word_count == 0


def test_entities_and_unicode_normalized():
    p = extract_page("<p>Caf&eacute; &amp; cr&egrave;me&nbsp;br&ucirc;l&eacute;e​!</p>", URL)
    assert p.text == "Café & crème brûlée!"
    assert normalize_text("a  b​") == "a b"


def test_wrapping_form_not_removed():  # ASP.NET WebForms wraps the whole page in <form>
    p = extract_page('<body><form id="aspnetForm"><p>Real content inside form</p></form></body>', URL)
    assert "Real content inside form" in p.text


def test_metadata_dict_is_json_serialisable(article):
    import json

    meta = article.metadata()
    assert json.loads(json.dumps(meta)) == meta
    assert meta["language"] == "en" and meta["word_count"] == article.word_count
    assert "text" not in meta


def test_integration_with_crawler_page_result():
    """Crawler output (PageResult) feeds directly into the extractor."""
    from app.crawler.crawler import PageResult

    pr = PageResult("https://s.test/a", "https://s.test/b", 1, 200, "<title>X</title><p>Body text</p>",
                    "2026-02-02T00:00:00+00:00", "h")
    page = extract_page(pr.html, pr.url, final_url=pr.final_url, crawled_at=pr.fetched_at)
    assert (page.url, page.final_url, page.crawled_at, page.title) == (
        "https://s.test/a", "https://s.test/b", "2026-02-02T00:00:00+00:00", "X")
