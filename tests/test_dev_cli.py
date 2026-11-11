import pytest

from app.config import Settings
from scripts.dev_cli import build_parser, cmd_crawl, cmd_hybrid, cmd_search, cmd_semantic, main
from tests.fake_embedder import HashingEmbedder
from app.search.embedder import EmbeddingError
from app.storage.repository import DocumentRepository


@pytest.fixture
def cli_settings(tmp_path):
    return Settings(
        _env_file=None, data_dir=tmp_path / "data", db_path=tmp_path / "data" / "goloka.db",
        vector_dir=tmp_path / "data" / "vectors", crawl_per_domain_delay_seconds=0.0,
        crawl_max_page_bytes=50_000, crawl_timeout_seconds=2.0,
    )


def page(title, body, links=()):
    anchors = "".join(f'<a href="{h}">l</a>' for h in links)
    return f"<html><head><title>{title}</title></head><body><main><h1>{title}</h1><p>{body}</p>{anchors}</main></body></html>"


def test_parser_defaults():
    args = build_parser().parse_args(["crawl", "https://a.test", "--allow", "b.test", "--allow", "c.test"])
    assert args.allow == ["b.test", "c.test"] and args.max_pages is None
    assert build_parser().parse_args(["search", "q"]).top == 10


def test_search_with_no_documents(cli_settings, capsys):
    args = build_parser().parse_args(["search", "anything"])
    assert cmd_search(cli_settings, args) == 1
    assert "No documents stored yet" in capsys.readouterr().out


def test_crawl_then_search_end_to_end(mock, cli_settings, resolver, capsys):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(200, html=page("Home", "Welcome to our garden site", ["/tomatoes", "/engines"]))
    mock.get("https://site.test/tomatoes").respond(200, html=page("Growing tomatoes", "Tomatoes need sun and water"))
    mock.get("https://site.test/engines").respond(200, html=page("Car engines", "Pistons and cylinders"))

    args = build_parser().parse_args(["crawl", "https://site.test/", "--max-pages", "10"])
    assert cmd_crawl(cli_settings, args, resolver=resolver) == 0
    out = capsys.readouterr().out
    assert "Crawled 3 page(s)" in out and "'created': 3" in out

    assert cmd_search(cli_settings, build_parser().parse_args(["search", "tomatoes sun"])) == 0
    out = capsys.readouterr().out
    assert "added 3" in out and "1. [" in out and "Growing tomatoes" in out
    assert out.index("https://site.test/tomatoes") > 0

    # second search: snapshot reloaded, nothing re-indexed
    assert cmd_search(cli_settings, build_parser().parse_args(["search", "pistons", "--backend", "native"])) == 0
    out = capsys.readouterr().out
    assert "unchanged 3" in out and "Car engines" in out


def test_search_no_results_message(mock, cli_settings, resolver, capsys):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(200, html=page("Home", "Only gardening here"))
    cmd_crawl(cli_settings, build_parser().parse_args(["crawl", "https://site.test/"]), resolver=resolver)
    capsys.readouterr()
    assert cmd_search(cli_settings, build_parser().parse_args(["search", "spaceship"])) == 0
    assert "No results." in capsys.readouterr().out


def test_crawl_rejects_unsafe_seed(cli_settings, resolver, capsys):
    args = build_parser().parse_args(["crawl", "http://127.0.0.1/"])
    assert cmd_crawl(cli_settings, args, resolver=resolver) == 2
    assert "Cannot crawl" in capsys.readouterr().err


def test_main_dispatches_search(cli_settings, capsys):
    assert main(["search", "x"], settings=cli_settings) == 1


def test_semantic_with_no_documents(cli_settings, capsys):
    assert cmd_semantic(cli_settings, build_parser().parse_args(["semantic", "x"]), HashingEmbedder()) == 1
    assert "No documents stored yet" in capsys.readouterr().out


def test_crawl_then_semantic_search_end_to_end(mock, cli_settings, resolver, capsys):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(200, html=page("Home", "Welcome", ["/a", "/b"]))
    mock.get("https://site.test/a").respond(200, html=page("Cooking", "Boil pasta in salted water with sauce"))
    mock.get("https://site.test/b").respond(200, html=page("Space", "Telescopes reveal distant galaxies"))
    cmd_crawl(cli_settings, build_parser().parse_args(["crawl", "https://site.test/"]), resolver=resolver)
    capsys.readouterr()

    embedder = HashingEmbedder(concepts={"noodles": "food", "pasta": "food"})
    assert cmd_semantic(cli_settings, build_parser().parse_args(["semantic", "noodles"]), embedder) == 0
    out = capsys.readouterr().out
    assert "added 3" in out and "1. [similarity" in out and "https://site.test/a" in out.split("1. ")[1].split("2. ")[0]

    second = HashingEmbedder(concepts={"noodles": "food", "pasta": "food"})        # restart: nothing re-embedded
    assert cmd_semantic(cli_settings, build_parser().parse_args(["semantic", "galaxies"]), second) == 0
    out = capsys.readouterr().out
    assert "unchanged 3" in out and second.embedded_texts == ["galaxies"]
    assert (cli_settings.vector_dir / "chunks.faiss").exists()


def test_semantic_reports_embedding_problem(mock, cli_settings, resolver, capsys):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(200, html=page("Home", "Some content here"))
    cmd_crawl(cli_settings, build_parser().parse_args(["crawl", "https://site.test/"]), resolver=resolver)
    capsys.readouterr()

    class Down(HashingEmbedder):
        def embed(self, texts):
            raise EmbeddingError("no internet")

    assert cmd_semantic(cli_settings, build_parser().parse_args(["semantic", "x"]), Down()) == 3
    assert "Embedding problem: no internet" in capsys.readouterr().err


CONCEPTS = {"noodles": "food", "pasta": "food", "automobile": "vehicle", "car": "vehicle"}


def crawl_garden_site(mock, cli_settings, resolver, capsys):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(200, html=page("Home", "Welcome to the site", ["/pasta", "/cars"]))
    mock.get("https://site.test/pasta").respond(200, html=page("Cooking", "Boil pasta in salted water with tomato sauce"))
    mock.get("https://site.test/cars").respond(200, html=page("Cars", "A fast car with a loud engine"))
    cmd_crawl(cli_settings, build_parser().parse_args(["crawl", "https://site.test/"]), resolver=resolver)
    capsys.readouterr()


def test_hybrid_end_to_end_and_method_flag(mock, cli_settings, resolver, capsys):
    crawl_garden_site(mock, cli_settings, resolver, capsys)
    for method in ("weighted", "rrf"):
        args = build_parser().parse_args(["hybrid", "noodles", "--method", method])
        assert cmd_hybrid(cli_settings, args, HashingEmbedder(concepts=CONCEPTS)) == 0
        out = capsys.readouterr().out
        assert f"Hybrid search ({method})" in out and "1. [" in out and "https://site.test/pasta" in out
        assert "semantic 0." in out and "bm25 -" in out        # found by meaning only: no BM25 score invented


def test_hybrid_degrades_to_keyword_when_model_unavailable(mock, cli_settings, resolver, capsys):
    crawl_garden_site(mock, cli_settings, resolver, capsys)

    class Down(HashingEmbedder):
        def embed(self, texts):
            raise EmbeddingError("no internet")

    assert cmd_hybrid(cli_settings, build_parser().parse_args(["hybrid", "pasta sauce"]), Down()) == 0
    out = capsys.readouterr().out
    assert "WARNING: semantic search unavailable (no internet)" in out and "https://site.test/pasta" in out


def test_hybrid_with_no_documents(cli_settings, capsys):
    assert cmd_hybrid(cli_settings, build_parser().parse_args(["hybrid", "x"]), HashingEmbedder()) == 1
    assert "No documents stored yet" in capsys.readouterr().out
