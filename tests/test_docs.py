import json
from pathlib import Path

from scripts.export_openapi import TARGET, build_schema


def test_committed_openapi_matches_the_code():
    """If this fails the API changed: run  python -m scripts.export_openapi  and update docs/API.md."""
    assert json.loads(TARGET.read_text(encoding="utf-8")) == json.loads(json.dumps(build_schema(), sort_keys=True))


def test_api_doc_mentions_every_endpoint():
    text = (Path(TARGET).parent / "API.md").read_text(encoding="utf-8")
    for path, methods in build_schema()["paths"].items():
        for method in methods:
            assert f"{method.upper()} {path}" in text, f"docs/API.md does not document {method.upper()} {path}"
