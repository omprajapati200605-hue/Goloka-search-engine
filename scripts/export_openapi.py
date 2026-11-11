"""Write the OpenAPI schema to docs/openapi.json.   python -m scripts.export_openapi"""
import json
from pathlib import Path

from app.api.main import create_app
from app.config import Settings

TARGET = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"


def build_schema() -> dict:
    return create_app(Settings(_env_file=None)).openapi()          # building the schema never starts the engine


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(build_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
