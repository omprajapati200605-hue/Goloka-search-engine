"""Initialize the GOLOKA database: python -m scripts.init_db"""
from app.config import get_settings
from app.storage.db import init_db


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    conn = init_db(settings.db_path)
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    conn.close()
    print(f"Database ready at {settings.db_path}")
    print("Tables:", ", ".join(tables))


if __name__ == "__main__":
    main()