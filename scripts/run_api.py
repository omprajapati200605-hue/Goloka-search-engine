"""Start the API:   python -m scripts.run_api      then open http://127.0.0.1:8000/docs"""
import os

import uvicorn

from app.config import get_settings


def main() -> None:
    s = get_settings()
    if s.api_host not in ("127.0.0.1", "localhost", "::1") and s.api_key is None:
        print(f"WARNING: listening on {s.api_host} without GOLOKA_API_KEY: anyone who can reach this port can "
              "start crawls. Set GOLOKA_API_KEY or use 127.0.0.1.")
    # Hosts such as Render tell the app which port to use through the PORT variable
    port = int(os.environ["PORT"]) if "PORT" in os.environ and "GOLOKA_API_PORT" not in os.environ else s.api_port
    uvicorn.run("app.api.main:app", host=s.api_host, port=port, log_level=s.log_level.lower())


if __name__ == "__main__":
    main()
