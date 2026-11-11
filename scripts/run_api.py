"""Start the API:   python -m scripts.run_api      then open http://127.0.0.1:8000/docs"""
import uvicorn

from app.config import get_settings


def main() -> None:
    s = get_settings()
    if s.api_host not in ("127.0.0.1", "localhost", "::1") and s.api_key is None:
        print(f"WARNING: listening on {s.api_host} without GOLOKA_API_KEY: anyone who can reach this port can "
              "start crawls. Set GOLOKA_API_KEY or use 127.0.0.1.")
    uvicorn.run("app.api.main:app", host=s.api_host, port=s.api_port, log_level=s.log_level.lower())


if __name__ == "__main__":
    main()
