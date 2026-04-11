"""ASGI entrypoint: uvicorn main:app --reload --host 0.0.0.0 --port 8080"""

from app.main import app

__all__ = ["app"]
