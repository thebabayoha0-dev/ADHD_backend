"""Convenience entrypoint.

Run locally:
  python -m uvicorn main:app --reload

This file re-exports the FastAPI app from app.main.
"""

from app.main import app  # noqa: F401
