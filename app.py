"""ASGI entrypoint.

    uvicorn app:app --reload
"""
from __future__ import annotations

from vistaar.api import create_app

app = create_app()
