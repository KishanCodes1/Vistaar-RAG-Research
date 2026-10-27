"""Vistaar: citation-grounded RAG with calibrated abstention."""

__version__ = "0.2.0"

from .config import Settings
from .pipeline import RAGPipeline, RetrievalConfig

__all__ = ["Settings", "RAGPipeline", "RetrievalConfig", "__version__"]
