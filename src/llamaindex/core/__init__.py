"""
Core components of the LlamaIndex package.

This module contains the foundational classes and interfaces that are used 
throughout the LlamaIndex package.
"""

from llamaindex.core.document import Document, DocumentChunk
from llamaindex.core.index import Index
from llamaindex.core.query import Query, QueryResult, QueryMatch

__all__ = [
    "Document",
    "DocumentChunk",
    "Index",
    "Query",
    "QueryResult",
    "QueryMatch",
] 