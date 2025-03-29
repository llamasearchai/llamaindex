"""
LlamaIndex - A high-performance indexing and search library for LlamaSearch.ai
"""

__version__ = "0.1.0"

from llamaindex.index import Index, DistributedIndex
from llamaindex.document import Document
from llamaindex.filter import Filter
from llamaindex.results import SearchResult, SearchResults
from llamaindex.config import IndexConfig

__all__ = [
    "Index",
    "DistributedIndex",
    "Document",
    "Filter",
    "SearchResult",
    "SearchResults",
    "IndexConfig",
] 