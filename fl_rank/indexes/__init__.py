# fl_rank/indexes/__init__.py
"""
Vector indexing implementations for fl-rank.
"""

from fl_rank.indexes.base import IndexConfig
from fl_rank.indexes.faiss_index import FaissIndex

__all__ = [
    "IndexConfig",
    "FaissIndex",
]