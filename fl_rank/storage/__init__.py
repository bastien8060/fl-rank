# fl_rank/storage/__init__.py
"""
Storage backends for fl-rank.
"""

from fl_rank.storage.base import StorageConfig
from fl_rank.storage.memory import InMemoryStorage

try:
    from fl_rank.storage.pgvector import PgVectorStorage
    __all__ = ["StorageConfig", "InMemoryStorage", "PgVectorStorage"]
except ImportError:
    __all__ = ["StorageConfig", "InMemoryStorage"]
