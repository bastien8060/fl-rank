# fl_rank/storage/base.py
"""
Base storage implementations and utilities.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from fl_rank.core.base import BaseStorage


@dataclass
class StorageConfig:
    """
    Configuration for storage backends.
    """
    connection_string: Optional[str] = None  # Database connection string
    table_name: str = "fl_rank_vectors"  # Table name for database storage
    vector_column: str = "embedding"  # Column name for vectors
    metadata_column: str = "metadata"  # Column name for metadata
    id_column: str = "id"  # Column name for IDs
