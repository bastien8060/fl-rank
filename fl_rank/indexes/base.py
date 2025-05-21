# fl_rank/indexes/base.py
"""
Base index implementations and utilities.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from fl_rank.core.base import BaseIndex


@dataclass
class IndexConfig:
    """
    Configuration for vector indexes.
    """
    index_type: str = "flat"  # "flat", "ivf", "hnsw", etc.
    metric_type: str = "ip"  # "ip" for inner product, "l2" for L2 distance
    nlist: int = 100  # Number of clusters for IVF indexes
    nprobe: int = 10  # Number of clusters to probe for IVF indexes
    m: int = 16  # Number of connections per layer for HNSW
    ef_construction: int = 100  # Construction-time parameter for HNSW
    ef_search: int = 50  # Search-time parameter for HNSW
