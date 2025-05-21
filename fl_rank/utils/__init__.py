# fl_rank/utils/__init__.py
"""
Utility functions for fl-rank.
"""

from fl_rank.utils.serialization import json_to_vectors, vectors_to_json
from fl_rank.utils.vector import (
    normalize_vectors, normalize_vector, 
    cosine_similarity, euclidean_distance, 
    dot_product, batch_cosine_similarity
)

__all__ = [
    "json_to_vectors",
    "vectors_to_json",
    "normalize_vectors",
    "normalize_vector",
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
    "batch_cosine_similarity"
]