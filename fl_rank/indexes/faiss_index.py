# fl_rank/indexes/faiss_index.py
"""
FAISS index implementation.
"""

import os
import tempfile
from typing import Any, List, Optional, Tuple, Union

import faiss
import numpy as np

from fl_rank.core.base import BaseIndex
from fl_rank.indexes.base import IndexConfig


class FaissIndex(BaseIndex):
    """
    Vector index using FAISS.
    """
    
    def __init__(self, config: Optional[IndexConfig] = None):
        """
        Initialize with optional configuration.
        
        Args:
            config: Index configuration
        """
        self.config = config or IndexConfig()
        self.index = None
        self.dimension = None
        self.id_map = {}  # Maps FAISS index positions to custom IDs
    
    def build(self, vectors: np.ndarray, ids: Optional[List[Any]] = None) -> None:
        """
        Build an index from vectors.
        
        Args:
            vectors: Vectors to index
            ids: Optional list of IDs corresponding to vectors
        """
        if vectors.shape[0] == 0:
            raise ValueError("Cannot build index with empty vectors array")
        
        self.dimension = vectors.shape[1]
        
        # Create appropriate index based on configuration
        if self.config.index_type == "flat":
            if self.config.metric_type == "ip":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        elif self.config.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            if self.config.metric_type == "ip":
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.config.nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.config.nlist, faiss.METRIC_L2
                )
            self.index.nprobe = self.config.nprobe
            
            # IVF indexes need training
            self.index.train(vectors)
        elif self.config.index_type == "hnsw":
            if self.config.metric_type == "ip":
                self.index = faiss.IndexHNSWFlat(self.dimension, self.config.m, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexHNSWFlat(self.dimension, self.config.m, faiss.METRIC_L2)
            self.index.hnsw.efConstruction = self.config.ef_construction
            self.index.hnsw.efSearch = self.config.ef_search
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
        
        # Copy vectors to enforce correct data type
        vectors_copy = vectors.astype(np.float32)
        
        # Normalize vectors if using inner product metric
        if self.config.metric_type == "ip":
            faiss.normalize_L2(vectors_copy)
        
        # Add vectors to index
        self.index.add(vectors_copy)
        
        # Store ID mapping if provided
        if ids is not None:
            self.id_map = {i: id_ for i, id_ in enumerate(ids)}
    
    def add(self, vectors: np.ndarray, ids: Optional[List[Any]] = None) -> None:
        """
        Add vectors to an existing index.
        
        Args:
            vectors: Vectors to add
            ids: Optional list of IDs corresponding to vectors
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call build() first.")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
        
        # Copy vectors to enforce correct data type
        vectors_copy = vectors.astype(np.float32)
        
        # Normalize vectors if using inner product metric
        if self.config.metric_type == "ip":
            faiss.normalize_L2(vectors_copy)
        
        # Get current size of index
        current_size = self.index.ntotal
        
        # Add vectors to index
        self.index.add(vectors_copy)
        
        # Update ID mapping if provided
        if ids is not None:
            for i, id_ in enumerate(ids):
                self.id_map[current_size + i] = id_
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (scores, indices)
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call build() first.")
        
        # Reshape and copy to ensure correct format
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype(np.float32)
        
        # Normalize if using inner product metric
        if self.config.metric_type == "ip":
            faiss.normalize_L2(query_vector)
        
        # Perform search
        scores, indices = self.index.search(query_vector, k)
        
        # Map indices to custom IDs if available
        if self.id_map and len(self.id_map) > 0:
            mapped_indices = np.array([[self.id_map.get(int(idx), idx) for idx in row] for row in indices])
            return scores, mapped_indices
        
        return scores, indices
    
    def save(self, path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save the index
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call build() first.")
        
        # Save the FAISS index
        faiss.write_index(self.index, path)
        
        # Save ID mapping if exists
        if self.id_map:
            import pickle
            with open(f"{path}.idmap", "wb") as f:
                pickle.dump(self.id_map, f)
    
    def load(self, path: str) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Path to load the index from
        """
        # Load the FAISS index
        self.index = faiss.read_index(path)
        self.dimension = self.index.d
        
        # Load ID mapping if exists
        idmap_path = f"{path}.idmap"
        if os.path.exists(idmap_path):
            import pickle
            with open(idmap_path, "rb") as f:
                self.id_map = pickle.load(f)