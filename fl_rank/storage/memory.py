# fl_rank/storage/memory.py
"""
In-memory storage implementation.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fl_rank.core.base import BaseStorage


class InMemoryStorage(BaseStorage):
    """
    In-memory storage backend.
    """
    
    def __init__(self):
        """
        Initialize in-memory storage.
        """
        self.vectors = {}  # {id: vector}
        self.metadata = {}  # {id: metadata}
        self.indexes = {}  # {name: index}
    
    def store_vectors(self, ids: List[Any], vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Store vectors with their IDs and optional metadata.
        
        Args:
            ids: IDs of the vectors
            vectors: Embedding vectors
            metadata: Optional metadata for each vector
        """
        if len(ids) != vectors.shape[0]:
            raise ValueError("Length of ids must match number of vectors")
        
        if metadata is not None and len(ids) != len(metadata):
            raise ValueError("Length of ids must match length of metadata")
        
        for i, id_ in enumerate(ids):
            self.vectors[id_] = vectors[i]
            if metadata is not None:
                self.metadata[id_] = metadata[i]
            elif id_ not in self.metadata:
                self.metadata[id_] = {}
    
    def get_vectors(self, ids: Optional[List[Any]] = None) -> Tuple[List[Any], np.ndarray, List[Dict[str, Any]]]:
        """
        Retrieve vectors by IDs.
        
        Args:
            ids: IDs of vectors to retrieve (None for all)
            
        Returns:
            Tuple[List[Any], np.ndarray, List[Dict[str, Any]]]: IDs, vectors, and metadata
        """
        if ids is None:
            ids = list(self.vectors.keys())
        
        result_ids = []
        result_vectors = []
        result_metadata = []
        
        for id_ in ids:
            if id_ in self.vectors:
                result_ids.append(id_)
                result_vectors.append(self.vectors[id_])
                result_metadata.append(self.metadata.get(id_, {}))
        
        if not result_vectors:
            return [], np.array([]), []
        
        return result_ids, np.array(result_vectors), result_metadata
    
    def delete_vectors(self, ids: List[Any]) -> None:
        """
        Delete vectors by IDs.
        
        Args:
            ids: IDs of vectors to delete
        """
        for id_ in ids:
            if id_ in self.vectors:
                del self.vectors[id_]
            if id_ in self.metadata:
                del self.metadata[id_]
    
    def store_index(self, index: Any, name: str = "default") -> None:
        """
        Store an index.
        
        Args:
            index: Index to store
            name: Name of the index
        """
        self.indexes[name] = index
    
    def get_index(self, name: str = "default") -> Any:
        """
        Retrieve an index.
        
        Args:
            name: Name of the index
            
        Returns:
            Any: The retrieved index
        """
        return self.indexes.get(name)
    
    def save(self, path: str) -> None:
        """
        Save the storage to disk.
        
        Args:
            path: Directory path to save storage
        """
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, "vectors.pkl"), "wb") as f:
            pickle.dump(self.vectors, f)
        
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
    
    def load(self, path: str) -> None:
        """
        Load the storage from disk.
        
        Args:
            path: Directory path to load storage from
        """
        vectors_path = os.path.join(path, "vectors.pkl")
        if os.path.exists(vectors_path):
            with open(vectors_path, "rb") as f:
                self.vectors = pickle.load(f)
        
        metadata_path = os.path.join(path, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
