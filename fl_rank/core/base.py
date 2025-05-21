# fl_rank/core/base.py
"""
Abstract base classes for fl-rank components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding models.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into embedding vectors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            int: Dimension of the embedding vectors
        """
        pass
    
    @abstractmethod
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length.
        
        Args:
            vectors: Array of vectors to normalize
            
        Returns:
            np.ndarray: Normalized vectors
        """
        pass


class BaseIndex(ABC):
    """
    Abstract base class for vector indexes.
    """
    
    @abstractmethod
    def build(self, vectors: np.ndarray, ids: Optional[List[Any]] = None) -> None:
        """
        Build an index from vectors.
        
        Args:
            vectors: Vectors to index
            ids: Optional list of IDs corresponding to vectors
        """
        pass
    
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: Optional[List[Any]] = None) -> None:
        """
        Add vectors to an existing index.
        
        Args:
            vectors: Vectors to add
            ids: Optional list of IDs corresponding to vectors
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (scores, indices)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save the index
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Path to load the index from
        """
        pass


class BaseStorage(ABC):
    """
    Abstract base class for storage backends.
    """
    
    @abstractmethod
    def store_vectors(self, ids: List[Any], vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Store vectors with their IDs and optional metadata.
        
        Args:
            ids: IDs of the vectors
            vectors: Embedding vectors
            metadata: Optional metadata for each vector
        """
        pass
    
    @abstractmethod
    def get_vectors(self, ids: Optional[List[Any]] = None) -> Tuple[List[Any], np.ndarray, List[Dict[str, Any]]]:
        """
        Retrieve vectors by IDs.
        
        Args:
            ids: IDs of vectors to retrieve (None for all)
            
        Returns:
            Tuple[List[Any], np.ndarray, List[Dict[str, Any]]]: IDs, vectors, and metadata
        """
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[Any]) -> None:
        """
        Delete vectors by IDs.
        
        Args:
            ids: IDs of vectors to delete
        """
        pass
    
    @abstractmethod
    def store_index(self, index: Any, name: str = "default") -> None:
        """
        Store an index.
        
        Args:
            index: Index to store
            name: Name of the index
        """
        pass
    
    @abstractmethod
    def get_index(self, name: str = "default") -> Any:
        """
        Retrieve an index.
        
        Args:
            name: Name of the index
            
        Returns:
            Any: The retrieved index
        """
        pass