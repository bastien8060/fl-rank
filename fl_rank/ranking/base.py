# fl_rank/ranking/base.py
"""
Base interfaces for ranking and scoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import numpy as np


class SimilarityMetric(ABC):
    """
    Abstract base class for similarity metrics.
    
    Similarity metrics determine the similarity between vectors.
    """
    
    @abstractmethod
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Similarity score
        """
        pass
    
    @abstractmethod
    def compute_batch(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute similarity between a query vector and multiple vectors.
        
        Args:
            query_vector: Query vector
            vectors: Matrix of vectors to compare against
            
        Returns:
            np.ndarray: Array of similarity scores
        """
        pass
    
    @abstractmethod
    def requires_normalization(self) -> bool:
        """
        Whether this metric requires normalized vectors.
        
        Returns:
            bool: True if normalization is required
        """
        pass
    
    @abstractmethod
    def higher_is_better(self) -> bool:
        """
        Whether higher scores indicate better matches.
        
        Returns:
            bool: True if higher scores are better
        """
        pass


class RankingStrategy(ABC):
    """
    Abstract base class for ranking strategies.
    
    Ranking strategies determine how to rank items by similarity.
    """
    
    @abstractmethod
    def rank(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        candidate_ids: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates by similarity to query.
        
        Args:
            query_vector: Query vector
            candidate_vectors: Candidate vectors
            candidate_ids: Candidate IDs
            metadata: Optional metadata for candidates
            k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Ranked results with scores
        """
        pass


class Ranker(ABC):
    """
    Abstract base class for rankers.
    
    Rankers coordinate the ranking process.
    """
    
    @abstractmethod
    def find_similar(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        candidate_ids: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find items similar to query.
        
        Args:
            query_vector: Query vector
            candidate_vectors: Candidate vectors
            candidate_ids: Candidate IDs
            metadata: Optional metadata for candidates
            k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Similar items with scores
        """
        pass