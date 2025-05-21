# fl_rank/ranking/rankers.py
"""
Ranker implementations.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from fl_rank.ranking.base import Ranker, SimilarityMetric
from fl_rank.ranking.metrics import CosineSimilarity
from fl_rank.ranking.strategies import DefaultRankingStrategy


class DefaultRanker(Ranker):
    """
    Default implementation of Ranker.
    """
    
    def __init__(self, metric: Optional[SimilarityMetric] = None):
        """
        Initialize the ranker.
        
        Args:
            metric: Similarity metric to use
        """
        self.strategy = DefaultRankingStrategy()
        
        # If a metric is provided, use it with the strategy
        if metric is not None and hasattr(self.strategy, 'metric'):
            self.strategy.metric = metric
    
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
        return self.strategy.rank(
            query_vector=query_vector,
            candidate_vectors=candidate_vectors,
            candidate_ids=candidate_ids,
            metadata=metadata,
            k=k
        )