# fl_rank/ranking/strategies.py
"""
Ranking strategy implementations.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import numpy as np

from fl_rank.ranking.base import RankingStrategy, SimilarityMetric
from fl_rank.ranking.metrics import CosineSimilarity


class BasicRankingStrategy(RankingStrategy):
    """
    Basic ranking strategy using a similarity metric.
    """
    
    def __init__(self, metric: Optional[SimilarityMetric] = None):
        """
        Initialize the strategy.
        
        Args:
            metric: Similarity metric to use
        """
        self.metric = metric or CosineSimilarity()
    
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
        # Compute similarity scores
        scores = self.metric.compute_batch(query_vector, candidate_vectors)
        
        # Sort by score (handle both higher-is-better and lower-is-better metrics)
        if self.metric.higher_is_better():
            indices = np.argsort(-scores)  # Descending order
        else:
            indices = np.argsort(scores)  # Ascending order
        
        # Limit to top k
        indices = indices[:k]
        
        # Prepare results
        results = []
        for idx in indices:
            result = {"id": candidate_ids[idx], "score": float(scores[idx])}
            
            # Add metadata if available
            if metadata is not None and idx < len(metadata):
                result.update(metadata[idx])
            
            results.append(result)
        
        return results


class WeightedRankingStrategy(RankingStrategy):
    """
    Ranking strategy that applies weights to scores based on metadata.
    """
    
    def __init__(
        self,
        metric: Optional[SimilarityMetric] = None,
        weight_fn: Optional[Callable[[Dict[str, Any]], float]] = None
    ):
        """
        Initialize the strategy.
        
        Args:
            metric: Similarity metric to use
            weight_fn: Function to compute weight from metadata
        """
        self.metric = metric or CosineSimilarity()
        self.weight_fn = weight_fn or (lambda _: 1.0)
    
    def rank(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        candidate_ids: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates by weighted similarity to query.
        
        Args:
            query_vector: Query vector
            candidate_vectors: Candidate vectors
            candidate_ids: Candidate IDs
            metadata: Optional metadata for candidates
            k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Ranked results with scores
        """
        # Compute similarity scores
        scores = self.metric.compute_batch(query_vector, candidate_vectors)
        
        # Apply weights based on metadata
        if metadata:
            weights = np.array([self.weight_fn(meta) for meta in metadata])
            weighted_scores = scores * weights
        else:
            weighted_scores = scores
        
        # Sort by weighted score
        if self.metric.higher_is_better():
            indices = np.argsort(-weighted_scores)  # Descending order
        else:
            indices = np.argsort(weighted_scores)  # Ascending order
        
        # Limit to top k
        indices = indices[:k]
        
        # Prepare results
        results = []
        for idx in indices:
            result = {
                "id": candidate_ids[idx],
                "score": float(scores[idx]),
                "weighted_score": float(weighted_scores[idx])
            }
            
            # Add metadata if available
            if metadata is not None and idx < len(metadata):
                result.update(metadata[idx])
            
            results.append(result)
        
        return results


class ReRankingStrategy(RankingStrategy):
    """
    Strategy that applies a two-phase ranking process.
    
    First, candidates are ranked by a primary metric.
    Then, the top candidates are re-ranked by a secondary metric.
    """
    
    def __init__(
        self,
        primary_metric: Optional[SimilarityMetric] = None,
        secondary_metric: Optional[SimilarityMetric] = None,
        prefilter_k: int = 100
    ):
        """
        Initialize the strategy.
        
        Args:
            primary_metric: Primary similarity metric for initial ranking
            secondary_metric: Secondary similarity metric for re-ranking
            prefilter_k: Number of results to keep after initial ranking
        """
        self.primary_metric = primary_metric or CosineSimilarity()
        self.secondary_metric = secondary_metric or self.primary_metric
        self.prefilter_k = prefilter_k
    
    def rank(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        candidate_ids: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates using two-phase process.
        
        Args:
            query_vector: Query vector
            candidate_vectors: Candidate vectors
            candidate_ids: Candidate IDs
            metadata: Optional metadata for candidates
            k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Ranked results with scores
        """
        # Phase 1: Initial ranking with primary metric
        primary_scores = self.primary_metric.compute_batch(query_vector, candidate_vectors)
        
        # Sort by primary score
        if self.primary_metric.higher_is_better():
            indices = np.argsort(-primary_scores)  # Descending order
        else:
            indices = np.argsort(primary_scores)  # Ascending order
        
        # Limit to top prefilter_k
        prefilter_k = min(self.prefilter_k, len(indices))
        prefilter_indices = indices[:prefilter_k]
        
        # Phase 2: Re-ranking with secondary metric
        prefiltered_vectors = candidate_vectors[prefilter_indices]
        secondary_scores = self.secondary_metric.compute_batch(query_vector, prefiltered_vectors)
        
        # Sort by secondary score
        if self.secondary_metric.higher_is_better():
            rerank_indices = np.argsort(-secondary_scores)  # Descending order
        else:
            rerank_indices = np.argsort(secondary_scores)  # Ascending order
        
        # Limit to top k
        final_indices = rerank_indices[:k]
        
        # Map back to original indices
        original_indices = prefilter_indices[final_indices]
        
        # Prepare results
        results = []
        for i, idx in enumerate(original_indices):
            result = {
                "id": candidate_ids[idx],
                "primary_score": float(primary_scores[idx]),
                "secondary_score": float(secondary_scores[final_indices[i]]),
                "score": float(secondary_scores[final_indices[i]])  # Use secondary as final score
            }
            
            # Add metadata if available
            if metadata is not None and idx < len(metadata):
                result.update(metadata[idx])
            
            results.append(result)
        
        return results


class DefaultRankingStrategy(BasicRankingStrategy):
    """
    Default ranking strategy for most use cases.
    
    This is a convenience class that uses BasicRankingStrategy with CosineSimilarity.
    """
    
    def __init__(self):
        """
        Initialize the default strategy.
        """
        super().__init__(metric=CosineSimilarity())