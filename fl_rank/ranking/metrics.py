# fl_rank/ranking/metrics.py
"""
Similarity metric implementations.
"""

from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

from fl_rank.ranking.base import SimilarityMetric


class CosineSimilarity(SimilarityMetric):
    """
    Cosine similarity metric.
    
    Measures the cosine of the angle between two vectors.
    """
    
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Cosine similarity
        """
        # Normalize vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vector1, vector2) / (norm1 * norm2)
    
    def compute_batch(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query vector and multiple vectors.
        
        Args:
            query_vector: Query vector
            vectors: Matrix of vectors to compare against
            
        Returns:
            np.ndarray: Array of cosine similarities
        """
        # Reshape query vector if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_norm = np.maximum(query_norm, np.finfo(query_vector.dtype).eps)
        query_normalized = query_vector / query_norm
        
        # Normalize candidate vectors
        candidates_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        candidates_norm = np.maximum(candidates_norm, np.finfo(vectors.dtype).eps)
        candidates_normalized = vectors / candidates_norm
        
        # Compute dot product of normalized vectors
        return np.dot(query_normalized, candidates_normalized.T).flatten()
    
    def requires_normalization(self) -> bool:
        """
        Whether this metric requires normalized vectors.
        
        Returns:
            bool: True if normalization is required
        """
        return True
    
    def higher_is_better(self) -> bool:
        """
        Whether higher scores indicate better matches.
        
        Returns:
            bool: True if higher scores are better
        """
        return True


class EuclideanDistance(SimilarityMetric):
    """
    Euclidean distance metric.
    
    Measures the straight-line distance between two vectors.
    """
    
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Euclidean distance
        """
        return float(np.linalg.norm(vector1 - vector2))
    
    def compute_batch(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance between a query vector and multiple vectors.
        
        Args:
            query_vector: Query vector
            vectors: Matrix of vectors to compare against
            
        Returns:
            np.ndarray: Array of Euclidean distances
        """
        # Reshape query vector if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Compute squared differences
        diff = vectors - query_vector
        squared_diff = np.square(diff)
        squared_distances = np.sum(squared_diff, axis=1)
        
        # Return square root of squared distances
        return np.sqrt(squared_distances)
    
    def requires_normalization(self) -> bool:
        """
        Whether this metric requires normalized vectors.
        
        Returns:
            bool: True if normalization is required
        """
        return False
    
    def higher_is_better(self) -> bool:
        """
        Whether higher scores indicate better matches.
        
        Returns:
            bool: True if higher scores are better
        """
        return False


class DotProduct(SimilarityMetric):
    """
    Dot product metric.
    
    Measures the dot product between two vectors.
    """
    
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute dot product between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Dot product
        """
        return float(np.dot(vector1, vector2))
    
    def compute_batch(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute dot product between a query vector and multiple vectors.
        
        Args:
            query_vector: Query vector
            vectors: Matrix of vectors to compare against
            
        Returns:
            np.ndarray: Array of dot products
        """
        # Reshape query vector if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Compute dot products
        return np.dot(query_vector, vectors.T).flatten()
    
    def requires_normalization(self) -> bool:
        """
        Whether this metric requires normalized vectors.
        
        Returns:
            bool: True if normalization is required
        """
        return False
    
    def higher_is_better(self) -> bool:
        """
        Whether higher scores indicate better matches.
        
        Returns:
            bool: True if higher scores are better
        """
        return True


class WeightedCosineSimilarity(SimilarityMetric):
    """
    Weighted cosine similarity metric.
    
    Applies weights to dimensions before computing cosine similarity.
    """
    
    def __init__(self, weights: Optional[np.ndarray] = None):
        """
        Initialize the metric.
        
        Args:
            weights: Weights for each dimension
        """
        self.weights = weights
    
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute weighted cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Weighted cosine similarity
        """
        if self.weights is not None:
            # Apply weights
            vector1 = vector1 * self.weights
            vector2 = vector2 * self.weights
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vector1, vector2) / (norm1 * norm2)
    
    def compute_batch(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute weighted cosine similarity between a query vector and multiple vectors.
        
        Args:
            query_vector: Query vector
            vectors: Matrix of vectors to compare against
            
        Returns:
            np.ndarray: Array of weighted cosine similarities
        """
        # Reshape query vector if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if self.weights is not None:
            # Apply weights
            weighted_query = query_vector * self.weights
            weighted_vectors = vectors * self.weights
        else:
            weighted_query = query_vector
            weighted_vectors = vectors
        
        # Normalize query vector
        query_norm = np.linalg.norm(weighted_query, axis=1, keepdims=True)
        query_norm = np.maximum(query_norm, np.finfo(weighted_query.dtype).eps)
        query_normalized = weighted_query / query_norm
        
        # Normalize candidate vectors
        candidates_norm = np.linalg.norm(weighted_vectors, axis=1, keepdims=True)
        candidates_norm = np.maximum(candidates_norm, np.finfo(weighted_vectors.dtype).eps)
        candidates_normalized = weighted_vectors / candidates_norm
        
        # Compute dot product of normalized vectors
        return np.dot(query_normalized, candidates_normalized.T).flatten()
    
    def requires_normalization(self) -> bool:
        """
        Whether this metric requires normalized vectors.
        
        Returns:
            bool: True if normalization is required
        """
        return True
    
    def higher_is_better(self) -> bool:
        """
        Whether higher scores indicate better matches.
        
        Returns:
            bool: True if higher scores are better
        """
        return True