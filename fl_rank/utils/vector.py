# fl_rank/utils/vector.py
"""
Vector manipulation utilities.
"""

from typing import List, Optional, Union

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length (L2 normalization).
    
    Args:
        vectors: Array of vectors to normalize
        
    Returns:
        np.ndarray: Normalized vectors
    """
    # Handle single vector case
    if len(vectors.shape) == 1:
        return normalize_vector(vectors)
    
    # Compute L2 norms
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms = np.maximum(norms, np.finfo(vectors.dtype).eps)
    
    # Normalize
    return vectors / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a single vector to unit length (L2 normalization).
    
    Args:
        vector: Vector to normalize
        
    Returns:
        np.ndarray: Normalized vector
    """
    # Compute L2 norm
    norm = np.linalg.norm(vector)
    
    # Avoid division by zero
    norm = max(norm, np.finfo(vector.dtype).eps)
    
    # Normalize
    return vector / norm


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        float: Cosine similarity
    """
    # Normalize vectors
    vector1_norm = normalize_vector(vector1)
    vector2_norm = normalize_vector(vector2)
    
    # Compute dot product
    return float(np.dot(vector1_norm, vector2_norm))


def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        float: Euclidean distance
    """
    return float(np.linalg.norm(vector1 - vector2))


def dot_product(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate dot product between two vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        float: Dot product
    """
    return float(np.dot(vector1, vector2))


def batch_cosine_similarity(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between batches of vectors.
    
    Args:
        vectors1: First batch of vectors
        vectors2: Second batch of vectors
        
    Returns:
        np.ndarray: Matrix of cosine similarities
    """
    # Normalize vectors
    vectors1_norm = normalize_vectors(vectors1)
    vectors2_norm = normalize_vectors(vectors2)
    
    # Compute dot products
    return np.dot(vectors1_norm, vectors2_norm.T)