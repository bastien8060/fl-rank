# fl_rank/utils/serialization.py
"""
Serialization utilities for vectors.
"""

import base64
import json
from typing import Dict, List, Union

import numpy as np


def vectors_to_json(vectors: np.ndarray) -> str:
    """
    Convert vectors to a JSON string.
    
    Args:
        vectors: Array of vectors to convert
        
    Returns:
        str: JSON string representation
    """
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
    
    # Convert to list for JSON serialization
    vectors_list = vectors.tolist()
    
    # Serialize to JSON
    return json.dumps(vectors_list)


def json_to_vectors(json_str: str) -> np.ndarray:
    """
    Convert a JSON string to vectors.
    
    Args:
        json_str: JSON string to convert
        
    Returns:
        np.ndarray: Array of vectors
    """
    # Parse JSON
    vectors_list = json.loads(json_str)
    
    # Convert to NumPy array
    return np.array(vectors_list)


def vectors_to_base64(vectors: np.ndarray) -> str:
    """
    Convert vectors to a base64 string.
    
    Args:
        vectors: Array of vectors to convert
        
    Returns:
        str: Base64 string representation
    """
    # Ensure correct type for binary serialization
    vectors_bytes = vectors.astype(np.float32).tobytes()
    
    # Encode as base64
    return base64.b64encode(vectors_bytes).decode('ascii')


def base64_to_vectors(base64_str: str, shape: tuple) -> np.ndarray:
    """
    Convert a base64 string to vectors.
    
    Args:
        base64_str: Base64 string to convert
        shape: Shape of the resulting array
        
    Returns:
        np.ndarray: Array of vectors
    """
    # Decode base64
    vectors_bytes = base64.b64decode(base64_str)
    
    # Convert to NumPy array
    vectors = np.frombuffer(vectors_bytes, dtype=np.float32)
    
    # Reshape to original shape
    return vectors.reshape(shape)
