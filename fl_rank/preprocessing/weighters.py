# fl_rank/preprocessing/weighters.py
"""
Weighter implementations for tag preprocessing.
"""

from typing import Dict, List, Optional

from fl_rank.preprocessing.base import Weighter


class UniformWeighter(Weighter):
    """
    Weighter that assigns uniform weights to all tokens.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize the weighter.
        
        Args:
            weight: Weight to assign to all tokens
        """
        self.uniform_weight = weight
    
    def weight(self, tokens: List[str]) -> Dict[str, float]:
        """
        Assign uniform weights to tokens.
        
        Args:
            tokens: List of tokens to weight
            
        Returns:
            Dict[str, float]: Token weights mapping
        """
        return {token: self.uniform_weight for token in tokens}


class PositionalWeighter(Weighter):
    """
    Weighter that assigns weights based on token position.
    
    Earlier tokens get higher weights.
    """
    
    def __init__(self, initial_weight: float = 1.0, decay_factor: float = 0.9):
        """
        Initialize the weighter.
        
        Args:
            initial_weight: Weight for the first token
            decay_factor: Factor to multiply weight by for each subsequent token
        """
        self.initial_weight = initial_weight
        self.decay_factor = decay_factor
    
    def weight(self, tokens: List[str]) -> Dict[str, float]:
        """
        Assign positional weights to tokens.
        
        Args:
            tokens: List of tokens to weight
            
        Returns:
            Dict[str, float]: Token weights mapping
        """
        weights = {}
        current_weight = self.initial_weight
        
        for token in tokens:
            # If token already has a weight, use the maximum
            weights[token] = max(weights.get(token, 0), current_weight)
            current_weight *= self.decay_factor
        
        return weights


class DefaultWeighter(Weighter):
    """
    Default weighter that applies a simple uniform weighting scheme.
    """
    
    def weight(self, tokens: List[str]) -> Dict[str, float]:
        """
        Apply a simple uniform weighting strategy.
        
        Args:
            tokens: List of tokens to weight
            
        Returns:
            Dict[str, float]: Token weights mapping
        """
        # Simply return a weight of 1.0 for each token
        return {token: 1.0 for token in tokens}