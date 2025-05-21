# fl_rank/preprocessing/tokenizers.py
"""
Tokenizer implementations for tag preprocessing.
"""

import re
from typing import List

from fl_rank.preprocessing.base import Tokenizer


class CompoundTagTokenizer(Tokenizer):
    """
    Tokenizer that splits compound tags by a specified separator.
    """
    
    def __init__(self, separator: str = "/"):
        """
        Initialize the tokenizer.
        
        Args:
            separator: Separator character for compound tags
        """
        self.separator = separator
    
    def tokenize(self, tags: List[str]) -> List[str]:
        """
        Tokenize tags by splitting on the separator.
        
        Args:
            tags: List of tags to tokenize
            
        Returns:
            List[str]: Tokenized tags
        """
        result = []
        for tag in tags:
            if self.separator in tag:
                parts = tag.split(self.separator)
                result.extend(parts)
            else:
                result.append(tag)
        return result


class DefaultTokenizer(Tokenizer):
    """
    Default tokenizer that combines multiple tokenization strategies.
    
    This tokenizer splits on common separators and also handles special cases.
    """
    
    def __init__(self, separators: List[str] = ["/", "-", "."]):
        """
        Initialize the tokenizer.
        
        Args:
            separators: List of separators to split on
        """
        self.separators = separators
    
    def tokenize(self, tags: List[str]) -> List[str]:
        """
        Tokenize tags using multiple strategies.
        
        Args:
            tags: List of tags to tokenize
            
        Returns:
            List[str]: Tokenized tags
        """
        result = []
        for tag in tags:
            # Check if tag contains any of the separators
            needs_splitting = any(sep in tag for sep in self.separators)
            
            if needs_splitting:
                # Replace all separators with a common one for splitting
                temp_tag = tag
                for sep in self.separators:
                    temp_tag = temp_tag.replace(sep, " ")
                
                # Split and add non-empty parts
                parts = [part for part in temp_tag.split() if part]
                result.extend(parts)
            else:
                result.append(tag)
        
        return result
