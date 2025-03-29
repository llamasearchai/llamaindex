"""
Text tokenization utilities for LlamaIndex.

This module provides functions for tokenizing text for indexing and search.
"""

import re
from typing import List, Set, Optional


def get_default_stopwords() -> Set[str]:
    """
    Get a set of common English stopwords.
    
    Returns:
        Set of stopwords
    """
    return {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
        "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
        "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
    }


def simple_tokenize(
    text: str,
    case_sensitive: bool = False,
    min_length: int = 2,
    stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Split text into tokens using a simple regex-based approach.
    
    Args:
        text: Text to tokenize
        case_sensitive: Whether to preserve case
        min_length: Minimum token length to keep
        stopwords: Set of stopwords to remove (if None, no stopwords are removed)
    
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    # Convert case if needed
    if not case_sensitive:
        text = text.lower()
    
    # Split on non-alphanumeric characters
    tokens = re.findall(r'\b\w+\b', text)
    
    # Filter by length and stopwords
    if stopwords:
        tokens = [
            token for token in tokens 
            if len(token) >= min_length and token not in stopwords
        ]
    else:
        tokens = [token for token in tokens if len(token) >= min_length]
    
    return tokens


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Apply simple stemming to tokens.
    
    This is a very basic implementation - for production use,
    consider using a proper stemming algorithm like Porter stemmer.
    
    Args:
        tokens: List of tokens to stem
        
    Returns:
        List of stemmed tokens
    """
    stemmed = []
    for token in tokens:
        # Simple suffix removal
        if token.endswith('ing'):
            token = token[:-3]
        elif token.endswith('ed'):
            token = token[:-2]
        elif token.endswith('s') and not token.endswith('ss'):
            token = token[:-1]
        elif token.endswith('ly'):
            token = token[:-2]
        elif token.endswith('ment'):
            token = token[:-4]
            
        stemmed.append(token)
    
    return stemmed


def tokenize_and_stem(
    text: str,
    case_sensitive: bool = False,
    min_length: int = 2,
    stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Tokenize text and apply stemming.
    
    Args:
        text: Text to tokenize
        case_sensitive: Whether to preserve case
        min_length: Minimum token length to keep
        stopwords: Set of stopwords to remove
    
    Returns:
        List of stemmed tokens
    """
    tokens = simple_tokenize(text, case_sensitive, min_length, stopwords)
    return stem_tokens(tokens)


def tokenize_with_positions(text: str) -> List[tuple]:
    """
    Tokenize text while preserving the position of each token in the original text.
    
    Args:
        text: The text to tokenize
        
    Returns:
        List of tuples (token, start_pos, end_pos)
    """
    if not text:
        return []
    
    # Find all word tokens with their positions
    token_positions = []
    for match in re.finditer(r'\b\w+\b', text):
        token = match.group(0)
        start_pos = match.start()
        end_pos = match.end()
        token_positions.append((token, start_pos, end_pos))
    
    return token_positions


def sentence_split(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: The text to split into sentences
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Simple regex-based sentence splitting
    # This handles most common cases but isn't perfect for all edge cases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()] 