"""
Text Preprocessing Module for E-commerce Text Classification

This module provides comprehensive text preprocessing functionality with
embedding-specific adaptations for TF-IDF, Word2Vec, and GloVe embeddings.

References:
- Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python.
- Manning, C. D., & Schütze, H. (1999). Foundations of Statistical NLP.
"""

import re
import string
from typing import List, Optional, Callable
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data
def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

download_nltk_resources()


class TextPreprocessor:
    """
    Text preprocessing class with embedding-specific adaptations.
    
    Attributes:
        embedding_type: Type of embedding ('tfidf', 'word2vec', 'glove')
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to lemmatize tokens
        stem: Whether to stem tokens (mutually exclusive with lemmatize)
        min_word_length: Minimum word length to keep
        
    Note:
        Different embedding techniques benefit from different preprocessing:
        - TF-IDF: Benefits from aggressive preprocessing (stopword removal, stemming)
        - Word2Vec: Benefits from preserving more context (minimal stopword removal)
        - GloVe: Pre-trained embeddings work best with standard tokenization
    """
    
    def __init__(
        self,
        embedding_type: str = 'tfidf',
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
        min_word_length: int = 2,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize the preprocessor with embedding-specific settings.
        
        Args:
            embedding_type: Type of embedding to optimize for
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens
            stem: Whether to stem tokens
            min_word_length: Minimum word length to keep
            custom_stopwords: Additional stopwords to remove
        """
        self.embedding_type = embedding_type.lower()
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_word_length = min_word_length
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = PorterStemmer() if stem else None
        
        # Build stopwords set
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
            
        # Apply embedding-specific configurations
        self._configure_for_embedding()
    
    def _configure_for_embedding(self):
        """Apply embedding-specific preprocessing configurations."""
        if self.embedding_type == 'tfidf':
            # TF-IDF benefits from aggressive preprocessing
            # Remove stopwords to focus on discriminative terms
            self.remove_stopwords = True
            self.lemmatize = True
            
        elif self.embedding_type == 'word2vec':
            # Word2Vec captures semantic relationships
            # Keep some context words for better embeddings
            self.remove_stopwords = False  # Keep context
            self.lemmatize = True
            self.min_word_length = 1
            
        elif self.embedding_type == 'glove':
            # GloVe pre-trained embeddings expect standard tokenization
            # Minimal preprocessing to match training vocabulary
            self.remove_stopwords = False
            self.lemmatize = False
            self.stem = False
            self.min_word_length = 1
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback to simple split if NLTK data not available
            tokens = text.split()
        return tokens
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered token list
        """
        return [t for t in tokens if t not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized token list
        """
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens using Porter Stemmer.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Stemmed token list
        """
        if self.stemmer:
            return [self.stemmer.stem(t) for t in tokens]
        return tokens
    
    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by minimum length.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered token list
        """
        return [t for t in tokens if len(t) >= self.min_word_length]
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline optimized for the embedding type.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Apply stopword removal if configured
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Apply lemmatization or stemming (not both)
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        elif self.stem:
            tokens = self.stem_tokens(tokens)
        
        # Filter by length
        tokens = self.filter_by_length(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_tokens(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline returning tokens.
        
        Args:
            text: Raw text string
            
        Returns:
            List of preprocessed tokens
        """
        preprocessed = self.preprocess(text)
        return preprocessed.split() if preprocessed else []
    
    def preprocess_series(self, series: pd.Series, show_progress: bool = True) -> pd.Series:
        """
        Preprocess a pandas Series of text.
        
        Args:
            series: Pandas Series containing text
            show_progress: Whether to show progress bar
            
        Returns:
            Preprocessed Series
        """
        if show_progress:
            try:
                from tqdm import tqdm
                tqdm.pandas()
                return series.progress_apply(self.preprocess)
            except ImportError:
                pass
        return series.apply(self.preprocess)
    
    def get_vocabulary(self, texts: List[str]) -> set:
        """
        Extract vocabulary from preprocessed texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Set of unique tokens
        """
        vocab = set()
        for text in texts:
            vocab.update(text.split())
        return vocab
    
    def get_preprocessing_stats(self, original: pd.Series, preprocessed: pd.Series) -> dict:
        """
        Calculate preprocessing statistics.
        
        Args:
            original: Original text series
            preprocessed: Preprocessed text series
            
        Returns:
            Dictionary of statistics
        """
        def avg_word_count(series):
            return series.str.split().str.len().mean()
        
        def avg_char_count(series):
            return series.str.len().mean()
        
        original_vocab = self.get_vocabulary(original.dropna().tolist())
        preprocessed_vocab = self.get_vocabulary(preprocessed.dropna().tolist())
        
        return {
            'original_avg_words': avg_word_count(original.fillna('')),
            'preprocessed_avg_words': avg_word_count(preprocessed.fillna('')),
            'original_avg_chars': avg_char_count(original.fillna('')),
            'preprocessed_avg_chars': avg_char_count(preprocessed.fillna('')),
            'original_vocab_size': len(original_vocab),
            'preprocessed_vocab_size': len(preprocessed_vocab),
            'vocab_reduction': 1 - len(preprocessed_vocab) / max(len(original_vocab), 1)
        }


def create_preprocessor_for_embedding(embedding_type: str) -> TextPreprocessor:
    """
    Factory function to create embedding-specific preprocessors.
    
    Args:
        embedding_type: Type of embedding ('tfidf', 'word2vec', 'glove')
        
    Returns:
        Configured TextPreprocessor instance
        
    Justification:
        - TF-IDF: Aggressive preprocessing improves discrimination between classes
          by focusing on unique, meaningful terms (Salton & Buckley, 1988).
        - Word2Vec: Preserving context words helps learn better semantic relationships
          as the model relies on surrounding words (Mikolov et al., 2013).
        - GloVe: Pre-trained embeddings have fixed vocabulary; minimal preprocessing
          ensures better vocabulary coverage (Pennington et al., 2014).
    """
    return TextPreprocessor(embedding_type=embedding_type)
