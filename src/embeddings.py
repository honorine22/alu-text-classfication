"""
Embedding Implementations for E-commerce Text Classification

This module implements three embedding techniques:
1. TF-IDF (Term Frequency-Inverse Document Frequency)
2. Word2Vec (Skip-gram and CBOW)
3. GloVe (Global Vectors for Word Representation)

References:
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space.
- Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation.
"""

import os
import pickle
import zipfile
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from tqdm import tqdm


class BaseEmbedding(ABC):
    """Abstract base class for all embedding implementations."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, texts: List[str]) -> 'BaseEmbedding':
        """Fit the embedding model on texts."""
        pass
    
    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to embeddings."""
        pass
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get embedding parameters."""
        pass
    
    def save(self, filepath: str):
        """Save embedding model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseEmbedding':
        """Load embedding model."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class TFIDFEmbedding(BaseEmbedding):
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) Embedding.
    
    TF-IDF captures the importance of terms in documents relative to a corpus.
    It's effective for text classification as it down-weights common terms and
    emphasizes discriminative terms.
    
    Attributes:
        max_features: Maximum number of features (vocabulary size)
        ngram_range: Range of n-grams to extract
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency for terms
        
    Reference:
        Salton, G., & Buckley, C. (1988). Term-weighting approaches in 
        automatic text retrieval. Information Processing & Management.
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        """
        Initialize TF-IDF embedding.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: N-gram range (e.g., (1, 2) for unigrams and bigrams)
            min_df: Ignore terms appearing in fewer than min_df documents
            max_df: Ignore terms appearing in more than max_df proportion of documents
            sublinear_tf: Apply sublinear scaling (1 + log(tf))
        """
        super().__init__(name='TF-IDF')
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf
        )
    
    def fit(self, texts: List[str]) -> 'TFIDFEmbedding':
        """
        Fit TF-IDF vectorizer on texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Self
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        self.vocabulary_size = len(self.vectorizer.vocabulary_)
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary)."""
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top n features by IDF score."""
        feature_names = self.get_feature_names()
        idf_scores = self.vectorizer.idf_
        top_indices = np.argsort(idf_scores)[::-1][:n]
        return [(feature_names[i], idf_scores[i]) for i in top_indices]
    
    def get_params(self) -> Dict[str, Any]:
        """Get TF-IDF parameters."""
        return {
            'name': self.name,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'sublinear_tf': self.sublinear_tf,
            'vocabulary_size': getattr(self, 'vocabulary_size', None)
        }


class Word2VecEmbedding(BaseEmbedding):
    """
    Word2Vec Embedding using Gensim.
    
    Word2Vec learns dense vector representations of words that capture semantic
    relationships. Documents are represented as the average of their word vectors.
    
    Attributes:
        vector_size: Dimensionality of word vectors
        window: Context window size
        min_count: Minimum word frequency
        sg: 1 for Skip-gram, 0 for CBOW
        workers: Number of worker threads
        
    Reference:
        Mikolov, T., et al. (2013). Efficient Estimation of Word Representations 
        in Vector Space. arXiv preprint arXiv:1301.3781.
    """
    
    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,  # 1 = Skip-gram, 0 = CBOW
        workers: int = 4,
        epochs: int = 10
    ):
        """
        Initialize Word2Vec embedding.
        
        Args:
            vector_size: Dimensionality of embeddings
            window: Context window size
            min_count: Ignore words with frequency < min_count
            sg: Training algorithm (1=Skip-gram, 0=CBOW)
            workers: Number of worker threads
            epochs: Number of training epochs
        """
        super().__init__(name='Word2Vec')
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.epochs = epochs
        self.model = None
    
    def fit(self, texts: List[str]) -> 'Word2VecEmbedding':
        """
        Train Word2Vec model on texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Self
        """
        # Tokenize texts for Word2Vec
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec model
        # Note: workers=1 avoids numpy/gensim compatibility issues with BLAS
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=1,  # Single thread to avoid numpy 2.x compatibility issues
            epochs=self.epochs
        )
        
        self.is_fitted = True
        self.vocabulary_size = len(self.model.wv)
        return self
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a single word."""
        if word in self.model.wv:
            return self.model.wv[word]
        return None
    
    def get_document_vector(self, text: str) -> np.ndarray:
        """
        Get document vector as average of word vectors.
        
        Args:
            text: Preprocessed text string
            
        Returns:
            Document embedding vector
        """
        words = text.split()
        word_vectors = []
        
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            # Return zero vector if no words found
            return np.zeros(self.vector_size)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to document vectors.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Array of document embeddings
        """
        if not self.is_fitted:
            raise ValueError("Word2Vec model not fitted. Call fit() first.")
        
        embeddings = []
        for text in tqdm(texts, desc="Generating Word2Vec embeddings"):
            embeddings.append(self.get_document_vector(text))
        
        return np.array(embeddings)
    
    def get_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Get most similar words."""
        if word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []
    
    def get_params(self) -> Dict[str, Any]:
        """Get Word2Vec parameters."""
        return {
            'name': self.name,
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'sg': 'Skip-gram' if self.sg == 1 else 'CBOW',
            'epochs': self.epochs,
            'vocabulary_size': getattr(self, 'vocabulary_size', None)
        }


class GloVeEmbedding(BaseEmbedding):
    """
    GloVe (Global Vectors) Pre-trained Embedding.
    
    GloVe embeddings are pre-trained on large corpora and capture global
    co-occurrence statistics. Documents are represented as the average
    of their word vectors.
    
    Attributes:
        glove_path: Path to GloVe embeddings file
        vector_size: Dimensionality of embeddings (50, 100, 200, or 300)
        
    Reference:
        Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global 
        Vectors for Word Representation. EMNLP.
    """
    
    def __init__(
        self,
        glove_path: Optional[str] = None,
        vector_size: int = 100
    ):
        """
        Initialize GloVe embedding.
        
        Args:
            glove_path: Path to GloVe file (e.g., 'glove.6B.100d.txt')
            vector_size: Embedding dimension (must match GloVe file)
        """
        super().__init__(name='GloVe')
        self.glove_path = glove_path
        self.vector_size = vector_size
        self.embeddings_index = {}
    
    def load_glove(self, glove_path: str) -> Dict[str, np.ndarray]:
        """
        Load GloVe embeddings from file.
        
        Args:
            glove_path: Path to GloVe text file
            
        Returns:
            Dictionary mapping words to vectors
        """
        embeddings_index = {}
        print(f"Loading GloVe embeddings from {glove_path}...")
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading GloVe"):
                values = line.split()
                word = values[0]
                try:
                    vector = np.asarray(values[1:], dtype='float32')
                    if len(vector) == self.vector_size:
                        embeddings_index[word] = vector
                except ValueError:
                    continue
        
        print(f"Loaded {len(embeddings_index)} word vectors")
        return embeddings_index
    
    def fit(self, texts: List[str] = None) -> 'GloVeEmbedding':
        """
        Load GloVe embeddings (texts parameter ignored for pre-trained).
        
        Args:
            texts: Ignored for pre-trained embeddings
            
        Returns:
            Self
        """
        if self.glove_path and os.path.exists(self.glove_path):
            self.embeddings_index = self.load_glove(self.glove_path)
        else:
            print("Warning: GloVe path not found. Using random initialization.")
            print("Download GloVe from: https://nlp.stanford.edu/projects/glove/")
            print("Expected file: glove.6B.100d.txt (or similar)")
        
        self.is_fitted = True
        self.vocabulary_size = len(self.embeddings_index)
        return self
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a single word."""
        return self.embeddings_index.get(word)
    
    def get_document_vector(self, text: str) -> np.ndarray:
        """
        Get document vector as average of word vectors.
        
        Args:
            text: Preprocessed text string
            
        Returns:
            Document embedding vector
        """
        words = text.lower().split()
        word_vectors = []
        
        for word in words:
            if word in self.embeddings_index:
                word_vectors.append(self.embeddings_index[word])
        
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to document vectors.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Array of document embeddings
        """
        if not self.is_fitted:
            raise ValueError("GloVe embeddings not loaded. Call fit() first.")
        
        embeddings = []
        for text in tqdm(texts, desc="Generating GloVe embeddings"):
            embeddings.append(self.get_document_vector(text))
        
        return np.array(embeddings)
    
    def get_coverage(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate vocabulary coverage statistics.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Coverage statistics
        """
        all_words = set()
        covered_words = set()
        
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
            covered_words.update(w for w in words if w in self.embeddings_index)
        
        return {
            'total_unique_words': len(all_words),
            'covered_words': len(covered_words),
            'coverage_ratio': len(covered_words) / max(len(all_words), 1),
            'oov_words': len(all_words) - len(covered_words)
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get GloVe parameters."""
        return {
            'name': self.name,
            'vector_size': self.vector_size,
            'glove_path': self.glove_path,
            'vocabulary_size': getattr(self, 'vocabulary_size', None)
        }


def create_embedding(embedding_type: str, **kwargs) -> BaseEmbedding:
    """
    Factory function to create embedding instances.
    
    Args:
        embedding_type: Type of embedding ('tfidf', 'word2vec', 'glove')
        **kwargs: Embedding-specific parameters
        
    Returns:
        Embedding instance
    """
    embedding_classes = {
        'tfidf': TFIDFEmbedding,
        'word2vec': Word2VecEmbedding,
        'glove': GloVeEmbedding
    }
    
    if embedding_type.lower() not in embedding_classes:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    return embedding_classes[embedding_type.lower()](**kwargs)


def compare_embeddings_coverage(
    texts: List[str],
    embeddings: List[BaseEmbedding]
) -> pd.DataFrame:
    """
    Compare vocabulary coverage across embeddings.
    
    Args:
        texts: List of preprocessed texts
        embeddings: List of embedding instances
        
    Returns:
        DataFrame with coverage comparison
    """
    results = []
    for emb in embeddings:
        if hasattr(emb, 'get_coverage'):
            coverage = emb.get_coverage(texts)
            coverage['embedding'] = emb.name
            results.append(coverage)
    
    return pd.DataFrame(results) if results else pd.DataFrame()
