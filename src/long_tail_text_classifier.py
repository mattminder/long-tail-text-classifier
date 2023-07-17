from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
from sklearn.utils.validation import check_consistent_length, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels

import numpy as np

class LongTailTextClassifier(BaseEstimator, ClassifierMixin):
    """Classifier that compares all n-grams and returns the class with the most similar n-grams.
    
    1. For all texts, extract all character n-grams within the texts with min_ngram_legnth ≤ n < max_ngram_length.
    2. For the extracted n-grams, calculate an hash code between 0 and hash_size.

    During fit: 
    For all texts in the training set belonging to the same topic, create a set of hash codes that are relevant 
    to said topic by using the set union on all the entries. 

    During predict: 
    Take some text, calculate the set of its hashed n-grams, and calculate the size of its intersection with 
    the "learned" topic-specific sets. We assign the query text to the topic with the largest intersection.
    """

    def __init__(self, min_ngram_length=5, max_ngram_length=50, hash_size=10e8, verbose=False):
        self.min_ngram_length = min_ngram_length
        self.max_ngram_length = max_ngram_length
        self.hash_size = int(hash_size)
        self.verbose = verbose

    def fit(self, X, y):
        self._check_X_y(X, y)
        encoded = self._hash_encode(X)
        self._print_if_verbose("Done with the encoding")
        self.groups_ = self._group_by_y(encoded, y)
        self._print_if_verbose("Done with training")
        self.classes_ = unique_labels(y)

        self.n_features_in_ = 1  # As required to follow scikit-learn API
        return self
    
    def _print_if_verbose(self, what):
        if self.verbose:
            print(what)

    def predict(self, X):
        self._check_X(X)
        check_is_fitted(self)
        return self._predict(X)

    def _predict(self, X):
        encoded = self._hash_encode(X)
        return [
            self._predict_single(query_encoded) for query_encoded in encoded
        ]

    def _predict_single(self, query_encoded):
        similarity = {
            label: len(query_encoded[0].intersection(group_encoded)) for label, group_encoded in self.groups_.items()
        }
        return max(similarity, key=lambda x: similarity[x])

    def _check_X_y(self, X, y):
        self._check_X(X)
        check_consistent_length(X, y)

    def _check_X(self, X):
        check_array(X, dtype=str)
        self._check_has_only_one_feature(X)

    @staticmethod
    def _check_has_only_one_feature(X):
        _, n_features = X.shape
        if n_features > 1:
            raise ValueError("Only a single string feature is supported at this time.")

    def _hash_encode(self, X):
        vectorized_string_to_ngrams = np.vectorize(self._string_to_ngrams_callable())
        return vectorized_string_to_ngrams(X)

    def _custom_hash(self, ngram):
        """Hashs the input ngram to an integer i such that 0 <= i < hash_size."""
        return hash(ngram) % int(self.hash_size)

    def _string_to_ngrams_callable(self):
        """Returns a function that converts a string to a set of its hashed n-grams."""

        def _custom_hash(ngram):
            return hash(ngram) % int(self.hash_size)
        
        def callable(text):
            return {
                _custom_hash(text[i:i+n]) 
                for n in range(self.min_ngram_length, self.max_ngram_length)
                for i in range(len(text)-n+1)
            }
        
        return callable
    
    def _group_by_y(self, encoded, y):
        """Groups the encodigns by y and aggregates with the union operation."""
        groups = defaultdict(set)

        for (some_encoded, some_y) in zip(encoded.flatten(), y):
            groups[some_y].update(some_encoded)
        return groups
