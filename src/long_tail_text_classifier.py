from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
from sklearn.utils.validation import check_consistent_length, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from tqdm.auto import tqdm

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

    Chunk size: Since updating the dictionary from label to hash set gets increasingly expensive as the hash set grows,
    we first create a separate ditionary for every "chunk_size" entries and then merge these in the end.
    """

    def __init__(self, min_ngram_length=5, max_ngram_length=50, hash_size=10e8, chunk_size=1000):
        self.min_ngram_length = min_ngram_length
        self.max_ngram_length = max_ngram_length
        self.hash_size = int(hash_size)
        self.chunk_size = chunk_size

    def fit(self, X, y):
        self._check_X_y(X, y)
        n_obs = len(X)

        # This factorization into chunks would also allow for easy parallelization
        groups_list = []
        for chunk_start in tqdm(range(0, n_obs, self.chunk_size)):
            chunk_end = min(chunk_start + self.chunk_size, n_obs)
            chunk_X = X[chunk_start:chunk_end]
            chunk_y = y[chunk_start:chunk_end]
            groups_list.append(
                self.fit_chunk(chunk_X, chunk_y)
            )

        self.groups_ = self.merge_chunks(groups_list)

        self.n_features_in_ = 1  # As required to follow scikit-learn API
        self.classes_ = unique_labels(y)
        return self
    
    def fit_chunk(self, X, y):
        groups = defaultdict(set)
        for text, label in zip(X.flatten(), y):
            hashed_set = self._string_to_hash_set(text)
            groups[label].update(hashed_set)        
        return groups

    @classmethod
    def merge_chunks(cls, groups_list):
        output = defaultdict(set)
        with tqdm(groups_list) as bar:
            for groups in groups_list:
                for k, v in groups.items():
                    output[k].update(v)
                bar.update()

        return output

    def predict(self, X):
        self._check_X(X)
        check_is_fitted(self)

        def inner(text, pbar):
            result = self._predict_single(text)
            pbar.update(1)
            return result

        # Display the predicting as a progress bar
        with tqdm(total=len(X)) as pbar:
            return [
                inner(text, pbar) for text in X.flatten()
            ]

    def _predict_single(self, text):
        encoded = self._string_to_hash_set(text)
        similarity = {
            label: len(encoded.intersection(group_encoded)) for label, group_encoded in self.groups_.items()
        }
        return max(similarity, key=lambda x: similarity[x])

    def _string_to_hash_set(self, text):
        """Returns a function that converts a string to a set of its hashed n-grams."""

        def _custom_hash(ngram):
            return hash(ngram) % int(self.hash_size)
        
        return {
            _custom_hash(text[i:i+n])
            for n in range(self.min_ngram_length, self.max_ngram_length)
            for i in range(len(text)-n+1)
        }
            
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
