import numpy as np
import torch

from joblib import Parallel, delayed
import multiprocessing
from time import time
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted


def _inner_has(attr):
    """available_if check: expose the method only when the inner estimator
    supports it. Lets Pipeline / CalibratedClassifierCV's hasattr checks
    pass through to the wrapped model."""
    def check(self):
        return hasattr(self.estimator, attr)
    return check

class ParallelEstimatorWrapper(ClassifierMixin, BaseEstimator):   
    def __init__(
        self, 
        estimator,
        batch_size,
        device,
        verbose=True
    ):
        self.estimator = estimator
        self.batch_size = batch_size
        self.device = device
        self.n_jobs = min(multiprocessing.cpu_count(), 16)
        self.verbose = verbose

    def fit(self, X, y, **fit_params): 
        self.estimator_ = clone(self.estimator)

        if self.verbose:
            start_time = time()

        self.estimator_.fit(X, y, **fit_params)

        if self.verbose:
            print(f"Train time: {time() - start_time}")

        self.is_fitted_ = True 
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X): 
        check_is_fitted(self, "is_fitted_")
        
        n_samples = len(X)
        preds = []

        batches = [
            (start, min(start + self.batch_size, n_samples))
            for start in range(0, n_samples, self.batch_size)
        ]

        def _predict_batch(start, end):
            return self.estimator_.predict(X[start:end])
        
        if self.device == "cpu":
            # Run batches in parallel
            pred_list = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_predict_batch)(start, end) for start, end in batches
            )
        else:
            pred_list = []

            with torch.no_grad():
                for start, end in tqdm(batches, desc="Batch processing"):
                    pred_list.append(_predict_batch(start, end))

                    # Free GPU memory
                    self.free_gpu_memory()

        # Concatenate results in correct order; squeeze (N,1) → (N,) for binary classifiers
        preds = np.concatenate(pred_list, axis=0)
        return preds.squeeze(axis=1) if preds.ndim == 2 and preds.shape[1] == 1 else preds

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError(f"{self.estimator_.__class__.__name__} does not support predict_proba().")
        
        n_samples = len(X)
        probs = []

        batches = [
            (start, min(start + self.batch_size, n_samples))
            for start in range(0, n_samples, self.batch_size)
        ]

        def _predict_batch(start, end):
            return self.estimator_.predict_proba(X[start:end])
        
        if self.verbose:
            start_time = time()
        
        if self.device == "cpu":
            # Run batches in parallel
            probs_list = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_predict_batch)(start, end) for start, end in batches
            )
        else:
            probs_list = []

            with torch.no_grad():
                for start, end in tqdm(batches, desc="Batch processing"):
                    probs_list.append(_predict_batch(start, end))

                    # Free GPU memory
                    self.free_gpu_memory()

        # Concatenate results in correct order
        probs = np.concatenate(probs_list, axis=0)

        if self.verbose:
            print(f"Inference time: {time() - start_time}")

        return probs

    @available_if(_inner_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self, "is_fitted_")

        n_samples = len(X)

        batches = [
            (start, min(start + self.batch_size, n_samples))
            for start in range(0, n_samples, self.batch_size)
        ]

        def _score_batch(start, end):
            return self.estimator_.decision_function(X[start:end])

        if self.verbose:
            start_time = time()

        if self.device == "cpu":
            score_list = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_score_batch)(start, end) for start, end in batches
            )
        else:
            score_list = []
            with torch.no_grad():
                for start, end in tqdm(batches, desc="Batch processing"):
                    score_list.append(_score_batch(start, end))
                    self.free_gpu_memory()

        scores = np.concatenate(score_list, axis=0)

        if self.verbose:
            print(f"Inference time: {time() - start_time}")

        return scores

    def free_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()