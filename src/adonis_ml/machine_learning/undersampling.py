import logging

from imblearn.under_sampling import RandomUnderSampler
import numpy as np

logger = logging.getLogger(__name__)

class NonRandomUnderSampler(RandomUnderSampler):
    def __init__(
        self, 
        sampling_column="selected_window", 
        data_fraction=1.0,
        sampling_strategy="auto",
        random_state=None, 
        replacement=False,
        verbose=True,  
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            replacement=replacement
        )

        self.sampling_column = sampling_column
        self.verbose = verbose
        self.data_fraction = data_fraction

    def _fit_resample(self, X, y):
        if self.verbose:
            logger.info("Before undersampling: input size = %d", X.shape[0])

        train_idx = (X[self.sampling_column] == 1)
        X_res, y_res = X[train_idx], y[train_idx]

        if self.data_fraction < 1.0:
            arr = self.sampling_column.split("_")
            groups_col = f"groups_knn_{arr[2]}"
            unique_groups = X_res[groups_col].unique()
            n_keep = max(1, int(np.ceil(len(unique_groups) * self.data_fraction)))
            rng = np.random.default_rng(self.random_state)
            kept_groups = rng.choice(unique_groups, size=n_keep, replace=False)
            mask = X_res[groups_col].isin(kept_groups)
            X_res, y_res = X_res[mask], y_res[mask]
            if self.verbose:
                logger.info("Data fraction %.2f: kept %d/%d groups", self.data_fraction, n_keep, len(unique_groups))

        if self.verbose:
            unique, counts = np.unique(y_res, return_counts=True)
            logger.info("After undersampling: input size = %d", X_res.shape[0])
            logger.info("After undersampling: class balance = %s", dict(zip(unique, counts)))
        return X_res, y_res