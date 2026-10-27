import numpy as np
from sklearn.model_selection import GroupKFold


class GroupKFoldFractional:
    """Outer CV wrapper: for each split, filter training groups to a disjoint subset.

    K = n_subsets disjoint chunks of groups partition the training set.
    subset_idx selects which chunk to keep. The permutation is deterministic in
    random_state so "subset k" refers to the same conceptual groups across all
    outer folds and across all K training runs.

    Test set is untouched — only training groups are filtered.
    """

    def __init__(self, n_splits, n_subsets, subset_idx, random_state=0):
        assert 0 <= subset_idx < n_subsets
        self.n_splits = n_splits
        self.n_subsets = n_subsets
        self.subset_idx = subset_idx
        self.random_state = random_state
        self._base = GroupKFold(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        groups = np.asarray(groups)
        rng = np.random.default_rng(self.random_state)
        unique_groups = np.sort(np.unique(groups))
        rng.shuffle(unique_groups)
        chunks = np.array_split(unique_groups, self.n_subsets)
        kept = set(chunks[self.subset_idx].tolist())
        for train_idx, test_idx in self._base.split(X, y, groups):
            mask = np.array([g in kept for g in groups[train_idx]])
            yield train_idx[mask], test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
