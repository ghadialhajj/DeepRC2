import numpy as np
import pandas as pd
import dill as pkl
from itertools import chain

metadata = pd.read_csv(f"/storage/ghadia/DeepRC2/deeprc/datasets/emerson_linz/AIRR/metadata.csv",
                       dtype=str, keep_default_na=False)
n_cv_folds = 4
n_test_files = 120

original_split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_splits.pkl"
with open(original_split_file, 'rb') as sfh:
    split_inds = pkl.load(sfh)
flattened_list = sorted(list(chain.from_iterable(split_inds)))[
                 :-n_test_files]  # indices of files **to use** (without invalid files) excluding the test files.

filtered_metadata = metadata.iloc[flattened_list]  # metadata file without the test files.

rnd_gen = np.random.RandomState(0)

cv_splits_indices = [[] for _ in range(n_cv_folds)]

# Positive class
indices = np.where((filtered_metadata['CMV'].values == '+'))[0]
rnd_gen.shuffle(indices)

n_sample_per_split = int(len(indices) / n_cv_folds)
for f in range(n_cv_folds):
    cv_splits_indices[f] += [indices[f * n_sample_per_split:(f + 1) * n_sample_per_split]]
cv_splits_indices[n_cv_folds - 1] += [
    indices[n_cv_folds * n_sample_per_split:]]  # assign left-over samples to last split

# Negative class
indices = np.where((filtered_metadata['CMV'].values == '-'))[0]
rnd_gen.shuffle(indices)

n_sample_per_split = int(len(indices) / n_cv_folds)
for f in range(n_cv_folds):
    cv_splits_indices[f] += [indices[f * n_sample_per_split:(f + 1) * n_sample_per_split]]
cv_splits_indices[n_cv_folds - 1] += [
    indices[n_cv_folds * n_sample_per_split:]]  # attribute left-over samples to last split

cv_splits_indices = [np.sort(np.concatenate(i)) for i in cv_splits_indices]
cv_splits_indices.append(np.array(list(range(len(metadata) - n_test_files, len(metadata)))))
cv_splits_ids = [metadata['filename'].values[i] for i in cv_splits_indices]

with open(f"/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_separate_test.pkl", 'wb') as fh:
    pkl.dump(dict(inds=cv_splits_indices, sample_keys=cv_splits_ids), file=fh)
