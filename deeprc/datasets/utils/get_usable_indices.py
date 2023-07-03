import pandas as pd
import numpy as np

parent_dir = "/storage/ghadia/DeepRC2/deeprc/datasets/emerson_linz/AIRR"

metadatafile = pd.read_csv(f"{parent_dir}/metadata.csv")
mask = metadatafile['CMV'] == 'Unknown'
nocmv = metadatafile.loc[mask].index.tolist()  # list of repertoires with unknown CMV status

nodup = []  # list of repertoires without "duplicate_count" column
negdup = []  # list of repertoires with negative values in the "duplicate_count" column
zerodup = []  # list of repertoires with negative values in the "duplicate_count" column

for idx, file in enumerate(metadatafile["filename"]):
    print(idx)
    csvfile = pd.read_csv(f"{parent_dir}/repertoires/{file}", sep="\t")
    if csvfile['duplicate_count'].isna().any():
        nodup.append(idx)
    if np.any(csvfile['duplicate_count'] < 0):
        negdup.append((idx, np.count_nonzero(csvfile['duplicate_count'] < 0)))
    if np.any(csvfile['duplicate_count'] == 0):
        zerodup.append((idx, np.count_nonzero(csvfile['duplicate_count'] == 0)))

to_remove_indices = nocmv + nodup + negdup + [z[0] for z in zerodup]
indices_to_use = [x for x in list(range(len(metadatafile))) if x not in to_remove_indices]
# todo find the corresponding indices in the metadata file
# Result: the resulting indices to use are the same as the ones used in the DeepRC paper without the 565, HIP14092.tsv
# file that has three sequences with zero counts.
# These indices have two files more than the ones used in the immuneML paper, 569 (HIP14106) and 175 (HIP04958). These
# Two extra files have NaNs for most columns but they have positive duplicate counts, but that's fine for the
# DeepRC-related work.

# todo save the file as pkl
# Since the files are the same except for one file, we will use the same pkl file from the original DeepRC paper.
# todo split the folds class-wise
# Done and saved in "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_separate_test.pkl"
# todo do the sweep with three different seeds
