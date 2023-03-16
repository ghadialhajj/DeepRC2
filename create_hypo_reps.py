import numpy as np
import pandas as pd

root_dir = "/home/ghadi/PycharmProjects/DeepRC2/deeprc/datasets/trb_dataset/"


def create_negative():
    rnd_seed = 0
    n_repertoires = 146
    n_splits = 5
    cross_validation_fold: int = 0

    n_repertoires_per_split = int(n_repertoires / n_splits)
    rnd_gen = np.random.RandomState(rnd_seed)
    shuffled_repertoire_inds = rnd_gen.permutation(n_repertoires)
    split_inds = [shuffled_repertoire_inds[s_i * n_repertoires_per_split:(s_i + 1) * n_repertoires_per_split]
                  if s_i != n_splits - 1 else
                  shuffled_repertoire_inds[s_i * n_repertoires_per_split:]  # Remaining repertoires to last split
                  for s_i in range(n_splits)]

    _ = split_inds.pop(cross_validation_fold - 1)
    trainingset_inds = np.concatenate(split_inds)

    meta_df = pd.read_csv(f"{root_dir}/AIRR/metadata2.tsv", sep="\t")

    neg_rep_df_list = [pd.read_csv(f"{root_dir}/AIRR/repertoires/{meta_df['ID'][i]}", sep="\t")[
                           ["sequence_id", "cdr3_aa", "duplicate_count", "matched"]] for i in trainingset_inds]

    all_neg_train_reps = pd.concat(neg_rep_df_list).drop_duplicates(subset=['cdr3_aa'])
    num_rows = len(all_neg_train_reps[["cdr3_aa"]])
    all_neg_train_reps['matched'] = np.zeros(num_rows, dtype=int)
    all_neg_train_reps.to_csv(f"{root_dir}/HypoAIRR/repertoires/negative_compiled.tsv", sep="\t")


def fix_positive():
    additional_set = pd.read_csv(f"{root_dir}/HypoAIRR/repertoires/additional_set.tsv", sep="\t")
    num_rows = len(additional_set[["cdr3_aa"]])
    additional_set['matched'] = np.ones(num_rows, dtype=int)
    additional_set.to_csv(f"{root_dir}/HypoAIRR/repertoires/additional_set.tsv", sep="\t")


if __name__ == '__main__':
    fix_positive()
    create_negative()
