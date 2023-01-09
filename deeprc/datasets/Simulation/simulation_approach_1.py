import os
import shutil
from random import choices, choice, random
import re
import numpy as np
import pandas as pd
from deeprc.utils import Timer
from pathos.multiprocessing import ProcessingPool as Pool

AA = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
# n_threads = 30
# n_reps = int(600)
# prevalence = 0.5
# SEQ_LENGTH_MEAN = 14.5
# SEQ_LENGTH_STDDEV = 1.1
# SEQ_PER_REP_MEAN = 1e5
# SEQ_PER_REP_STDDEV = 0
# SIZE_PER_POOL = int(5e7)
# swap_fraction = 1
# source_data_path = "/storage/ghadia/DeepRC2/deeprc/datasets/benchmarking1"
n_threads = 4
n_reps = int(5)
prevalence = 0.5
SEQ_LENGTH_MEAN = 14.5
SEQ_LENGTH_STDDEV = 1.1
SEQ_PER_REP_MEAN = int(1e3)
SEQ_PER_REP_STDDEV = 0
SIZE_PER_POOL = int(1e5)
swap_fraction = 1
source_data_path = "/home/ghadi/PycharmProjects/DeepRC2/deeprc/datasets/benchmarking1"
wr_list = [1]
pos = [0.2]

replace_probs = {1: 0.5}
delete_probs = {1: 0.5}
n_motifs = 1

seeds = np.random.randint(0, int(1e6), size=int(n_reps * swap_fraction))

print("Finished defining variables")


# return seq

class SequenceChecker():
    def __init__(self, motifs: list, delete_idxs: list, replace_idxs: list):
        processed_motifs = self._process_motifs(motifs, delete_idxs, replace_idxs)
        print(processed_motifs)
        self.patterns = self._get_regex_patterns(processed_motifs)

    def _get_regex_patterns(self, motifs):
        regex_patterns = []
        for motif in motifs:
            pattern = re.sub(r"([A-Z])d", r"(?:\1)?", motif)
            pattern = re.sub(r"Z", r"[A-Z]", pattern)
            regex_patterns.append(pattern)
        return regex_patterns

    def _check_validity(self, sequence: str):
        valid = True
        for pattern in self.patterns:
            if re.search(pattern, sequence):
                valid = False
                break
        return valid

    def _process_motifs(self, base_motifs, delete_idxs, replace_idxs):
        processed_motifs = []
        for motif in base_motifs:
            chars = []
            for idx in range(len(motif)):
                cur_char = motif[idx] if idx not in replace_idxs else "Z"
                if idx in delete_idxs:
                    chars.append(cur_char + "d")
                else:
                    chars.append(cur_char)
            processed_motifs.append("".join(chars))
        return processed_motifs

    def __call__(self, *args, **kwargs):
        return self._check_validity(*args, **kwargs)


class SequenceGenerator():
    def __init__(self, n_motifs: int = 1, k: int = 4, replace_probs=None, delete_probs=None, base_motifs: list = None):
        self.k = k
        # only one key-value pair for deletion
        if delete_probs is None:
            delete_probs = {}
        if replace_probs is None:
            replace_probs = {}

        if base_motifs is None:
            self.base_motifs = ["GLIM", "FHRS", "ASCG", "ERGK", "KYEA", "VGNH", "PDSI", "HMQR", "RWYA", "RIEI", "GFEQ",
                                "YAVT", "ITFL", "DPKT", "PDCY"][:n_motifs]
        else:
            self.base_motifs = ["".join(choices(AA, k=k)) for _ in range(n_motifs)]

        for ind in replace_probs.keys():
            assert 0 < ind < k - 1, "Invalid replace index"
        for ind in delete_probs.keys():
            assert 0 < ind < k - 1, "Invalid delete index"

        self.replace_probs = replace_probs
        self.delete_probs = delete_probs

        self.sequence_checker = SequenceChecker(motifs=self.base_motifs, delete_idxs=list(delete_probs.keys()),
                                                replace_idxs=list(replace_probs.keys()))

    def _replace_aa(self, motif):
        idxs = [idx for idx, prob in self.replace_probs.items() if random() < prob]
        for ind in sorted(idxs, reverse=True):
            motif = motif[:ind] + choice(AA) + motif[ind + 1:]
        return motif

    def _delete_aas(self, motif):
        idxs = [idx for idx, prob in self.delete_probs.items() if random() < prob]
        for ind in sorted(idxs, reverse=True):
            motif = motif[:ind] + motif[ind + 1:]
        return motif

    def _prepare_motif(self):
        motif = choice(self.base_motifs)
        if self.replace_probs:
            motif = self._replace_aa(motif=motif)
        if self.delete_probs:
            motif = self._delete_aas(motif=motif)
        return motif

    def _implant_motif(self, motif: str, sequence: str):
        """Implant a given motif into a given sequence.

        Args:
            motif (str): The motif to be implanted.
            sequence (str): The sequence where the motif will be implanted.

        Returns:
            str: The sequence with the motif implanted.
        """
        implant_idx = choice(list(range(len(sequence) - len(motif))))
        sequence = sequence[:implant_idx] + motif + sequence[implant_idx + len(motif):]
        return "".join(sequence)

    def get_pos_seq(self, sequence: str):
        motif = self._prepare_motif()
        return self._implant_motif(motif, sequence)

    def get_neg_seq(self, length):
        seq = "".join(choices(AA, k=length))
        valid = self.sequence_checker(seq)
        return seq if valid else self.get_neg_seq(length)


def split_disease_pool(disease_pool):
    split_proportion = 0.5
    observed = disease_pool[:int(len(disease_pool) * split_proportion)]
    unobserved = disease_pool[int(len(disease_pool) * split_proportion):]
    return observed, unobserved


def get_base_pool(n_sequences: int):
    return [sequence_generator.get_neg_seq(int(np.random.normal(SEQ_LENGTH_MEAN, SEQ_LENGTH_STDDEV))) for _ in
            range(n_sequences)]


def get_disease_specific_pool(n_sequences: int):
    base_pool = get_base_pool(n_sequences=n_sequences)
    return [sequence_generator.get_pos_seq(sequence) for sequence in base_pool]


def get_positive_bag(split_disease_pool: list, unsplit_disease_pool: list, base_pool: list,
                     po: float, puo: float, seed: int = None, n_sequences: int = None):
    if seed is not None:
        np.random.seed(seed)
    seq_pools = np.random.choice(np.asarray([0, 1, 2]), p=[po, puo, 1 - (po + puo)], size=n_sequences).tolist()

    rep = [choice([split_disease_pool, unsplit_disease_pool, base_pool][pool]) for pool in seq_pools]  # change to numpy
    label = np.where(np.array(seq_pools) == 0, 1, 0)  # [(pool < 2) * 1 for pool in seq_pools]
    ret_dict = {"amino_acid": rep, "templates": [1] * len(rep), "pool_label": seq_pools, "label": label}
    emp = not bool(ret_dict["templates"])
    return emp, ret_dict


def get_negative_bag(base_pool: list, n_sequences: int):
    rep = np.random.choice(base_pool, size=n_sequences).tolist()
    ret_dict = {"amino_acid": rep, "templates": [1] * len(rep), "pool_label": [2] * len(rep), "label": [0] * len(rep)}
    emp = not bool(ret_dict["templates"])
    return emp, ret_dict


def get_bag(po, puo, status, seed=None):
    emp = True
    n_sequences = int(np.random.normal(SEQ_PER_REP_MEAN, SEQ_PER_REP_STDDEV))
    while emp:
        if status:
            emp, return_dict = get_positive_bag(observed, unobserved, base_pool, po, puo, seed, n_sequences)
        else:
            emp, return_dict = get_negative_bag(base_pool, n_sequences)
    return return_dict


def create_pools(size_per_pool: int = SIZE_PER_POOL):
    base_pool = get_base_pool(size_per_pool)
    disease_pool = get_disease_specific_pool(size_per_pool)
    observed, unobserved = split_disease_pool(disease_pool)
    return base_pool, observed, unobserved


def find_indices(lst, char, n):
    indices = []
    for i, x in enumerate(lst):
        if x == char and n > 0:
            indices.append(i)
            n -= 1
        elif n == 0:
            break
    return indices


def create_directory(output_path):
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path + "/repertoires")
    print(f"{output_path} directory created")


def unpack_dicts(*dicts):
    combined = {}
    for dictionary in dicts:
        for key, value in dictionary.items():
            combined[key] = combined.get(key, []) + value
    return combined


def create_dataset(wr: float, po: float = 0.8, swap_fraction: float = 0.5, hypo_reps: bool = False):
    """
    :param wr: proportion of positive sequences (not percentage)
    :param po: proportion of observed sequences (not percentage)
    """
    with Timer("first"):

        output_path = source_data_path + f"/n_{n_reps}_wr_{wr:.3%}_po_{po:.0%}"
        if swap_fraction != 1:
            output_path = output_path + f"_sw_{swap_fraction:.0%}"
        # output_path = source_data_path + f"/delete"
        print(f"\033[44;97m {output_path} \033[0m")
        # output_path = source_data_path + f"/n_{n_reps}_wr_{wr}"
        create_directory(output_path)

        statuses = np.random.binomial(n=1, p=prevalence, size=n_reps)
        meta_dict = {"ID": [f"rep_{i:05}.tsv" for i in range(n_reps)],
                     "binary_target_1": ["+" if status else "-" for status in statuses]}

        pd.DataFrame(meta_dict).to_csv(output_path + "/metadata.tsv", sep="\t", index=False)
    basic_list = []
    n_reps_first_set = int(n_reps * swap_fraction)
    n_reps_second_set = int(n_reps * (1 - swap_fraction))

    with Timer("first set"):
        with Pool(n_threads) as pool:
            for idx, result in enumerate(
                    pool.map(get_bag, [po * wr] * n_reps_first_set, [(1 - po) * wr] * n_reps_first_set,
                             statuses[:n_reps_first_set], seeds)):
                basic_list.append(result)

    with Timer("second set"):
        with Pool(n_threads) as pool:
            for idx, result in enumerate(
                    pool.map(get_bag, [(1 - po) * wr] * n_reps_second_set, [po * wr] * n_reps_second_set,
                             statuses[n_reps_first_set:])):
                basic_list.append(result)

    if hypo_reps:
        first_n_negatives = find_indices(statuses.tolist(), 0, 2)
        dicts = [basic_list[i] for i in first_n_negatives]
        hypo_neg = unpack_dicts(*dicts)
        _, hypo_pos = get_positive_bag(observed, unobserved, base_pool, po=1, puo=0, n_sequences=1000)
        sep_meta = {"ID": ["hypo_neg.tsv", "hypo_pos.tsv"], "binary_target_1": ["-", "+"]}
        create_directory(output_path + "/hypo")
        pd.DataFrame(hypo_neg).to_csv(output_path + "/hypo/repertoires/" + sep_meta["ID"][0], sep="\t", index=False)
        pd.DataFrame(hypo_pos).to_csv(output_path + "/hypo/repertoires/" + sep_meta["ID"][1], sep="\t", index=False)
        pd.DataFrame(sep_meta).to_csv(output_path + "/hypo/" + "metadata.tsv", sep="\t", index=False)

    # pd.Series(repeat_list).to_csv(output_path + "/repeat")
    with Timer("Saving repertoires"):
        print("Saving repertoires")
        # with Pool() as pool:
        #     pool.map(save_repertoire, meta_dict["ID"], basic_list)
        for ind, rep in enumerate(basic_list):
            pd.DataFrame(rep).to_csv(output_path + "/repertoires/" + meta_dict["ID"][ind], sep="\t", index=False)


if __name__ == "__main__":

    sequence_generator = SequenceGenerator(replace_probs=replace_probs, delete_probs=delete_probs, n_motifs=n_motifs)
    print("Creating the pools")
    with Timer(name="create pools"):
        base_pool, observed, unobserved = create_pools()
    for wr in wr_list:
        print(f"witness rate: {wr}")
        for po in pos:
            hypo_reps = bool(po == 1)
            create_dataset(wr / 100, po=po, swap_fraction=swap_fraction, hypo_reps=hypo_reps)
