print("importing")
import os
import shutil
from random import choices, choice, random
import re
import numpy as np
import pandas as pd
from deeprc.utils import Timer
from pathos.multiprocessing import ProcessingPool as Pool

print("imported")
AA = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
n_threads = 60
n_reps = int(600)
prevalence = 0.5
SEQ_LENGTH_MEAN = 14.5
SEQ_LENGTH_STDDEV = 1.1
SEQ_PER_REP_MEAN = 1e5
SEQ_PER_REP_STDDEV = 0
SIZE_PER_POOL = int(5e7)
NUM_IN_OBSERVED = None
PROP_OBSERVED = 0.5  # the proportion of AAs that will be used to replace Z in observed motifs
swap_fraction = [0.2, 0.5, 0.8]
source_data_path = "/storage/ghadia/DeepRC2/deeprc/datasets/SSM2"
wr_list = [0.15]
P_FP_GIVEN_NEGATIVE = None  # 0.0015  # WR/(1-WR)
pos = [0.6, 0.8, 1]
# n_threads = 4
# n_reps = int(20)
# prevalence = 0.5
# SEQ_LENGTH_MEAN = 14.5
# SEQ_LENGTH_STDDEV = 1.1
# SEQ_PER_REP_MEAN = int(1e4)
# SEQ_PER_REP_STDDEV = 0
# SIZE_PER_POOL = int(1e5)
# NUM_IN_OBSERVED = None
# PROP_OBSERVED = 0.5  # the proportion of AAs that will be used to replace Z in observed motifs
# # TODO: Should we restrict the number of AAs that can be used for the observed and non observed motifs instances?
# swap_fraction = 0.5
# source_data_path = "/home/ghadi/PycharmProjects/DeepRC2/deeprc/datasets/SSM2"
# P_FP_GIVEN_NEGATIVE = None  # 0.01
# wr_list = [0.4]
# pos = [1]

replace_probs = {2: 1}
delete_probs = None  # {1: 0.5}
n_motifs = 10


print("Finished defining variables")


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

    def _replace_aa(self, motif, observed: bool):
        idxs = [idx for idx, prob in self.replace_probs.items() if random() < prob]
        for ind in sorted(idxs, reverse=True):
            _AA = AA if PROP_OBSERVED is None else self._split_aas()[0] if observed else self._split_aas()[1]
            motif = motif[:ind] + choice(_AA) + motif[ind + 1:]
        return motif

    def _split_aas(self):
        # todo: Do we need to shuffle?
        obs = AA[:int(PROP_OBSERVED * len(AA))]
        unobs = AA[int(PROP_OBSERVED * len(AA)):]
        return obs, unobs

    def _delete_aas(self, motif):
        idxs = [idx for idx, prob in self.delete_probs.items() if random() < prob]
        for ind in sorted(idxs, reverse=True):
            motif = motif[:ind] + motif[ind + 1:]
        return motif

    def _prepare_motif(self, observed: bool):
        """
        This function takes in an observed parameter, which determines whether the motif chosen will be from the
        observed or unobserved set. If NUM_IN_OBSERVED is not set, a random motif from the base_motifs is chosen.
        Otherwise, if observed is True, the motif chosen will be from the first NUM_IN_OBSERVED elements of base_motifs.
         If observed is False, the motif chosen will be from the remaining elements of base_motifs.

        The function also applies any replacement or deletion with probabilities specified in the replace_probs and
        delete_probs attributes of the class.
        Args:
            observed (bool): A flag indicating whether the chosen motif should be from the observed or unobserved set.

        Returns:
            motif (str): The prepared motif.
        """
        if NUM_IN_OBSERVED is None:
            motif = choice(self.base_motifs)
        else:
            if observed:
                motif = choice(self.base_motifs[:NUM_IN_OBSERVED])
            else:
                motif = choice(self.base_motifs[NUM_IN_OBSERVED:])
        if self.replace_probs:
            motif = self._replace_aa(motif=motif, observed=observed)
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

    def get_pos_seq(self, sequence: str, observed: bool):
        motif = self._prepare_motif(observed)
        return self._implant_motif(motif, sequence)

    def get_neg_seq(self, length):
        seq = "".join(choices(AA, k=length))
        valid = self.sequence_checker(seq)
        return seq if valid else self.get_neg_seq(length)


def get_disease_specific_pool(n_sequences: int):
    base_pool1 = get_base_pool(n_sequences=int(n_sequences * 0.5))
    base_pool2 = get_base_pool(n_sequences=int(n_sequences * 0.5))
    observed = [sequence_generator.get_pos_seq(sequence, observed=True) for sequence in base_pool1]
    unobserved = [sequence_generator.get_pos_seq(sequence, observed=False) for sequence in base_pool2]
    return observed, unobserved


def get_positive_bag(observed_disease_pool: list, unobserved_disease_pool: list, base_pool: list,
                     po: float, puo: float, seed: int = None, n_sequences: int = None):
    if seed is not None:
        np.random.seed(seed)
    seq_pools = np.random.choice(np.asarray([0, 1, 2]), p=[po, puo, 1 - (po + puo)], size=n_sequences).tolist()

    rep = [choice([observed_disease_pool, unobserved_disease_pool, base_pool][pool]) for pool in seq_pools]
    if P_FP_GIVEN_NEGATIVE is not None:
        seq_pools = [3 if (x == 2 and np.random.rand() < P_FP_GIVEN_NEGATIVE) else x for x in seq_pools]
    label = np.isin(seq_pools, [0, 3]) * 1  # np.where(np.array(seq_pools) == 0, 1, 0)
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


def create_dataset(wr: float, po: float = 0.8, hypo_reps: bool = False, po2: float = None, swap_fraction: float = 0.5):
    """
    :param wr: proportion of positive sequences (not percentage)
    :param po: proportion of observed sequences (not percentage)
    """
    if po2 is None:
        po2 = round(1 - po, 2)
    with Timer("first"):

        output_path = source_data_path + f"/n_{n_reps}_wr_{wr:.3%}_po_{po:.0%}_nmotif_{n_motifs}"
        if swap_fraction != 1:
            output_path = output_path + f"_sw_{swap_fraction:.0%}"
        if P_FP_GIVEN_NEGATIVE is not None:
            output_path = output_path + f"_fpgn_{P_FP_GIVEN_NEGATIVE:.3%}"
        if NUM_IN_OBSERVED is not None:
            output_path = output_path + f"_nmio_{NUM_IN_OBSERVED}"
        if po2 is not None:
            output_path = output_path + f"_po2_{po2:.0%}"
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
                    pool.map(get_bag, [po2 * wr] * n_reps_second_set, [(1 - po2) * wr] * n_reps_second_set,
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


def create_poolswmt(size_per_pool: int = SIZE_PER_POOL):
    base_pool = get_base_pool(size_per_pool)
    observed, unobserved = get_disease_specific_pool(size_per_pool)
    return base_pool, observed, unobserved


def get_base_pool(n_sequences: int):
    with Pool(processes=n_threads) as pool:
        result = pool.map(sequence_generator.get_neg_seq,
                          [int(np.random.normal(SEQ_LENGTH_MEAN, SEQ_LENGTH_STDDEV)) for _ in range(n_sequences)])
    return result


if __name__ == "__main__":
    create_from_saved = False
    sequence_generator = SequenceGenerator(replace_probs=replace_probs, delete_probs=delete_probs, n_motifs=n_motifs)
    print("Creating the pools")
    if not create_from_saved:
        with Timer(name="create pools"):
            base_pool, observed, unobserved = create_poolswmt()
    else:
        base_path = "/home/ghadi/PycharmProjects/DeepRC2/base_pool_mt.tsv"
        obse_path = "/home/ghadi/PycharmProjects/DeepRC2/observed_mt.tsv"
        unob_path = "/home/ghadi/PycharmProjects/DeepRC2/unobserved_mt.tsv"

        base_pool = pd.read_csv(base_path)
        observed = pd.read_csv(obse_path)
        unobserved = pd.read_csv(unob_path)

    # If you want sw*n_reps to have **only** obs and the other (1-sw)*n_reps to have **only** unobs, use po=1 and po2=0,
    # with swap_fraction determining the proportion between the two
    for wr in wr_list:
        print(f"witness rate: {wr}")
        for sw in swap_fraction:
            seeds = np.random.randint(0, int(1e6), size=int(n_reps * sw))
            for po in pos:
                hypo_reps = False  # bool(po == 1)
                create_dataset(wr / 100, po=po, hypo_reps=hypo_reps, swap_fraction=sw)
