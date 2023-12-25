import os
import random
import numpy as np
import pandas as pd
import pickle
import re
from multiprocessing import Pool

# Define the mapping between numbers and strings
# mapping = {0: "TN", 1: "FP", 2: "FN", 3: "TP"}
mapping = {0: "LW̅", 1: "HW̅", 2: "LW", 3: "HW"}
transposed_mapping = {v: k for k, v in mapping.items()}


class AnnotateReps:
    def __init__(self, root_directory, n_signal=40, suffix="simulated_repertoires", n_seq=25e3, n_threads=10):
        self.root_directory = root_directory
        self.data_directory = root_directory + "/" + suffix
        self.metadata = pd.read_csv(root_directory + "/metadata.csv")
        self.n_signal = n_signal
        self.suffix = suffix
        self.n_threads = n_threads
        self.wr = n_signal / n_seq
        self.TPR = np.array([5, 10, 20, 50, 100])
        self.FDR = np.array([0, 10, 50, 80])
        self.columns_to_keep = ["cdr3_aa", "v_call", "j_call", "is_signal"]
        self.negative_sequences_file = root_directory + "/negative_sequences.pkl"
        self.positive_sequences_file = root_directory + "/signal_components/filtered_implantable_signal_pool.tsv"
        negative_sequences = self.read_negative_sequences() if os.path.exists(
            self.negative_sequences_file) else self.save_and_return_negative_sequences()
        positive_sequences = self.get_implanted_signals()
        self.tpr_fdr_dict = self.get_tpr_fpr_dict(negative_sequences, positive_sequences)

    def get_tpr_fpr_dict(self, negative_sequences, positive_sequences):
        random.seed(0)
        tpr_fdr_dict = {}
        for tpr in self.TPR:
            tpr = tpr / 100
            observed_signals = random.sample(positive_sequences, int(len(positive_sequences) * tpr))
            tpr_fdr_dict[f"tpr_{tpr}"] = observed_signals
            for fdr in self.FDR:
                fdr = fdr / 100
                fpr = self.compute_FPR(tpr, fdr, self.wr)
                false_signals = random.sample(negative_sequences, int(len(negative_sequences) * fpr))
                tpr_fdr_dict[f"tpr_{tpr}_fdr_{fdr}"] = false_signals
        return tpr_fdr_dict

    def annotate_rep(self, tsv_file):
        print("File #", self.metadata.loc[self.metadata['filename'] == tsv_file].index[0])
        df = self.annotate(tsv_file)
        df.to_csv(f"{self.data_directory}/{tsv_file}", sep='\t', index=False)

    def get_implanted_signals(self):
        # read the csv file where the first column is not the index column
        df = pd.read_csv(self.positive_sequences_file, sep='\t', header=None)
        sequences = df[1].tolist()
        # remove the first and last character from each sequence
        assert all([type(x) == str for x in sequences])
        sequences = [x[1:-1] for x in sequences]
        return sequences

    def save_and_return_negative_sequences(self):
        # read all repertoires in the directory, and get the negative sequences that have is_signal = 0, and add them
        # to a set of all the negative sequences
        negative_sequences = set()
        for id, file in enumerate(os.listdir(self.root_directory + "/simulated_repertoires")):
            if id % 25 == 0:
                print(id)
            if file.endswith(".tsv"):
                df = pd.read_csv(self.root_directory + "/simulated_repertoires/" + file, sep='\t')
                negative_sequences.update(df[df['is_signal'] == 0]['cdr3_aa'].tolist())
                # if float("nan") in negative_sequences:
                #     print(file)
        negative_sequences = [x for x in negative_sequences if not (x != x)]
        # write the negative sequences to a pickle file
        with open(self.negative_sequences_file, "wb") as f:
            pickle.dump(negative_sequences, f)
        return negative_sequences

    def read_negative_sequences(self):
        # read the negative sequences from the pickle file
        with open(self.negative_sequences_file, "rb") as f:
            negative_sequences = pickle.load(f)
        return negative_sequences

    def annotate(self, tsv_file):
        class_label = self.metadata.loc[self.metadata["filename"] == tsv_file, "label_positive"].tolist()[0]
        df = pd.read_csv(f"{self.data_directory}/{tsv_file}", sep='\t')
        df = df[self.columns_to_keep]
        if not class_label:
            assert not np.count_nonzero(df['is_signal'] == 1)
        for tpr in self.TPR:
            tpr = tpr / 100
            # observed_signals = self.tpr_fpr_dict[f"tpr_{tpr}"]
            for fdr in self.FDR:
                label = 'is_signal_TPR_' + str(int(tpr * 100)) + '%_FDR_' + str(fdr) + '%'
                df[label] = df['is_signal']
                fdr = fdr / 100
                fpr = self.compute_FPR(tpr, fdr, self.wr)
                # false_signals = self.tpr_fpr_dict[f"fpr_{fpr}"]
                indices_0_to_1 = (df['is_signal'] == 0) & df['cdr3_aa'].isin(self.tpr_fdr_dict[f"tpr_{tpr}_fdr_{fdr}"])
                df.loc[indices_0_to_1, label] = 1
                indices_1_to_0 = (df['is_signal'] == 1) & ~df['cdr3_aa'].isin(self.tpr_fdr_dict[f"tpr_{tpr}"])
                df.loc[indices_1_to_0, label] = 0
                # if class_label:
                #     tpr_calc = np.count_nonzero((df[label] == 1) & (df["is_signal"] == 1)) / np.count_nonzero(
                #         df['is_signal'] == 1)
                # if np.count_nonzero(df[label] == 1):
                #     fdr_calc = np.count_nonzero((df[label] == 1) & (df["is_signal"] == 0)) / np.count_nonzero(
                #         df[label] == 1)
                # fpr_calc = np.count_nonzero((df[label] == 1) & (df["is_signal"] == 0)) / np.count_nonzero(
                #     df['is_signal'] == 0)
                # if class_label:
                #     print(f"TPR: True: {tpr}, Calc: {tpr_calc}")
                #     if np.count_nonzero(df[label] == 1):
                #         print(f"FDR: True: {fdr}, Calc: {fdr_calc}")
                # print(f"FPR: True: {fpr}, Calc: {fpr_calc}")
                df[label + '_pool'] = df['is_signal'] * 2 + df[label]

        return df

    def compute_FPR(self, tpr, fdr, witness_rate):
        fpr = (tpr * witness_rate * fdr) / ((1 - fdr) * (1 - witness_rate))
        return fpr

    def annotate_reps_all_files(self):
        for id, file in enumerate(os.listdir(self.data_directory)):
            if id % 50 == 0:
                print(id)
            if file.endswith(".tsv"):
                self.annotate_rep(file)

    # make a parallel version of annotate_reps_all_files using multiprocessing
    def annotate_reps_all_files_parallel(self):
        files = [file for file in os.listdir(self.data_directory) if file.endswith(".tsv")]

        # Create a pool with the number of available CPUs
        with Pool(self.n_threads) as pool:
            # Use starmap to pass multiple arguments to the wrapper function
            pool.map(self.annotate_rep, files)

    def test_metrics_per_label_per_rep(self, HW, HW̅, LW, LW̅):
        wr = (HW + LW) / (HW + HW̅ + LW + LW̅)
        fdr = HW̅ / (HW̅ + HW)
        fpr1 = HW̅ / (HW̅ + LW̅)
        tpr = HW / (HW + LW)
        fpr2 = self.compute_FPR([tpr], [fdr], wr)
        assert np.abs(fpr1 - fpr2[0]) < 1e-5

    def calculate_wr_tpr_fdr(self, HW, HW̅, LW, LW̅):
        wr = (HW + LW) / (HW + HW̅ + LW + LW̅)
        try:
            fdr = HW̅ / (HW̅ + HW)
        except ZeroDivisionError:
            fdr = 0
        try:
            fpr = HW̅ / (HW̅ + LW̅)
        except ZeroDivisionError:
            fpr = 0
        try:
            tpr = HW / (HW + LW)
        except ZeroDivisionError:
            tpr = 0
        return tpr, fdr, fpr, wr

    def get_metrics_per_label_per_rep(self, df, label):
        HW = np.count_nonzero(df[label + '_pool'] == transposed_mapping["HW"])
        HW̅ = np.count_nonzero(df[label + '_pool'] == transposed_mapping["HW̅"])
        LW = np.count_nonzero(df[label + '_pool'] == transposed_mapping["LW"])
        LW̅ = np.count_nonzero(df[label + '_pool'] == transposed_mapping["LW̅"])
        return HW, HW̅, LW, LW̅

    def test_metrics_across_labels(self):
        # compile results in dict of labels of dicts of metrics
        results = {}
        counter = 0
        for id, file in enumerate(os.listdir(self.data_directory)):
            if id % 50 == 0:
                print(id)
            if file.endswith(".tsv"):
                df = pd.read_csv(f"{self.data_directory}/{file}", sep='\t')
                class_label = self.metadata[self.metadata['filename'] == file]['label_positive'].values[0]
                for tpr in self.TPR:
                    tpr = tpr / 100
                    for fdr in self.FDR:
                        label = 'is_signal_TPR_' + str(int(tpr * 100)) + '%_FDR_' + str(fdr) + '%'
                        HW, HW̅, LW, LW̅ = self.get_metrics_per_label_per_rep(df, label)
                        if label not in results:
                            results[label] = {"TPR": [], "FDR": [], "FPR": [], "WR": []}
                        tpr_e, fdr_e, fpr_e, wr_e = self.calculate_wr_tpr_fdr(HW, HW̅, LW, LW̅)
                        if class_label and wr_e == 0:
                            counter += 1
                        if not class_label and wr_e != 0:
                            print(file)
                            print("negative file found with positives")
                            break
                        if class_label:
                            results[label]["TPR"].append(tpr_e)
                            results[label]["FDR"].append(fdr_e)
                            results[label]["WR"].append(wr_e)
                        results[label]["FPR"].append(fpr_e)
                        # self.test_metrics_per_label_per_rep(TP, FP, FN, TN)
        print(f"Empty positive files @ {self.wr}: {counter}")
        pattern = r"TPR_(\d+)%_FDR_(\d+)%"
        # compute mean and std across reps
        for label in results:
            print(label)
            match = re.search(pattern, label)
            for metric in results[label]:
                results[label][metric] = np.mean(results[label][metric]), np.std(results[label][metric])
                if metric == "FPR":
                    continue
                print(metric, results[label][metric])
                # # if not (self.n_signal in [5, 13] and tpr in ["5", "10"]):
                # if self.n_signal not in [5, 13]:
                #     if metric == "TPR":
                #         assert np.abs((results[label][metric][0] * 100 - int(tpr)) / int(tpr) * 100) < 10
                #     if metric == "FDR":
                #         if int(fdr) == 0:
                #             assert results[label][metric][0] == 0
                #         else:
                #             assert np.abs((results[label][metric][0] * 100 - int(fdr)) / int(fdr) * 100) < 10
                # if metric == "WR":
                #     assert np.abs((results[label][metric][0] - self.wr) / self.wr * 100) < 10


if __name__ == '__main__':
    # input_file = "/Users/annashcherbina/Projects/deeprc/deeprc/datasets/encode_roadmap/encode_roadmap_1000_0.1.tsv"
    # test_file = "/storage/ghadia/DeepRC2/deeprc/datasets/HIV/test.tsv"
    # annotate_reps(test_file)
    # set a seed for all random operations
    random.seed(0)
    np.random.seed(0)
    for n_signal in [5, 13, 25, 50, 250, 500]:
        # for n_signal in [500]:
        annotater = AnnotateReps(f"/storage/ghadia/DeepRC2/deeprc/datasets/HIV/final/phenotype_burden_{n_signal}/data",
                                 n_signal=n_signal, n_threads=30)
        # annotater.annotate_reps_all_files()
        annotater.annotate_reps_all_files_parallel()
        annotater.test_metrics_across_labels()
