import os
import random
import numpy as np
import pandas as pd
import pickle
import re

# Define the mapping between numbers and strings
mapping = {0: "TN", 1: "FP", 2: "FN", 3: "TP"}
transposed_mapping = {v: k for k, v in mapping.items()}


class AnnotateReps:
    def __init__(self, root_directory, n_signal=40, suffix="simulated_repertoires", n_seq=25e3):
        self.root_directory = root_directory
        self.n_signal = n_signal
        self.suffix = suffix
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

    def annotate_rep(self, tsv_file, wr, class_label):
        df = self.annotate(tsv_file, wr, class_label)
        df.to_csv(tsv_file, sep='\t', index=False)

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

    def annotate(self, tsv_file, wr, class_label):
        df = pd.read_csv(tsv_file, sep='\t')
        columns_to_keep = ["cdr3_aa", "v_call", "j_call", "is_signal"]
        df = df[columns_to_keep]
        if class_label:
            assert np.count_nonzero(df['is_signal'] == 1)
        else:
            assert not np.count_nonzero(df['is_signal'] == 1)
        for tpr in self.TPR:
            tpr = tpr / 100
            # observed_signals = self.tpr_fpr_dict[f"tpr_{tpr}"]
            for fdr in self.FDR:
                label = 'is_signal_TPR_' + str(int(tpr * 100)) + '%_FDR_' + str(fdr) + '%'
                df[label] = df['is_signal']
                fdr = fdr / 100
                fpr = self.compute_FPR(tpr, fdr, wr)
                # false_signals = self.tpr_fpr_dict[f"fpr_{fpr}"]
                indices_0_to_1 = (df['is_signal'] == 0) & df['cdr3_aa'].isin(self.tpr_fdr_dict[f"tpr_{tpr}_fdr_{fdr}"])
                df.loc[indices_0_to_1, label] = 1
                indices_1_to_0 = (df['is_signal'] == 1) & ~df['cdr3_aa'].isin(self.tpr_fdr_dict[f"tpr_{tpr}"])
                df.loc[indices_1_to_0, label] = 0
                if class_label:
                    tpr_calc = np.count_nonzero((df[label] == 1) & (df["is_signal"] == 1)) / np.count_nonzero(
                        df['is_signal'] == 1)
                if np.count_nonzero(df[label] == 1):
                    fdr_calc = np.count_nonzero((df[label] == 1) & (df["is_signal"] == 0)) / np.count_nonzero(
                        df[label] == 1)
                fpr_calc = np.count_nonzero((df[label] == 1) & (df["is_signal"] == 0)) / np.count_nonzero(
                    df['is_signal'] == 0)
                if class_label:
                    print(f"TPR: True: {tpr}, Calc: {tpr_calc}")
                    if np.count_nonzero(df[label] == 1):
                        print(f"FDR: True: {fdr}, Calc: {fdr_calc}")
                print(f"FPR: True: {fpr}, Calc: {fpr_calc}")
                # df.loc[(df['is_signal'] == 1) & (rand_nums < 1 - tpr), label] = 0
                # df.loc[(df['is_signal'] == 0) & (rand_nums < fpr), label] = 1
                df[label + '_pool'] = df['is_signal'] * 2 + df[label]
                # df[label + '_pool'] = df[label + '_pool'].replace(mapping)
                # FN = np.count_nonzero(df[label + '_pool'] == "FN")
                # TP = np.count_nonzero(df[label + '_pool'] == "TP")
                # FP = np.count_nonzero(df[label + '_pool'] == "FP")
                # print()

        return df

    def compute_FPR(self, tpr, fdr, witness_rate):
        fpr = (tpr * witness_rate * fdr) / ((1 - fdr) * (1 - witness_rate))
        return fpr

    def annotate_reps_all_files(self):
        directory = self.root_directory + "/" + self.suffix
        metadata = pd.read_csv(self.root_directory + "/metadata.csv")
        for id, file in enumerate(os.listdir(directory)):
            if id % 50 == 0:
                print(id)
            if file.endswith(".tsv"):
                class_label = metadata.loc[metadata["filename"] == file, "label_positive"].tolist()[0]
                self.annotate_rep(f"{directory}/{file}", self.wr, class_label)

    def test_metrics_per_label_per_rep(self, TP, FP, FN, TN):
        wr = (TP + FN) / (TP + FP + FN + TN)
        fdr = FP / (FP + TP)
        fpr1 = FP / (FP + TN)
        tpr = TP / (TP + FN)
        fpr2 = self.compute_FPR([tpr], [fdr], wr)
        assert np.abs(fpr1 - fpr2[0]) < 1e-5

    def calculate_wr_tpr_fdr(self, TP, FP, FN, TN):
        wr = (TP + FN) / (TP + FP + FN + TN)
        try:
            fdr = FP / (FP + TP)
        except ZeroDivisionError:
            fdr = 0
        try:
            fpr = FP / (FP + TN)
        except ZeroDivisionError:
            fpr = 0
        try:
            tpr = TP / (TP + FN)
        except ZeroDivisionError:
            tpr = 0
        return tpr, fdr, fpr, wr

    def get_metrics_per_label_per_rep(self, df, label):
        TP = np.count_nonzero(df[label + '_pool'] == transposed_mapping["TP"])
        FP = np.count_nonzero(df[label + '_pool'] == transposed_mapping["FP"])
        FN = np.count_nonzero(df[label + '_pool'] == transposed_mapping["FN"])
        TN = np.count_nonzero(df[label + '_pool'] == transposed_mapping["TN"])
        return TP, FP, FN, TN

    def test_metrics_across_labels(self):
        directory = self.root_directory + "/" + self.suffix
        metadata = pd.read_csv(self.root_directory + "/metadata.csv")
        # compile results in dict of labels of dicts of metrics
        results = {}
        for id, file in enumerate(os.listdir(directory)):
            if id % 50 == 0:
                print(id)
            if file.endswith(".tsv"):
                df = pd.read_csv(f"{directory}/{file}", sep='\t')
                class_label = metadata[metadata['filename'] == file]['label_positive'].values[0]
                for tpr in self.TPR:
                    tpr = tpr / 100
                    for fdr in self.FDR:
                        label = 'is_signal_TPR_' + str(int(tpr * 100)) + '%_FDR_' + str(fdr) + '%'
                        TP, FP, FN, TN = self.get_metrics_per_label_per_rep(df, label)
                        if label not in results:
                            results[label] = {"TPR": [], "FDR": [], "FPR": [], "WR": []}
                        tpr_e, fdr_e, fpr_e, wr_e = self.calculate_wr_tpr_fdr(TP, FP, FN, TN)
                        if class_label and wr_e == 0:
                            print(file)
                            print("positive file found without true positives")
                            break
                        if not class_label and wr_e != 0:
                            print(file)
                            print("negative file found with positives")
                            break
                        if class_label:
                            results[label]["TPR"].append(tpr_e)
                            results[label]["FDR"].append(fdr_e)
                        results[label]["FPR"].append(fpr_e)
                        if class_label:
                            results[label]["WR"].append(wr_e)
                        # self.test_metrics_per_label_per_rep(TP, FP, FN, TN)
        # compute mean and std across reps
        for label in results:
            print(label)
            for metric in results[label]:
                results[label][metric] = np.mean(results[label][metric]), np.std(results[label][metric])
                print(metric, results[label][metric])


if __name__ == '__main__':
    # input_file = "/Users/annashcherbina/Projects/deeprc/deeprc/datasets/encode_roadmap/encode_roadmap_1000_0.1.tsv"
    # test_file = "/storage/ghadia/DeepRC2/deeprc/datasets/HIV/test.tsv"
    # annotate_reps(test_file)
    # set a seed for all random operations
    random.seed(0)
    np.random.seed(0)

    annotater = AnnotateReps("/storage/ghadia/DeepRC2/deeprc/datasets/HIV/v2/phenotype_burden_40/data")
    # annotater.annotate_reps_all_files()
    annotater.test_metrics_across_labels()
