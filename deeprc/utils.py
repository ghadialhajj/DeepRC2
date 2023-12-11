# -*- coding: utf-8 -*-
"""
Utility functions and classes

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shutil

import torch
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from tqdm import tqdm
import wandb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from time import time
import logomaker
import dill as pkl


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time()

    def __exit__(self, type, value, traceback):
        self.end = time()
        print(f"{self.name}: {self.end - self.start}")


def user_confirmation(text: str = "Continue?", continue_if: str = 'y', abort_if: str = 'n'):
    """Wait for user confirmation"""
    while True:
        user_input = input(f"{text} ({continue_if}/{abort_if})")
        if user_input == continue_if:
            break
        elif user_input == abort_if:
            exit("Session terminated by user.")


def url_get(url: str, dst: str, verbose: bool = True):
    """Download file from `url` to file `dst`"""
    stream = requests.get(url, stream=True)
    try:
        stream_size = int(stream.headers['Content-Length'])
    except KeyError:
        raise FileNotFoundError(f"Sorry, the URL {url} could not be reached. "
                                f"Either your connection has a problem or the server is down."
                                f"Please check your connection, try again later, "
                                f"or notify me per email if the problem persists.")
    src = stream.raw
    windows = os.name == 'nt'
    copy_bufsize = 1024 * 1024 if windows else 64 * 1024
    update_progess_bar = tqdm(total=stream_size, disable=not verbose,
                              desc=f"Downloading {stream_size * 1e-9:0.3f}GB dataset")
    with open(dst, 'wb') as out_file:
        while True:
            buf = src.read(copy_bufsize)
            if not buf:
                break
            update_progess_bar.update(copy_bufsize)
            out_file.write(buf)
        shutil.copyfileobj(stream.raw, out_file)
    print()
    del stream


class Logger():
    def __init__(self, dataloaders, with_FPs=False):
        self.with_FPs = with_FPs
        self.dataloaders = dataloaders

    def log_motifs(self, params: np.ndarray, step):
        cnn_weights = params[:, :20, :].squeeze()

        for motif_idx in range(cnn_weights.shape[0]):
            motif_matrix = cnn_weights[motif_idx].T
            fig = plot_motifs(motif_matrix)
            wandb.log({f'Motifs/motif_{str(motif_idx)}': wandb.Image(fig)}, step=step)
            plt.close(fig)

    def log_stats(self, step: int, all_logits, all_targets, all_emb_reps, all_attentions, all_pools,
                  att_hists: bool = False, log_per_kernel: bool = False, logit_hist: bool = False,
                  dl_name: str = "trainingset_eval"):
        """
        Logs model statistics including repertoire embeddings, logits, and attentions for each dataloader.
        """
        print("Logging stats:")
        # if not dl.batch_sampler.sampler.data_source.indices.any():
        #     continue
        split_logits, split_attentions, split_rep_embs = self.get_values_per_dl(all_logits, all_targets, all_emb_reps,
                                                                                all_attentions, all_pools)
        split_rep_embs_pca = perform_pca(split_rep_embs)
        # split_rep_embs_tsne = perform_tsne(split_rep_embs)
        self.log_repertoire_rep(dl_name, split_rep_embs_pca, None, step)
        if att_hists:
            self.log_attention(dl_name, split_attentions, step)
        if logit_hist:
            self.log_logits(dl_name, split_logits, step)
        if log_per_kernel and dl_name == "validationset_eval":
            self.log_per_kernel(split_rep_embs)

    def log_repertoire_rep(self, dl_name, split_rep_embs_pca, split_rep_embs_tsne, step):
        self.plot_scatter(pos_vals=split_rep_embs_pca[0], neg_vals=split_rep_embs_pca[1], dl_name=dl_name,
                          plot_title=f"Positive (B) and Negative (R) PCAcomp", step=step, method="PCA")
        # self.plot_scatter(pos_vals=split_rep_embs_tsne[0], neg_vals=split_rep_embs_tsne[1], dl_name=dl_name,
        #                   plot_title=f"Positive (B) and Negative (R) TSNEcomp", step=step, method="TSNE")

    def log_attention(self, dl_name, split_attentions, step):
        self.plot_histogram(split_attentions, dl_name=dl_name,
                            plot_title=f"Raw Attention", xaxis_title="Attention value",
                            step=step)

    def log_logits(self, dl_name, split_logits, step):
        self.plot_histogram(data={"positive": split_logits[0], "negative": split_logits[1]}, dl_name=dl_name,
                            plot_title=f"Positive (B) and Negative (R) Logits", xaxis_title="Logit value",
                            step=step)

    def get_values_per_dl(self, all_logits, all_targets, all_emb_reps, all_attentions, all_pools):
        split_logits = self.split_outputs(all_logits, all_targets)
        split_attentions = self.split_outputs(all_attentions, all_pools, sequence_level=True)
        split_rep_embs = self.split_outputs(all_emb_reps, all_targets, flatten=False)
        return split_logits, split_attentions, split_rep_embs

    @staticmethod
    def plot_histogram(data: dict, n_bins: int = 50, dl_name: str = "", plot_title: str = "",
                       xaxis_title: str = "Attention value", step: int = 0):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        config = dict(opacity=0.6, histnorm="percent", nbinsx=n_bins)

        for name, values in data.items():
            secondary_y = True if name in ["TN", "FP"] else False
            fig.add_trace(go.Histogram(x=values, **config, name=name), secondary_y=secondary_y)
        # Set y-axes titles
        fig.update_yaxes(title_text="Postives (TP+FN)", secondary_y=False)
        fig.update_yaxes(title_text="Negatives (TN+FP)", secondary_y=True)

        fig.update_layout(title=plot_title, xaxis_title=xaxis_title, yaxis_title="Percentage", barmode='overlay')
        fig.update_traces(autobinx=False, selector=dict(type='histogram'))

        # Log the plot
        wandb.log({f"{xaxis_title}/{dl_name}": fig}, step=step)
        fig.data = []

    @staticmethod
    def plot_scatter(pos_vals: list, neg_vals: list, dl_name: str = "", plot_title: str = "", step: int = 0,
                     method: str = "PCA"):
        fig = make_subplots()

        fig.add_trace(
            go.Scatter(x=pos_vals[:, 0], y=pos_vals[:, 1], mode='markers', marker=go.scatter.Marker(color="blue"),
                       name="positive"))
        fig.add_trace(
            go.Scatter(x=neg_vals[:, 0], y=neg_vals[:, 1], mode='markers', marker=go.scatter.Marker(color="red"),
                       name="negative"))

        fig.update_layout(title=plot_title)

        # Log the plot
        wandb.log({f"Repertoire Embeddings/{method}/{dl_name}": fig}, step=step)
        fig.data = []

    def split_outputs(self, all_values, all_targets, flatten: bool = True, sequence_level: bool = False):
        """
        Split logits (or attentions) between positive and negative repertoires (or sequences)
        Args:
            all_values: raw values of the positive and negative samples
            all_targets: targets of the samples

        Returns:
            positive_raw: raw values of positive samples
            negative_raw: raw values of negative samples
        """
        all_values = all_values.detach().cpu()
        all_targets = all_targets.flatten().detach().cpu()
        if sequence_level:
            TP = all_values[np.where(all_targets == 3)[0]].detach().cpu().numpy().flatten().tolist()
            TN = all_values[np.where(all_targets == 0)[0]].detach().cpu().numpy().flatten().tolist()
            FN = all_values[np.where(all_targets == 2)[0]].detach().cpu().numpy().flatten().tolist()
            FP = all_values[np.where(all_targets == 1)[0]].detach().cpu().numpy().flatten().tolist()
            return {"TP": TP, "TN": TN, "FN": FN, "FP": FP}

        else:
            pos_vals = all_values[np.where(all_targets)[0]].detach().cpu().numpy()
            neg_vals = all_values[np.where(np.logical_not(all_targets))[0]].detach().cpu().numpy()
            if flatten:
                pos_vals = pos_vals.flatten()
                neg_vals = neg_vals.flatten()

            return pos_vals, neg_vals

    def log_per_kernel(self, split_rep_embs):
        config = dict(opacity=0.6, histnorm="percent", nbinsx=50)
        split_rep_embs = torch.tensor(np.array(split_rep_embs[0]))
        for kernel_idx in range(split_rep_embs.shape[1]):
            pos_per_dim = split_rep_embs[0, :, kernel_idx]
            neg_per_dim = split_rep_embs[1, :, kernel_idx]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=pos_per_dim.cpu().numpy(), **config, name="pos"))
            fig.add_trace(go.Histogram(x=neg_per_dim.cpu().numpy(), **config, name="neg"))
            wandb.log({f'Per Kernel Activations/kernel_{str(kernel_idx)}': fig})


def get_outputs(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, rep_level_eval: bool = True,
                show_progress: bool = True, device: torch.device = torch.device('cuda:0')):
    with torch.no_grad():
        model.to(device=device)
        all_logits = []
        all_emb_reps = []
        all_targets = []
        all_attentions = []
        all_seq_targets = []
        all_pools = []
        all_seq_counts = []
        all_n_sequences = []
        for scoring_data in tqdm(dataloader, total=len(dataloader), desc="Evaluating model", disable=not show_progress):
            # Get samples as lists
            targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence, pools_per_sequence, sample_ids = scoring_data

            # Apply attention-based sequence reduction and create minibatch
            targets, inputs, sequence_lengths, sequence_counts, sequence_labels, sequence_pools, n_sequences, sequence_attentions = model.reduce_and_stack_minibatch(
                targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence, pools_per_sequence,
                attention_based=rep_level_eval)

            if rep_level_eval:
                # Compute predictions from reduced sequences
                raw_outputs, emb_reps_after_attention = model(inputs_flat=inputs,
                                                              sequence_lengths_flat=sequence_lengths,
                                                              n_sequences_per_bag=n_sequences,
                                                              sequence_counts=sequence_counts,
                                                              sequence_labels=sequence_labels)

                # Store predictions and labels
                all_logits.append(raw_outputs.detach())
                all_targets.append(targets.detach())
                all_emb_reps.append(emb_reps_after_attention.detach())

            all_attentions.append(sequence_attentions.float().detach())
            all_seq_counts.append(sequence_counts.detach())
            all_n_sequences.append(n_sequences.detach())
            all_seq_targets.append(sequence_labels.detach())
            all_pools.append(sequence_pools.detach())
        # Compute scores
        all_logits = torch.cat(all_logits, dim=0)
        all_emb_reps = torch.cat(all_emb_reps, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_seq_counts = torch.cat(all_seq_counts, dim=0)
        all_n_sequences = torch.cat(all_n_sequences, dim=0)
        all_attentions = torch.cat(all_attentions, dim=0)
        all_seq_targets = torch.cat(all_seq_targets, dim=0)
        all_pools = torch.cat(all_pools, dim=0)
        # get_boundaries(all_emb_reps, all_targets)

    return all_logits, all_targets, all_emb_reps, all_attentions, all_seq_targets, all_pools, all_seq_counts, all_n_sequences


def get_boundaries(all_embs, all_targets):
    all_angles = torch.stack([get_avg_angle(all_embs[i, :]) for i in range(all_targets.shape[0])])
    all_l1_norms = torch.stack(
        [torch.linalg.vector_norm(all_embs[i, :], ord=1) for i in range(all_targets.shape[0])])
    all_l2_norms = torch.stack(
        [torch.linalg.vector_norm(all_embs[i, :], ord=2) for i in range(all_targets.shape[0])])
    if torch.any(all_targets):
        max_pos_l1_norm = torch.max(all_l1_norms[torch.nonzero(all_targets.flatten())])
        min_pos_l1_norm = torch.min(all_l1_norms[torch.nonzero(all_targets.flatten())])
        max_pos_l2_norm = torch.max(all_l2_norms[torch.nonzero(all_targets.flatten())])
        min_pos_l2_norm = torch.min(all_l2_norms[torch.nonzero(all_targets.flatten())])
        max_pos_ang = torch.max(all_angles[torch.nonzero(all_targets.flatten())])
        min_pos_ang = torch.min(all_angles[torch.nonzero(all_targets.flatten())])
    else:
        max_pos_l1_norm, min_pos_l1_norm, max_pos_l2_norm, min_pos_l2_norm, max_pos_ang, min_pos_ang = (
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")

    max_neg_l1_norm = torch.max(all_l1_norms[torch.nonzero(1 - all_targets.flatten())])
    min_neg_l1_norm = torch.min(all_l1_norms[torch.nonzero(1 - all_targets.flatten())])
    max_neg_l2_norm = torch.max(all_l2_norms[torch.nonzero(1 - all_targets.flatten())])
    min_neg_l2_norm = torch.min(all_l2_norms[torch.nonzero(1 - all_targets.flatten())])
    max_neg_ang = torch.max(all_angles[torch.nonzero(1 - all_targets.flatten())])
    min_neg_ang = torch.min(all_angles[torch.nonzero(1 - all_targets.flatten())])

    print(f"min pos angle: {min_pos_ang}, max pos angle: {max_pos_ang}")
    print(f"min neg angle: {min_neg_ang}, max neg angle: {max_neg_ang}")
    print(f"min pos l1_norm: {min_pos_l1_norm}, max pos l1_norm: {max_pos_l1_norm}")
    print(f"min neg l1_norm: {min_neg_l1_norm}, max neg l1_norm: {max_neg_l1_norm}")
    print(f"min pos l2_norm: {min_pos_l2_norm}, max pos l2_norm: {max_pos_l2_norm}")
    print(f"min neg l2_norm: {min_neg_l2_norm}, max neg l2_norm: {max_neg_l2_norm}")


def get_avg_angle(x):
    n_dims = x.shape[0]
    axis = torch.eye(n_dims, dtype=x.dtype, device=x.device)
    cos_angles = torch.matmul(x, axis) / (torch.norm(x) * torch.norm(axis, dim=0))
    angles = torch.acos(cos_angles) * 180 / torch.pi
    return torch.mean(angles)


def perform_pca(split_rep_embs):
    pca = PCA(n_components=2)
    pca.fit(np.concatenate([split_rep_embs[0], split_rep_embs[1]]))
    pos_embs, neg_embds = [pca.transform(x) for x in split_rep_embs]
    return pos_embs, neg_embds


def perform_tsne(split_rep_embs):
    pos_len = split_rep_embs[0].shape[0]
    tsne = TSNE(random_state=42, n_components=2, verbose=0, perplexity=40, n_iter=400)
    results = tsne.fit_transform(np.concatenate([split_rep_embs[0], split_rep_embs[1]]))
    pos_embs = results[:pos_len, :]
    neg_embds = results[pos_len:, :]
    return pos_embs, neg_embds


def plot_motifs(motif_matrix, num_aas: int = 3, kernel_size: int = 5):
    indices = np.argsort(-motif_matrix, axis=1)[:, :num_aas]

    mask = np.zeros_like(motif_matrix, dtype=bool)
    mask[np.arange(kernel_size)[:, None], indices] = True

    # set the unwanted values in the original motif_matrix to zero
    motif_matrix[~mask] = 0

    max_values = np.max(motif_matrix, axis=1).reshape(-1, 1)
    motif_matrix = motif_matrix / max_values * 5

    matrix_df = pd.DataFrame(motif_matrix, columns=[
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

    # create Logo object
    crp_logo = logomaker.Logo(matrix_df,
                              shade_below=.5,
                              fade_below=.5, )

    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    crp_logo.ax.xaxis.set_ticks_position('none')
    crp_logo.ax.xaxis.set_tick_params(pad=-1)
    crp_logo.ax.set_ylim([0, 15])
    return crp_logo.fig


def get_split_inds(n_folds, cohort, n_tr, n_v, n_te, seed):
    assert n_tr + n_v + n_te <= 120, "Too many samples requested"
    split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_separate_test_correct.pkl"
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)["inds"]
    if cohort == 2:
        split_inds = split_inds[-1]
    elif cohort == 1:
        split_inds = split_inds[:-1]
        split_inds = [a for b in split_inds for a in b]
    else:
        split_inds = [a for b in split_inds for a in b]
    np.random.seed(seed)
    np.random.shuffle(split_inds)
    # split_inds = [split_inds[i * int(len(split_inds) / n_folds): (i + 1) * int(len(split_inds) / n_folds)] for i in
    #               range(n_folds)]
    train_split_inds = split_inds[:n_tr]
    val_split_inds = split_inds[n_tr: n_tr + n_v]
    test_split_inds = split_inds[n_tr + n_v: n_tr + n_v + n_te]
    return [test_split_inds, train_split_inds, val_split_inds]


def get_splits_new_emerson(n_folds, cohort, n_tr, n_v, n_te, seed):
    assert n_tr + n_v + n_te <= 120, "Too many samples requested"
    if cohort == 2:
        split_inds = np.arange(120)
    np.random.seed(seed)
    np.random.shuffle(split_inds)
    # split_inds = [split_inds[i * int(len(split_inds) / n_folds): (i + 1) * int(len(split_inds) / n_folds)] for i in
    #               range(n_folds)]
    train_split_inds = split_inds[:n_tr]
    val_split_inds = split_inds[n_tr: n_tr + n_v]
    test_split_inds = split_inds[n_tr + n_v: n_tr + n_v + n_te]
    return [test_split_inds, train_split_inds, val_split_inds]


def get_correct_indices(seed):
    split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_separate_test_correct.pkl"
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)["inds"]
    test_split_inds = list(split_inds[-1])
    split_inds = split_inds[:-1]
    split_inds = [a for b in split_inds for a in b]

    np.random.seed(seed)
    np.random.shuffle(split_inds)
    folds = 4
    split_inds = [split_inds[i * int(len(split_inds) / folds): (i + 1) * int(len(split_inds) / folds)] for i in
                  range(folds)]
    train_split_inds = split_inds[:3]
    val_split_inds = split_inds[3]
    return [test_split_inds, *train_split_inds, val_split_inds]


def get_cherry_picked_inds(cohort: int = 1, n_t: int = 30, n_v: int = 160, best_pos: bool = True,
                           best_neg: bool = True):
    split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_separate_test_correct.pkl"
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)["inds"]
        train_split_inds = []
        if cohort == 2:
            split_inds = split_inds[-1]
            # Perfect indices: High for pos, low for neg
            # train_split_inds = [731, 704, 753, 762, 730, 715, 686, 691, 758, 703, 710, 669, 709, 712, 713, 718, 723,
            #                     728, 683, 687]
            # High for pos and neg
            # train_split_inds = [731, 704, 753, 762, 730, 715, 686, 691, 758, 703, 772, 692, 700, 705, 706, 734, 666,
            #                     696, 750, 754]
            # Low for pos and High for neg
            # train_split_inds = [697, 680, 720, 708, 732, 768, 773, 745, 748, 749, 772, 692, 700, 705, 706, 734, 666,
            #                     696, 750, 754]
            # Low for pos and for neg
            # train_split_inds = [697, 680, 720, 708, 732, 768, 773, 745, 748, 749, 710, 669, 709, 712, 713, 718, 723,
            #                     728, 683, 687]
            positive = [779, 756, 746, 744, 774, 721, 737, 724, 719, 765, 729, 783, 781, 773, 740, 696, 754, 671, 697,
                        768, 760, 668, 741, 755, 687, 682, 673, 718, 717, 782, 763, 677, 690, 753, 748, 708, 716, 732,
                        704, 726, 699, 714, 707, 723, 679, 772, 665, 731, 739, 694, 752]

            negative = [734, 747, 778, 759, 749, 693, 758, 745, 775, 735, 761, 685, 769, 725, 757, 751, 713, 736, 715,
                        733, 777, 720, 722, 701, 750, 728, 730, 712, 669, 706, 705, 703, 695, 764, 738, 689, 684, 770,
                        743, 678, 672, 666, 784, 762, 766, 767, 771, 776, 780, 688, 667, 670, 674, 675, 676, 680, 681,
                        683, 686, 742, 691, 692, 698, 700, 702, 709, 710, 711, 727]

        elif cohort == 1:
            split_inds = split_inds[:-1]
            split_inds = [a for b in split_inds for a in b]
            positive = [307, 161, 473, 208, 236, 343, 508, 386, 28, 551, 231, 215, 233, 213, 542, 63, 528, 283, 235,
                        415, 290, 462, 380, 648, 295, 568, 658, 664, 437, 83, 164, 168, 596, 26, 399, 35, 111, 650, 418,
                        94, 326, 287, 306, 285, 172, 641, 554, 11, 34, 105, 15, 87, 126, 237, 574, 495, 144, 260, 385,
                        67, 409, 222, 467, 365, 254, 275, 362, 345, 459, 268, 55, 299, 464, 518, 604, 185, 108, 615, 54,
                        25, 639, 524, 463, 396, 192, 81, 112, 361, 581, 656, 57, 494, 229, 557, 273, 546, 389, 412, 378,
                        419, 424, 586, 321, 649, 127, 265, 88, 196, 489, 247, 174, 477, 152, 342, 301, 86, 141, 68, 188,
                        328, 398, 182, 522, 202, 391, 388, 527, 445, 417, 513, 349, 82, 2, 427, 241, 240, 334, 250, 189,
                        223, 123, 16, 532, 548, 550, 622, 274, 590, 592, 539, 53, 602, 319, 194, 316, 584, 486, 576,
                        538, 535, 92, 322, 133, 654, 110, 523, 259, 176, 272, 52, 30, 140, 382, 659, 565, 148, 93, 158,
                        500, 488, 482, 428, 267, 509, 99, 340, 177, 187, 280, 264, 210, 198, 333, 411, 6, 404, 376, 125,
                        446, 351, 405, 390, 327, 184, 318, 291, 485, 438, 453, 569, 497, 616, 470, 332, 600, 620, 452,
                        635, 599, 136, 163, 493, 206, 491, 638, 634, 400, 121, 323, 618, 570, 149, 142, 607, 201, 628,
                        309, 567, 41, 512, 245, 506, 487, 360, 119, 324, 128, 170, 359, 337, 19, 367, 255, 277, 279,
                        162, 315, 450, 263, 344, 253]
            negative = [230, 167, 529, 101, 221, 348, 471, 269, 393, 49, 329, 138, 43, 218, 186, 431, 284, 249, 154,
                        104, 226, 37, 31, 66, 39, 595, 292, 354, 79, 439, 100, 276, 481, 124, 645, 89, 132, 545, 180,
                        45, 78, 14, 643, 147, 651, 243, 251, 534, 338, 454, 289, 238, 220, 183, 61, 559, 636, 346, 114,
                        96, 363, 353, 314, 246, 358, 414, 242, 179, 191, 613, 190, 59, 606, 173, 134, 571, 256, 209,
                        225, 262, 563, 395, 469, 547, 435, 429, 294, 451, 553, 536, 603, 579, 281, 311, 543, 560, 293,
                        562, 258, 228, 217, 203, 504, 505, 530, 106, 401, 368, 605, 117, 625, 97, 8, 341, 633, 614, 644,
                        377, 27, 193, 20, 22, 42, 65, 36, 159, 153, 271, 150, 200, 216, 310, 352, 76, 21, 588, 227, 219,
                        171, 165, 122, 120, 71, 270, 46, 303, 38, 373, 653, 631, 413, 379, 577, 433, 461, 566, 514, 525,
                        540, 519, 516, 502, 483, 544, 422, 143, 476, 197, 442, 432, 199, 248, 297, 317, 330, 331, 312,
                        339, 239, 169, 369, 298, 139, 137, 420, 449, 80, 484, 32, 13, 10, 531, 3, 549, 647, 597, 558,
                        47, 661, 623, 90, 91, 181, 73, 64, 578, 40, 29, 113, 58, 151, 499, 537, 160, 72, 145, 443, 448,
                        77, 205, 479, 583, 69, 23, 541, 617, 166, 335, 175, 421, 455, 116, 475, 12, 498, 336, 304, 430,
                        109, 533, 266, 261, 356, 257, 252, 102, 95, 610, 425, 407, 406, 392, 384, 372, 496, 350, 521,
                        300, 572, 288, 601, 608, 440, 612, 232, 619, 0, 207, 5, 9, 24, 178, 50, 51, 129, 98, 564, 573,
                        646, 1, 85, 74, 44, 56, 103, 107, 18, 115, 118, 135, 157, 552, 282, 561, 296, 302, 305, 347,
                        355, 370, 492, 371, 460, 403, 447]

    n_per_class = int(n_t / 2)
    if best_pos:
        train_split_inds.extend(positive[:n_per_class])
    else:
        train_split_inds.extend(positive[-n_per_class:])
    if best_neg:
        train_split_inds.extend(negative[-n_per_class:])
    else:
        train_split_inds.extend(negative[:n_per_class])
    np.random.seed(0)
    np.random.shuffle(train_split_inds)
    val_split_inds = [x for x in split_inds if x not in train_split_inds]
    np.random.shuffle(val_split_inds)
    val_split_inds = val_split_inds[:n_v]
    return [[], train_split_inds, val_split_inds]


def get_original_inds():
    # Get file for dataset splits
    split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_splits.pkl"
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    return split_inds


if __name__ == '__main__':
    run = wandb.init(project="Test", reinit=True)  # , tags=config["tag"])
    cnn_weights = np.random.randn(4, 23, 5)
    plot_motifs(cnn_weights)
