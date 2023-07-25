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

    def log_stats(self, model: torch.nn.Module, device=torch.device('cuda:0'), step: int = 0,
                  log_and_att_hists: bool = False, log_per_kernel: bool = False):
        """
        Logs model statistics including repertoire embeddings, logits, and attentions for each dataloader.
        """
        print("Logging stats:")
        for dl_name, dl in self.dataloaders.items():
            split_logits, split_attentions, split_rep_embs = self.get_values_per_dl(model, dl, device=device)
            split_rep_embs_pca = perform_pca(split_rep_embs)
            # split_rep_embs_tsne = perform_tsne(split_rep_embs)
            self.log_repertoire_rep(dl_name, split_rep_embs_pca, None, step)
            if log_and_att_hists:
                self.log_logits(dl_name, split_logits, step)
                self.log_attention(dl_name, split_attentions, step)
                if log_per_kernel and dl_name == "validationset_eval":
                    self.log_per_kernel(split_rep_embs)

    def log_repertoire_rep(self, dl_name, split_rep_embs_pca, split_rep_embs_tsne, step):
        self.plot_scatter(pos_vals=split_rep_embs_pca[0], neg_vals=split_rep_embs_pca[1], dl_name=dl_name,
                          plot_title=f"Positive (B) and Negative (R) PCAcomp", step=step, method="PCA")
        # self.plot_scatter(pos_vals=split_rep_embs_tsne[0], neg_vals=split_rep_embs_tsne[1], dl_name=dl_name,
        #                   plot_title=f"Positive (B) and Negative (R) TSNEcomp", step=step, method="TSNE")

    def log_attention(self, dl_name, split_attentions, step):
        data = {"observed": split_attentions[0],
                "unobserved": split_attentions[1], "negative": split_attentions[2]}
        if self.with_FPs:
            data.update({"false_positives": split_attentions[3]})
        self.plot_histogram(data, dl_name=dl_name,
                            plot_title=f"Positive (B) and Negative (R) Raw Att", xaxis_title="Attention value",
                            step=step)

    def log_logits(self, dl_name, split_logits, step):
        self.plot_histogram(data={"positive": split_logits[0], "negative": split_logits[1]}, dl_name=dl_name,
                            plot_title=f"Positive (B) and Negative (R) Logits", xaxis_title="Logit value",
                            step=step)

    def get_values_per_dl(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                          show_progress: bool = True, device: torch.device = torch.device('cuda:0')):
        """Compute DeepRC model scores on given dataset for tasks specified in `task_definition`

        Parameters
        ----------
        model: torch.nn.Module
             deeprc.architectures.DeepRC or similar model as PyTorch module
        dataloader: torch.utils.data.DataLoader
             Data loader for dataset to calculate scores on
        show_progress: bool
             Show progressbar?
        device: torch.device
             Device to use for computations. E.g. `torch.device('cuda:0')` or `torch.device('cpu')`.

        Returns
        ---------
        scores: dict
            Nested dictionary of format `{task_id: {score_id: score_value}}`, e.g.
        """
        all_logits, all_targets, all_attentions, all_seq_targets, all_seq_counts, all_emb_reps = get_outputs(
            model=model,
            dataloader=dataloader,
            device=device,
            show_progress=show_progress)
        split_logits = self.split_outputs(all_logits, all_targets)
        split_attentions = self.split_outputs(all_attentions, all_seq_targets, sequence_level=True)
        split_rep_embs = self.split_outputs(all_emb_reps, all_targets, flatten=False)
        return split_logits, split_attentions, split_rep_embs

    @staticmethod
    def plot_histogram(data: dict, n_bins: int = 50, dl_name: str = "", plot_title: str = "",
                       xaxis_title: str = "Attention value", step: int = 0):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        config = dict(opacity=0.6, histnorm="percent", nbinsx=n_bins)

        for name, values in data.items():
            secondary_y = True if name == "negative" else False
            fig.add_trace(go.Histogram(x=values, **config, name=name), secondary_y=secondary_y)

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
            observed = all_values[np.where(all_targets == 0)[0]].detach().cpu().numpy().flatten().tolist()
            not_observed = all_values[np.where(all_targets == 1)[0]].detach().cpu().numpy().flatten().tolist()
            negative = all_values[np.where(all_targets == 2)[0]].detach().cpu().numpy().flatten().tolist()
            if not self.with_FPs:
                return observed, not_observed, negative
            else:
                false_positives = all_values[np.where(all_targets == 3)[0]].detach().cpu().numpy().flatten().tolist()
            return observed, not_observed, negative, false_positives

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


def get_outputs(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                show_progress: bool = True, device: torch.device = torch.device('cuda:1')):
    print("Weight Average: ", torch.mean(model.sequence_embedding.network[0].weight))
    with torch.no_grad():
        model.to(device=device)
        all_logits = []
        all_emb_reps = []
        all_targets = []
        all_attentions = []
        all_seq_targets = []
        all_seq_counts = []
        for scoring_data in tqdm(dataloader, total=len(dataloader), desc="Evaluating model", disable=not show_progress):
            # Get samples as lists
            targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence, sample_ids = scoring_data

            # Apply attention-based sequence reduction and create minibatch
            targets, inputs, sequence_lengths, sequence_counts, sequence_labels, n_sequences = model.reduce_and_stack_minibatch(
                targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence)

            # Compute predictions from reduced sequences
            raw_outputs, attention_outputs, emb_reps_after_attention = model(inputs_flat=inputs,
                                                                             sequence_lengths_flat=sequence_lengths,
                                                                             n_sequences_per_bag=n_sequences,
                                                                             sequence_counts=sequence_counts)

            # Store predictions and labels
            all_logits.append(raw_outputs.detach())
            all_emb_reps.append(emb_reps_after_attention.detach())
            all_targets.append(targets.detach())
            all_seq_counts.append(sequence_counts.detach())
            all_attentions.append(attention_outputs.detach())
            all_seq_targets.append(sequence_labels.detach())
        # Compute scores
        all_logits = torch.cat(all_logits, dim=0)
        all_emb_reps = torch.cat(all_emb_reps, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_seq_counts = torch.cat(all_seq_counts, dim=0)
        all_attentions = torch.cat(all_attentions, dim=0)
        all_seq_targets = torch.cat(all_seq_targets, dim=0)

    return all_logits, all_targets, all_attentions, all_seq_targets, all_seq_counts, all_emb_reps


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


if __name__ == '__main__':
    run = wandb.init(project="Test", reinit=True)  # , tags=config["tag"])
    cnn_weights = np.random.randn(4, 23, 5)
    plot_motifs(cnn_weights)
