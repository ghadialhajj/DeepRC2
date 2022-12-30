# -*- coding: utf-8 -*-
"""
Utility functions and classes

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os

import numpy as np
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
    update_progess_bar = tqdm.tqdm(total=stream_size, disable=not verbose,
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
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def log_stats(self, model: torch.nn.Module, device=torch.device('cuda:0'), step: int = 0,
                  logg_and_att: bool = True):
        for dl_name, dl in self.dataloaders.items():
            split_logits, split_attentions, split_rep_embs = self.get_values_per_dl(model, dl, device=device)
            split_rep_embs_pca = perform_pca(split_rep_embs)
            split_rep_embs_tsne = perform_pca(split_rep_embs)
            self.log_repertoire_rep(dl_name, split_rep_embs_pca, split_rep_embs_tsne, step)
            if logg_and_att:
                self.log_logits(dl_name, split_logits, step)

                self.log_attention(dl_name, split_attentions, step)

    def log_repertoire_rep(self, dl_name, split_rep_embs_pca, split_rep_embs_tsne, step):
        self.plot_scatter(pos_vals=split_rep_embs_pca[0], neg_vals=split_rep_embs_pca[1], dl_name=dl_name,
                          plot_title=f"Positive (B) and Negative (R) PCAcomp", step=step, method="PCA")
        self.plot_scatter(pos_vals=split_rep_embs_tsne[0], neg_vals=split_rep_embs_tsne[1], dl_name=dl_name,
                          plot_title=f"Positive (B) and Negative (R) TSNEcomp", step=step, method="TSNE")

    def log_attention(self, dl_name, split_attentions, step):
        self.plot_histogram(data={"observed": split_attentions[0],
                                  "unobserved": split_attentions[1], "negative": split_attentions[2]}, dl_name=dl_name,
                            plot_title=f"Positive (B) and Negative (R) Raw Att", xaxis_title="Attention value",
                            step=step)

    def log_logits(self, dl_name, split_logits, step):
        self.plot_histogram(data={"positive": split_logits[0], "negative": split_logits[1]}, dl_name=dl_name,
                            plot_title=f"Positive (B) and Negative (R) Logits", xaxis_title="Logit value",
                            step=step)

    def get_values_per_dl(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                          show_progress: bool = True,
                          device: torch.device = torch.device('cuda:0')):
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
        all_logits, all_targets, all_attentions, all_seq_targets, all_seq_pools, all_emb_reps = get_outputs(model=model,
                                                                                                            dataloader=dataloader,
                                                                                                            device=device,
                                                                                                            show_progress=show_progress)
        split_logits = self.split_outputs(all_logits, all_targets)
        split_attentions = self.split_outputs(all_attentions, all_seq_pools, three_classes=True)
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

    @staticmethod
    def split_outputs(all_values, all_targets, flatten: bool = True, three_classes: bool = False):
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
        if not three_classes:
            pos_vals = all_values[np.where(all_targets)[0]].detach().cpu().numpy()
            neg_vals = all_values[np.where(np.logical_not(all_targets))[0]].detach().cpu().numpy()
            if flatten:
                pos_vals = pos_vals.flatten()
                neg_vals = neg_vals.flatten()

            return pos_vals, neg_vals

        else:
            observed = all_values[np.where(all_targets == 0)[0]].detach().cpu().numpy().flatten().tolist()
            not_observed = all_values[np.where(all_targets == 1)[0]].detach().cpu().numpy().flatten().tolist()
            negative = all_values[np.where(all_targets == 2)[0]].detach().cpu().numpy().flatten().tolist()

            return observed, not_observed, negative


def get_outputs(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                show_progress: bool = True, device: torch.device = torch.device('cuda:1')):
    with torch.no_grad():
        model.to(device=device)
        all_logits = []
        all_emb_reps = []
        all_targets = []
        all_attentions = []
        all_seq_targets = []
        all_seq_pools = []
        for scoring_data in tqdm(dataloader, total=len(dataloader), desc="Evaluating model", disable=not show_progress):
            # Get samples as lists
            targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence, pools_per_sequence, sample_ids = scoring_data

            # Apply attention-based sequence reduction and create minibatch
            targets, inputs, sequence_lengths, sequence_labels, sequence_pools, n_sequences = model.reduce_and_stack_minibatch(
                targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence, pools_per_sequence)

            # Compute predictions from reduced sequences
            raw_outputs, attention_outputs, emb_reps_after_attention = model(inputs_flat=inputs,
                                                                             sequence_lengths_flat=sequence_lengths,
                                                                             sequence_labels_flat=sequence_labels,
                                                                             n_sequences_per_bag=n_sequences)

            # Store predictions and labels
            all_logits.append(raw_outputs.detach())
            all_emb_reps.append(emb_reps_after_attention.detach())
            all_targets.append(targets.detach())
            all_attentions.append(attention_outputs.detach())
            all_seq_targets.append(sequence_labels.detach())
            all_seq_pools.append(sequence_pools.detach())
        # Compute scores
        all_logits = torch.cat(all_logits, dim=0)
        all_emb_reps = torch.cat(all_emb_reps, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_attentions = torch.cat(all_attentions, dim=0)
        all_seq_targets = torch.cat(all_seq_targets, dim=0)
        all_seq_pools = torch.cat(all_seq_pools, dim=0)

    return all_logits, all_targets, all_attentions, all_seq_targets, all_seq_pools, all_emb_reps


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
