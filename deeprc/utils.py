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


def log_logits_and_attentions(model: torch.nn.Module, dataloaders: dict, device=torch.device('cuda:0'), step: int = 0):
    for dl_name, dl in dataloaders.items():
        split_logits, split_attentions = get_values_per_dl(model, dl, device=device)
        plot_histogram(pos_vals=split_logits[0].flatten().detach().cpu().tolist(),
                       neg_vals=split_logits[1].flatten().detach().cpu().tolist(), dl_name=dl_name,
                       plot_title=f"Positive (B) and Negative (R) Logits", xaxis_title="Logit value", step=step)

        plot_histogram(pos_vals=split_attentions[0].flatten().detach().cpu().tolist(),
                       neg_vals=split_attentions[1].flatten().detach().cpu().tolist(), dl_name=dl_name,
                       plot_title=f"Positive (B) and Negative (R) Raw Att", xaxis_title="Attention value", step=step)


def get_values_per_dl(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, show_progress: bool = True,
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
    all_logits, all_targets, all_attentions, all_seq_targets = get_outputs(model=model, dataloader=dataloader,
                                                                           device=device,
                                                                           show_progress=show_progress)

    return split_outputs(all_logits, all_targets), split_outputs(all_attentions, all_seq_targets)


def plot_histogram(pos_vals: list, neg_vals: list, n_bins: int = 50, dl_name: str = "", plot_title: str = "",
                   xaxis_title: str = "Attention value", step: int = 0):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    config = dict(opacity=0.6, histnorm="percent", nbinsx=n_bins)

    fig.add_trace(go.Histogram(x=pos_vals, **config, name="positive"), secondary_y=False)
    fig.add_trace(go.Histogram(x=neg_vals, **config, name="negative"), secondary_y=True)

    fig.update_layout(title=plot_title, xaxis_title=xaxis_title, yaxis_title="Percentage")

    # Log the plot
    wandb.log({f"{xaxis_title}/{dl_name}": fig}, step=step)


def split_outputs(all_values, all_targets):
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
    all_targets = all_targets.detach().cpu()
    return all_values[np.where(all_targets)], all_values[np.where(np.logical_not(all_targets))]


def get_outputs(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                show_progress: bool = True, device: torch.device = torch.device('cuda:1')):
    with torch.no_grad():
        model.to(device=device)
        all_logits = []
        all_targets = []
        all_attentions = []
        all_seq_targets = []
        for scoring_data in tqdm(dataloader, total=len(dataloader), desc="Evaluating model", disable=not show_progress):
            # Get samples as lists
            targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence, sample_ids = scoring_data

            # Apply attention-based sequence reduction and create minibatch
            targets, inputs, sequence_lengths, sequence_labels, n_sequences = model.reduce_and_stack_minibatch(
                targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence)

            # Compute predictions from reduced sequences
            raw_outputs, attention_outputs, _ = model(inputs_flat=inputs, sequence_lengths_flat=sequence_lengths,
                                                      sequence_labels_flat=sequence_labels,
                                                      n_sequences_per_bag=n_sequences)

            # Store predictions and labels
            all_logits.append(raw_outputs.detach())
            all_targets.append(targets.detach())
            all_attentions.append(attention_outputs.detach())
            all_seq_targets.append(sequence_labels.detach())
        # Compute scores
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_attentions = torch.cat(all_attentions, dim=0)
        all_seq_targets = torch.cat(all_seq_targets, dim=0)

    return all_logits, all_targets, all_attentions, all_seq_targets
