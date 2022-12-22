# -*- coding: utf-8 -*-
"""
Utility functions and classes

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os
from typing import Tuple

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


def plot_histogram(pos_att: list, neg_att: list, n_bins: int = 50, dl_name: str = ""):
    plot_title = f"Positive (B) and Negative (R) Raw Att"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    config = dict(opacity=0.6, histnorm="percent", nbinsx=n_bins)

    fig.add_trace(go.Histogram(x=pos_att, **config, name="positive"), secondary_y=False)
    fig.add_trace(go.Histogram(x=neg_att, **config, name="negative"), secondary_y=True)

    fig.update_layout(title=plot_title, xaxis_title="Attention value", yaxis_title="Percentage")

    # Log the plot
    wandb.log({dl_name: fig})


def get_attentions_per_dl(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, show_progress: bool = True,
                          device: torch.device = torch.device('cuda:0')) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DeepRC model scores on given dataset for tasks specified in `task_definition`

    Parameters
    ----------
    model: torch.nn.Module
         deeprc.architectures.DeepRC or similar model as PyTorch module
    dataloader: torch.utils.data.DataLoader
         Data loader for dataset to calculate scores on
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
    show_progress: bool
         Show progressbar?
    device: torch.device
         Device to use for computations. E.g. `torch.device('cuda:0')` or `torch.device('cpu')`.

    Returns
    ---------
    scores: dict
        Nested dictionary of format `{task_id: {score_id: score_value}}`, e.g.
        `{"binary_task_1": {"auc": 0.6, "bacc": 0.5, "f1": 0.2, "loss": 0.01}}`. The scores returned are computed using
        the .get_scores() methods of the individual target instances (e.g. `deeprc.task_definitions.BinaryTarget()`).
        See `deeprc/examples/` for examples.
    """
    with torch.no_grad():
        model.to(device=device)
        all_att = []
        all_seq_tar = []

        for scoring_data in tqdm(dataloader, total=len(dataloader), desc="Getting attention",
                                 disable=not show_progress):
            # Get samples as lists
            targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence, sample_ids = scoring_data

            # Apply attention-based sequence reduction and create minibatch
            targets, inputs, sequence_lengths, sequence_labels, n_sequences = model.reduce_and_stack_minibatch(
                targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence)

            # Compute predictions from reduced sequences
            _, attention_outputs = model(inputs_flat=inputs, sequence_lengths_flat=sequence_lengths,
                                         sequence_labels_flat=sequence_labels,
                                         n_sequences_per_bag=n_sequences)

            # Store predictions and labels
            all_att.append(attention_outputs.detach())
            all_seq_tar.append(sequence_labels.detach())

        # Compute scores
        all_att = torch.cat(all_att, dim=0)
        all_seq_tar = torch.cat(all_seq_tar, dim=0)

        # First object is the attention values for the positive sequences, the second, the negative ones.
        return all_att[all_seq_tar.nonzero().flatten()], all_att[(all_seq_tar == 0).nonzero().flatten()]


def log_attentions(model: torch.nn.Module, dataloaders: dict, device=torch.device('cuda:0')):
    for dl_name, dl in dataloaders.items():
        pos_att, neg_att = get_attentions_per_dl(model, dl, device=device)
        plot_histogram(pos_att=pos_att.flatten().detach().cpu().tolist(),
                       neg_att=neg_att.flatten().detach().cpu().tolist(),
                       dl_name=dl_name)
