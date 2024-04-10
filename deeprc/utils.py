# -*- coding: utf-8 -*-
"""
Utility functions and classes

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os
from typing import Tuple

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
from deeprc.task_definitions import TaskDefinition
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
    def __init__(self, dataloaders, root_dir: str, strategy="", experiment: str = None, att_hists: bool = False):
        self.dataloaders = dataloaders
        self.strategy = strategy
        self.root_dir = root_dir
        self.experiment = experiment  # find a better name for this
        self.att_hists = att_hists

    def log_motifs(self, params: np.ndarray, step):
        cnn_weights = params[:, :20, :].squeeze()

        for motif_idx in range(cnn_weights.shape[0]):
            motif_matrix = cnn_weights[motif_idx].T
            fig = plot_motifs(motif_matrix)
            wandb.log({f'Motifs/motif_{str(motif_idx)}': wandb.Image(fig)}, step=step)
            plt.close(fig)

    def log_stats(self, step: int, all_logits, all_targets, all_emb_reps, all_attentions, all_pools,
                  log_per_kernel: bool = False, logit_hist: bool = False,
                  dl_name: str = "val_eval_dl"):
        """
        Logs model statistics including repertoire embeddings, logits, and attentions for each dataloader.
        """
        print("Logging stats:")
        # if not dl.batch_sampler.sampler.data_source.indices.any():
        #     continue
        split_logits, split_attentions, split_rep_embs = self.get_values_per_dl(all_logits, all_targets, all_emb_reps,
                                                                                all_attentions, all_pools)
        if split_rep_embs is not None:
            split_rep_embs_pca = perform_pca(split_rep_embs)
            self.log_repertoire_rep(dl_name, split_rep_embs_pca, step)
        if self.att_hists:
            self.log_attention(dl_name, split_attentions, step, save_data=dl_name == "test_eval_dl")
        if logit_hist:
            self.log_logits(dl_name, split_logits, step)
        if log_per_kernel and dl_name == "val_eval_dl":
            self.log_per_kernel(split_rep_embs)

    def log_repertoire_rep(self, dl_name, split_rep_embs_pca, step):
        self.plot_scatter(pos_vals=split_rep_embs_pca[0], neg_vals=split_rep_embs_pca[1], dl_name=dl_name,
                          plot_title=f"Positive (B) and Negative (R) PCAcomp", step=step, method="PCA")

    def log_attention(self, dl_name, split_attentions, step, save_data: bool):
        self.plot_histogram(split_attentions, dl_name=dl_name,
                            plot_title=f"Raw Attention", xaxis_title="Attention value",
                            step=step, save_data=save_data)

    def log_logits(self, dl_name, split_logits, step):
        self.plot_histogram(data={"positive": split_logits[0], "negative": split_logits[1]}, dl_name=dl_name,
                            plot_title=f"Positive (B) and Negative (R) Logits", xaxis_title="Logit value",
                            step=step)

    def get_values_per_dl(self, all_logits, all_targets, all_emb_reps, all_attentions, all_pools):
        split_logits = self.split_outputs(all_logits, all_targets)
        split_attentions = self.split_outputs(all_attentions, all_pools, sequence_level=True)
        split_rep_embs = self.split_outputs(all_emb_reps, all_targets, flatten=False)
        return split_logits, split_attentions, split_rep_embs

    def plot_histogram(self, data: dict, n_bins: int = 50, dl_name: str = "", plot_title: str = "",
                       xaxis_title: str = "Attention value", step: int = 0, save_data: bool = False):
        # fig = make_subplots()
        #
        # min = np.min([np.min(class_data) for class_data in data.values() if class_data.size > 0])
        # max = np.max([np.max(class_data) for class_data in data.values() if class_data.size > 0])
        # bins = np.linspace(min, max, n_bins)
        #
        # pdfs = {class_name: np.histogram(class_data, bins=bins, density=True)[0] for
        #         class_name, class_data in data.items()}
        #
        # pmfs = {class_name: pdfs[class_name] / np.sum(pdfs[class_name]) * 100 for class_name in pdfs}
        #
        # for name, hist in pmfs.items():
        #     fig.add_trace(go.Scatter(x=bins, y=hist, mode='lines', name=name))

        fig = make_subplots()
        config = dict(opacity=0.6, histnorm="percent", nbinsx=n_bins)

        for name, values in data.items():
            fig.add_trace(go.Histogram(x=values, **config, name=name))
        # Set y-axes titles
        fig.update_yaxes(title_text="Postives")
        fig.update_yaxes(title_text="Negatives")

        fig.update_layout(title=plot_title, xaxis_title=xaxis_title, yaxis_title="Percentage", barmode='overlay')
        fig.update_traces(autobinx=False, selector=dict(type='histogram'))

        fig.update_layout(
            title='Histogram of Classes',
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title="Percentage"),
            showlegend=True
        )

        if save_data:
            # create folders if they don't exist first and save the figure
            save_dir = f"{self.root_dir}/results/Attentions/{self.experiment}"
            json_dir = f"{save_dir}/JSON/{self.strategy}"
            png_dir = f"{save_dir}/PNG/{self.strategy}"
            if not os.path.isdir(json_dir):
                os.makedirs(json_dir, exist_ok=True)
            if not os.path.isdir(png_dir):
                os.makedirs(png_dir, exist_ok=True)

            fig.write_image(f"{png_dir}/{wandb.run.name}.png")
            fig.write_json(f"{json_dir}/{wandb.run.name}.json")

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
            HW = all_values[np.where(all_targets == 3)[0]].detach().cpu().numpy().flatten()  # .tolist()
            LW̅ = all_values[np.where(all_targets == 0)[0]].detach().cpu().numpy().flatten()  # .tolist()
            LW = all_values[np.where(all_targets == 2)[0]].detach().cpu().numpy().flatten()  # .tolist()
            HW̅ = all_values[np.where(all_targets == 1)[0]].detach().cpu().numpy().flatten()  # .tolist()
            return {"HW": HW, "LW̅": LW̅, "LW": LW, "HW̅": HW̅}

        else:
            # todo fix for additive
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
                raw_outputs, _, emb_reps_after_attention = model(inputs_flat=inputs,
                                                                 sequence_lengths_flat=sequence_lengths,
                                                                 n_sequences_per_bag=n_sequences,
                                                                 sequence_labels=sequence_labels)

                # Store predictions and labels
                all_logits.append(raw_outputs.detach())
                all_targets.append(targets.detach())
                all_emb_reps.append(emb_reps_after_attention.detach())

            all_attentions.append(sequence_attentions.detach())
            all_seq_counts.append(sequence_counts.detach())
            all_n_sequences.append(n_sequences.detach())
            all_seq_targets.append(sequence_labels.detach())
            all_pools.append(sequence_pools.detach())
        # Compute scores
        if rep_level_eval:
            all_logits = torch.cat(all_logits, dim=0)
            all_emb_reps = torch.cat(all_emb_reps, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
        all_seq_counts = torch.cat(all_seq_counts, dim=0)
        all_n_sequences = torch.cat(all_n_sequences, dim=0)
        all_attentions = torch.cat(all_attentions, dim=0)
        all_seq_targets = torch.cat(all_seq_targets, dim=0)
        all_pools = torch.cat(all_pools, dim=0)

    return all_logits, all_targets, all_emb_reps, all_attentions, all_seq_targets, all_pools, all_seq_counts, all_n_sequences


def perform_pca(split_rep_embs):
    pca = PCA(n_components=2)
    pca.fit(np.concatenate([split_rep_embs[0], split_rep_embs[1]]))
    pos_embs, neg_embds = [pca.transform(x) for x in split_rep_embs]
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


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, task_definition: TaskDefinition,
             logger: [Logger, None], step: [int, None], show_progress: bool = True,
             device: torch.device = torch.device('cuda:1'), log_stats=True, dl_name="val_set_eval",
             ) -> Tuple[dict, dict]:
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
        all_logits, all_targets, all_emb_reps, *_ = get_outputs(
            model=model, dataloader=dataloader, rep_level_eval=True,
            show_progress=show_progress,
            device=device)

        scores = task_definition.get_scores(raw_outputs=all_logits, targets=all_targets)

        _, _, _, all_attentions, all_seq_targets, all_pools, all_seq_counts, all_n_sequences = get_outputs(
            model=model, dataloader=dataloader, rep_level_eval=False,
            show_progress=show_progress,
            device=device)

        sequence_scores = task_definition.get_sequence_scores(raw_attentions=all_attentions.squeeze(),
                                                              sequence_pools=all_pools,
                                                              sequence_counts=all_seq_counts,
                                                              n_sequences=all_n_sequences)

        if log_stats and dl_name in ["val_eval_dl", "test_eval_dl"]:
            logger.log_stats(step, all_logits, all_targets, all_emb_reps, all_attentions, all_pools, dl_name=dl_name,
                             log_per_kernel=False, logit_hist=False)

        return scores, sequence_scores


def eval_on_test(task_definition, best_model, test_eval_dl, logger, device, n_updates):
    classes_dict = task_definition.__sequence_targets__[0].classes_dict
    assert not best_model.training_mode, "Model is in training mode!"
    scores, sequence_scores = evaluate(model=best_model, dataloader=test_eval_dl,
                                       task_definition=task_definition, step=n_updates,
                                       device=device, logger=logger, log_stats=False,
                                       dl_name="test_eval_dl")
    curves = sequence_scores['sequence_class'].pop('curves')
    wandb.run.summary.update(scores["label_positive"])
    wandb.run.summary.update(sequence_scores["sequence_class"])
    # remove random AP scores and curves
    for id, curve in curves.items():
        wandb.log({f"test/label_positive/PRC_{classes_dict[id]}": wandb.Image(curve.figure_)})
        wandb.run.summary.update({f"AP_{classes_dict[id]}": curve.average_precision,
                                  f"ranAP_{classes_dict[id]}": curve.prevalence_pos_label})


def get_hpo_combinations(HPs: dict, default_idx: int = 1):
    # Get the default values (middle values in each list)
    defaults = {key: values[default_idx] for key, values in HPs.items()}

    # Initialize an empty list to store the combinations
    combinations = []

    # Iterate through each key in the dictionary
    for key in HPs.keys():
        # Create combinations where the first and third values are chosen for the current key
        for value in [HPs[key][0], HPs[key][2]]:
            current_combination = {k: value if k == key else defaults[k] for k in HPs.keys()}
            combinations.append(current_combination)
            # Create combinations where the third value is chosen for all keys
    all_default_values = {key: values[default_idx] for key, values in HPs.items()}
    combinations.append(all_default_values)

    return combinations


if __name__ == '__main__':
    run = wandb.init(project="Test", reinit=True)  # , tags=config["tag"])
    cnn_weights = np.random.randn(4, 23, 5)
    plot_motifs(cnn_weights)
