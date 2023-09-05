# -*- coding: utf-8 -*-
"""
Simple example for training binary DeepRC classifier on datasets of category "real-world data"

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import argparse
import numpy as np
import torch
import dill as pkl
import wandb

from deeprc.dataset_readers import make_dataloaders, plain_log_sequence_count_scaling
from deeprc.predefined_datasets import cmv_dataset
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.task_definitions import TaskDefinition, BinaryTarget, Sequence_Target
from deeprc.training import train, evaluate
from deeprc.utils import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--n_updates', help='Number of updates to train for. Default: int(1e5)', type=int,
                    default=int(1e5))
parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping.'
                                          ' Default: int(5e3)', type=int,
                    default=int(2.5e3))
parser.add_argument('--device', help='Device to use for NN computations, as passed to `torch.device()`. '
                                     'Default: "cuda:0".',
                    type=str, default="cuda:0")
parser.add_argument('--rnd_seed', help='Random seed to use for PyTorch and NumPy. Results will still be '
                                       'non-deterministic due to multiprocessing but weight initialization will be the'
                                       ' same). Default: 0.',
                    type=int, default=0)
args = parser.parse_args()
# Set computation device
device = torch.device(args.device)
# Set random seed (will still be non-deterministic due to multiprocessing but weight initialization will be the same)
torch.manual_seed(args.rnd_seed)
np.random.seed(args.rnd_seed)

#
# Get dataset
#
# Get data loaders for training set and training-, validation-, and test-set in evaluation mode (=no random subsampling)
# task_definition, trainingset, trainingset_eval, validationset_eval, testset_eval = cmv_dataset()
config = {"sequence_reduction_fraction": 0.01, "reduction_mb_size": int(5e3),
          "prop": 0.2,
          "dataset": "AIRR", "pos_weight_seq": 100, "pos_weight_rep": 1., "Branch": "Emerson",
          "dataset_type": "emerson_linz_2", "attention_temperature": 1, "best_pos": None, "best_neg": None,
          "max_factor": 150, "consider_seq_counts": True, "consider_seq_counts_after_cnn": False,
          "consider_seq_counts_after_att": False, "consider_seq_counts_after_softmax": False}


def get_original_inds():
    # Get file for dataset splits
    split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_splits.pkl"
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    return split_inds


root_dir = "/storage/ghadia/DeepRC2/deeprc"

task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
    BinaryTarget(column_name='CMV', true_class_value='+'),
    Sequence_Target(pos_weight=1, weigh_seq_by_weight=False, weigh_pos_by_inverse=False,
                    normalize=False, add_in_loss=False,
                    device=device), ]).to(device=device)
#
trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
    task_definition=task_definition,
    metadata_file="/storage/ghadia/DeepRC2/deeprc/datasets/emerson_linz_2/AIRR/metadata.csv",
    metadata_file_column_sep=",",
    n_worker_processes=4,
    repertoiresdata_path="/storage/ghadia/DeepRC2/deeprc/datasets/emerson_linz_2/AIRR/repertoires",
    metadata_file_id_column='filename',
    sequence_column='cdr3_aa',
    sequence_counts_column="duplicate_count",
    sequence_labels_column='matched',
    sample_n_sequences=10000,
    sequence_counts_scaling_fn=plain_log_sequence_count_scaling,
    # if strategy == "PDRC" else log_sequence_count_scaling,
    with_test=False,
    split_inds=get_original_inds(),
    force_pos_in_subsampling=False,
    min_count=1,
    max_factor=1
)

#
# Create DeepRC Network
#
# Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=9, n_kernels=32, n_layers=1)
# Create attention network
attention_network = AttentionNetwork(n_input_features=32, n_layers=2, n_units=32)
# Create output network
output_network = OutputNetwork(n_input_features=32, n_output_features=task_definition.get_n_output_features(),
                               n_layers=1, n_units=32)
# Combine networks to DeepRC network
model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
               attention_network=attention_network,
               output_network=output_network,
               consider_seq_counts=True, n_input_features=20, add_positional_information=True,
               sequence_reduction_fraction=0.1, reduction_mb_size=int(5e4),
               consider_seq_counts_after_cnn=False,
               training_mode=True,
               consider_seq_counts_after_att=False,
               consider_seq_counts_after_softmax=False,
               device=device,
               forced_attention=False, force_pos_in_attention=False,
               temperature=1).to(device=device)

dl_dict = {"trainingset_eval": trainingset_eval, "validationset_eval": validationset_eval, "testset_eval": testset_eval}

logger = Logger(dataloaders=dl_dict, with_FPs=False)
run = wandb.init(project="CM - Scaling", group=f"PDRC_CMV_reproduction")
wandb.config.update(args)
wandb.config.update(config)

#
# Train DeepRC model
#
train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
      trainingset_eval_dataloader=trainingset_eval,
      early_stopping_target_id='CMV',  # Get model that performs best for this task
      validationset_eval_dataloader=validationset_eval, n_updates=args.n_updates, evaluate_at=args.evaluate_at,
      device=device, results_directory="results/cmv",  # Here our results and trained models will be stored
      logger=logger, train_then_freeze=False, staged_training=False, plain_DeepRC=True, rep_loss_only=False
      )
# You can use "tensorboard --logdir [results_directory] --port=6060" and open "http://localhost:6060/" in your
# web-browser to view the progress

#
# Evaluate trained model on testset
#
scores = evaluate(model=model, dataloader=testset_eval, task_definition=task_definition, device=device)
print(f"Test scores:\n{scores}")
