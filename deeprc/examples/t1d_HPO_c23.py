# -*- coding: utf-8 -*-
"""
Example for training DeepRC with CNN sequence embedding in a single-task setting.
IMPORTANT: The used task is a small dummy-task with random data, so DeepRC should over-fit on the main task on the
training set.

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import argparse
import numpy as np
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling, make_test_dataloader
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.training import train, evaluate
import dill as pkl
from itertools import product

#
# Get command line arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--n_updates', help='Number of updates to train for. Recommended: int(1e5). Default: int(1e3)',
                    type=int, default=int(3e5))
parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping. '
                                          'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(5e3))
parser.add_argument('--sample_n_sequences', help='Number of instances to reduce repertoires to during training via'
                                                 'random dropout. This should be less than the number of instances per '
                                                 'repertoire. Only applied during training, not for evaluation. '
                                                 'Default: int(1e4)',
                    type=int, default=int(1e4))
parser.add_argument('--learning_rate', help='Learning rate of DeepRC using Adam optimizer. Default: 1e-4',
                    type=float, default=5e-5)
parser.add_argument('--device', help='Device to use for NN computations, as passed to `torch.device()`. '
                                     'Default: "cuda:0".',
                    type=str, default="cuda:0")
args = parser.parse_args()
# Set computation device
device = torch.device(args.device)
seed = 0
# Set random seed (will still be non-deterministic due to multiprocessing but weight initialization will be the same)
torch.manual_seed(seed)
np.random.seed(seed)

#
# Create Task definitions
#
# Assume we want to train on 1 main task as binary task at column 'binary_target_1' of our metadata file.
task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
    BinaryTarget(  # Add binary classification task with sigmoid output function
        column_name='ML_class',  # Column name of task in metadata file
        true_class_value='T1D',  # Entries with value '+' will be positive class, others will be negative class
        pos_weight=1.,  # We can up- or down-weight the positive class if the classes are imbalanced
    ),
]).to(device=device)

config = {
    'kernel_size': 5,
    'n_kernels': 64,
    'l1_l2': (0, 0),
}
results_directory = "/storage/ghadia/DeepRC2/results/T1D_AC_clip_c23"

# todo use the same split as before
# todo split cohort 1 into tr+val, build a logreg morel based on the predictions of the 5 DeepRC models, then test it
# on cohort 2+3
# generate 1298 integers from 0 to 1297, shuffle them and split them into 5 folds
split_inds = np.random.RandomState(seed=seed).permutation(1298)
split_inds = np.array_split(split_inds, 5)

dl_config = {"task_definition": task_definition,
             "metadata_file_id_column": 'filename',
             "metadata_file_column_sep": ',',
             "sequence_column": 'amino_acid',
             "sequence_counts_column": 'templates',
             "sample_n_sequences": args.sample_n_sequences,
             "sequence_counts_scaling_fn": no_sequence_count_scaling,
             "cross_validation_fold": 0,
             "with_test": False}

#
# Get dataset
#
# Get data loaders for training set and training-, validation-, and test-set in evaluation mode (=no random subsampling)
trainingset, trainingset_eval, validationset_eval, _ = make_dataloaders(
    metadata_file="/storage/ghadia/DeepRC2/deeprc/datasets/T1D/cohort1_wws.csv",
    repertoiresdata_path="/storage/ghadia/DeepRC2/deeprc/datasets/T1D/cohort_1",
    split_inds=split_inds,
    **dl_config
    # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
)
testset_eval = make_test_dataloader(metadata_file="/storage/ghadia/DeepRC2/deeprc/datasets/T1D/cohorts2_3.csv",
                                    repertoiresdata_path="/storage/ghadia/DeepRC2/deeprc/datasets/T1D/cohorts2_3",
                                    split_inds=list(range(857)),
                                    **dl_config
                                    )
#
# Create DeepRC Network
#
# Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=config['kernel_size'],
                                                  n_kernels=config['n_kernels'], n_layers=1)
# Create attention network
attention_network = AttentionNetwork(n_input_features=config['n_kernels'], n_layers=2,
                                     n_units=32)
# Create output network
output_network = OutputNetwork(n_input_features=config['n_kernels'],
                               n_output_features=task_definition.get_n_output_features(), n_layers=1, n_units=64)
# Combine networks to DeepRC network
model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
               attention_network=attention_network,
               output_network=output_network,
               consider_seq_counts=False, n_input_features=20, add_positional_information=True,
               sequence_reduction_fraction=0.1, reduction_mb_size=int(5e4),
               device=device).to(device=device)

#
# Train DeepRC model
#
val_loss = train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
                 trainingset_eval_dataloader=trainingset_eval, learning_rate=args.learning_rate,
                 early_stopping_target_id='ML_class',  # Get model that performs best for this task
                 validationset_eval_dataloader=validationset_eval, n_updates=args.n_updates,
                 evaluate_at=args.evaluate_at, l1_weight_decay=config['l1_l2'][0],
                 l2_weight_decay=config['l1_l2'][1],  # Regularization
                 device=device, results_directory=results_directory,
                 # Here our results and trained models will be stored
                 fold=0, idx=0)
# You can use "tensorboard --logdir [results_directory] --port=6060" and open "http://localhost:6060/" in your
# web-browser to view the progress

#
# Evaluate trained model on testset
#
scores = evaluate(model=model, dataloader=testset_eval, task_definition=task_definition, device=device)
print(f"Test scores:\n{scores}")
with open(f'{results_directory}/test.txt', 'a') as f:
    f.write(f"Test scores: {scores}")
