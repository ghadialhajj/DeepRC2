# -*- coding: utf-8 -*-
"""
Example for training DeepRC with CNN sequence embedding in a single-task setting.
IMPORTANT: The used task is a small dummy-task with random data, so DeepRC should over-fit on the main task on the
training set.
Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import argparse
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget, MulticlassTarget, RegressionTarget, Sequence_Target
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.training import train, evaluate
import wandb
import os
import datetime
from deeprc.utils import Logger

#
# Get command line arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--n_updates', help='Number of updates to train for. Recommended: int(1e5). Default: int(1e3)',
                    type=int, default=int(20e3))
# type=int, default=int(20))
parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping. '
                                          'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(5e2))
# type=int, default=int(4))
parser.add_argument('--log_training_stats_at', help='Log training stats every `log_training_stats_at` updates. '
                                                    'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(1e3))
# type=int, default=int(2))
parser.add_argument('--kernel_size', help='Size of 1D-CNN kernels (=how many sequence characters a CNN kernel spans).'
                                          'Default: 9',
                    type=int, default=5)
parser.add_argument('--n_kernels', help='Number of kernels in the 1D-CNN. This is an important hyper-parameter. '
                                        'Default: 32',
                    type=int, default=32)
parser.add_argument('--sample_n_sequences', help='Number of instances to reduce repertoires to during training via'
                                                 'random dropout. This should be less than the number of instances per '
                                                 'repertoire. Only applied during training, not for evaluation. '
                                                 'Default: int(1e4)',
                    type=int, default=int(1e4))
parser.add_argument('--learning_rate', help='Learning rate of DeepRC using Adam optimizer. Default: 1e-4',
                    type=float, default=1e-4)
# parser.add_argument('--device', help='Device to use for NN computations, as passed to `torch.device()`. '
#                                      'Default: "cuda:0".',
#                                       type=str, default="cuda:1")
# parser.add_argument('--rnd_seed', help='Random seed to use for PyTorch and NumPy. Results will still be '
#                                       'non-deterministic due to multiprocessing but weight initialization will be the'
#                                        ' same). Default: 0.',
#                     type=int, default=0)
parser.add_argument('--idx', help='Index of the run. Default: 0.',
                    type=int, default=1)

# parser.add_argument('--ideal', help='1 (True) means "use ideal attention values, 0 (False), otherwise. Default: 0.',
#                     type=int, default=0)

args = parser.parse_args()
# Set computation device
device_name = "cuda:0"  # + str(int((args.ideal + args.idx)%2))
device = torch.device(device_name)
# Set random seed (will still be non-deterministic due to multiprocessing but weight initialization will be the same)
# torch.manual_seed(args.rnd_seed)
# np.random.seed(args.rnd_seed)

# root_dir = "/home/ghadi/PycharmProjects/DeepRC2/deeprc"
root_dir = "/storage/ghadia/DeepRC2/deeprc"
# root_dir = "/itf-fi-ml/shared/users/ghadia/deeprc"
# root_dir = "/fp/homes01/u01/ec-ghadia/DeepRC2/deeprc"
# root_dir = "/cluster/work/projects/ec35/ec-ghadia/"
base_results_dir = "/results/singletask_cnn/ideal"

config = {"sequence_reduction_fraction": 0.1, "reduction_mb_size": int(5e3),
          "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), "prop": 0.77,
          "dataset": "n_600_wr_0.100%_po_80%", "pos_weight": 1, "train_then_freeze": True,
          "tag": ["AdHoc1.3.1"], "staged_training": True} #n_600_op_1_po_0.100%_pu_0
            # "dataset": "n_600_wr_0.100%_po_80%", "pos_weight": 100, "train_then_freeze": False}
# "dataset": "n_20_op_1_po_0.100%25_pu_0", "pos_weight": 100}  # , "run": "run_2"}

# Append current timestamp to results directory
results_dir = os.path.join(f"{base_results_dir}_{config['dataset']}", config["timestamp"])

if config["staged_training"]:
    group = f"TASTE_n_up_{args.n_updates}_pw_{config['pos_weight']}"
    if config["train_then_freeze"]:
        group = f"TTF_n_up_{args.n_updates}_prop_{config['prop']}_pw_{config['pos_weight']}"
else:
    group = f"TE_n_up_{args.n_updates}_pw_{config['pos_weight']}"

run = wandb.init(project="DeepRC_AdHoc1.3", group=group, tags=config["tag"])
run.name = f"results_idx_{str(args.idx)}"  # config["run"] +   # += f"_ideal_{config['ideal']}"

wandb.config.update(args)
wandb.config.update(config)
#
# Create Task definitions
#
# Assume we want to train on 1 main task as binary task at column 'binary_target_1' of our metadata file.
task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
    BinaryTarget(  # Add binary classification task with sigmoid output function
        column_name='binary_target_1',  # Column name of task in metadata file
        true_class_value='+',  # Entries with value '+' will be positive class, others will be negative class
        pos_weight=1.,  # We can up- or down-weight the positive class if the classes are imbalanced
    ),
    Sequence_Target(pos_weight=config["pos_weight"]),
]).to(device=device)
#
# Get dataset
#
# Get data loaders for training set and training-, validation-, and test-set in evaluation mode (=no random subsampling)
trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
    task_definition=task_definition,
    metadata_file=f"{root_dir}/datasets/{config['dataset']}/metadata.tsv",
    n_worker_processes=4,
    repertoiresdata_path=f"{root_dir}/datasets/{config['dataset']}/repertoires",
    metadata_file_id_column='ID',
    sequence_column='amino_acid',
    sequence_counts_column='templates',
    sequence_labels_column='label',
    sample_n_sequences=args.sample_n_sequences,
    sequence_counts_scaling_fn=no_sequence_count_scaling

    # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
)
dl_dict = {"trainingset": trainingset, "trainingset_eval": trainingset_eval, "validationset_eval": validationset_eval,
           "testset_eval": testset_eval}
#
# Create DeepRC Network
#
# Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=args.kernel_size,
                                                  n_kernels=args.n_kernels, n_layers=1)
# Create attention network
attention_network = AttentionNetwork(n_input_features=args.n_kernels, n_layers=2, n_units=32)
# Create output network
output_network = OutputNetwork(n_input_features=args.n_kernels,
                               n_output_features=task_definition.get_n_output_features(), n_layers=1, n_units=32)
# Combine networks to DeepRC network
model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
               attention_network=attention_network,
               output_network=output_network,
               consider_seq_counts=False, n_input_features=20, add_positional_information=True,
               sequence_reduction_fraction=config["sequence_reduction_fraction"],
               reduction_mb_size=config["reduction_mb_size"], device=device).to(device=device)
#
# Train DeepRC model
#
logger = Logger(dataloaders=dl_dict)
logger.log_stats(model=model, device=device, step=0)

train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
      trainingset_eval_dataloader=trainingset_eval, learning_rate=args.learning_rate,
      early_stopping_target_id='binary_target_1',  # Get model that performs best for this task
      validationset_eval_dataloader=validationset_eval, logger=logger, n_updates=args.n_updates,
      evaluate_at=args.evaluate_at, device=device, results_directory=f"{root_dir}{results_dir}", prop=config["prop"],
      log_training_stats_at=args.log_training_stats_at,  # Here our results and trained models will be stored
      train_then_freeze=config["train_then_freeze"], staged_training=config["staged_training"])

logger.log_stats(model=model, device=device, step=args.n_updates)

#
# Evaluate trained model on testset
#

scores, sequence_scores = evaluate(model=model, dataloader=testset_eval, task_definition=task_definition, device=device)
wandb.run.summary.update(scores["binary_target_1"])
wandb.run.summary.update(sequence_scores["sequence_class"])
print(f"Test scores:\n{scores}")
wandb.finish()
