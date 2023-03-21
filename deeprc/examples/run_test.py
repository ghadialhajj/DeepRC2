# -*- coding: utf-8 -*-
"""
Example for training DeepRC with CNN sequence embedding in a single-task setting.
IMPORTANT: The used task is a small dummy-task with random data, so DeepRC should over-fit on the main task on the
training set.
Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
# todo: add key-query comparisons
import argparse

import numpy as np
import pandas as pd
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget, MulticlassTarget, RegressionTarget, Sequence_Target
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.training2 import train, evaluate
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
                    type=int, default=int(1e3))
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
parser.add_argument('--idx', help='Index of the run. Default: 0.',
                    type=int, default=0)

args = parser.parse_args()
# Set computation device
device_name = "cuda:0"  # + str(int((args.ideal + args.idx)%2))
with_test = False
device = torch.device(device_name)

seeds = [92, 9241, 5149, 41, 720, 813, 48525]

root_dir = "/home/ghadi/PycharmProjects/DeepRC2/deeprc"
# root_dir = "/storage/ghadia/DeepRC2/deeprc"
dataset_type = "trb_dataset"
base_results_dir = "/results/singletask_cnn/ideal"
strategies = ["TASTER"]  #"TASTER", "TASTE", "TE",  , "FG", "T-SAFTE"]
datasets = ["AIRR"]


for datastet in datasets:
    print(datastet)
    config = {"sequence_reduction_fraction": 0.1, "reduction_mb_size": int(5e3),
              "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), "prop": 0.02,
              "dataset": datastet, "pos_weight": 100, "Branch": "AdHoc1",
              "dataset_type": dataset_type}
    # Append current timestamp to results directory
    results_dir = os.path.join(f"{base_results_dir}_{config['dataset']}", config["timestamp"])

    #
    # Create Task definitions
    #
    # Assume we want to train on 1 main task as binary task at column 'binary_target_1' of our metadata file.
    task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
        BinaryTarget(  # Add binary classification task with sigmoid output function
            column_name='disease_status',  # Column name of task in metadata file
            true_class_value='CeD',  # Entries with value '+' will be positive class, others will be negative class
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
        metadata_file=f"{root_dir}/datasets/{dataset_type}/{config['dataset']}/metadata2.tsv",
        n_worker_processes=4,
        repertoiresdata_path=f"{root_dir}/datasets/{dataset_type}/{config['dataset']}/repertoires",
        metadata_file_id_column='ID',
        sequence_column='cdr3_aa',
        sequence_counts_column=None,
        sequence_pools_column='matched',
        sequence_labels_column='matched',
        sample_n_sequences=args.sample_n_sequences,
        sequence_counts_scaling_fn=no_sequence_count_scaling,
        with_test=with_test
        # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
    )

    stage1_dataloader, *_ = make_dataloaders(
        task_definition=task_definition,
        metadata_file=f"{root_dir}/datasets/{dataset_type}/Hypo{config['dataset']}/metadata2.tsv",
        n_worker_processes=4,
        batch_size=2,
        repertoiresdata_path=f"{root_dir}/datasets/{dataset_type}/Hypo{config['dataset']}/repertoires",
        metadata_file_id_column='ID',
        sequence_column='cdr3_aa',
        sequence_counts_column=None,
        sequence_pools_column='matched',
        sequence_labels_column='matched',
        sample_n_sequences=args.sample_n_sequences,
        sequence_counts_scaling_fn=no_sequence_count_scaling,
        with_test=with_test,
        all_sets=False
        # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
    )
    dl_dict = {"trainingset_eval": trainingset_eval, "validationset_eval": validationset_eval}
    if with_test:
        dl_dict.update({"testset_eval": testset_eval})

    logger = Logger(dataloaders=dl_dict)

    for strategy in strategies:
        print(strategy)
        if strategy == "TE":
            group = f"TE_n_up_{args.n_updates}_pw_{config['pos_weight']}"
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                           "plain_DeepRC": False})
        elif strategy == "TASTE":
            group = f"TASTE_n_up_{args.n_updates}_prop_{config['prop']}_pw_{config['pos_weight']}"
            config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": False})
        elif strategy == "TASTER":
            group = f"TASTER_n_up_{args.n_updates}_prop_{config['prop']}_pw_{config['pos_weight']}"
            config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": True})
        elif strategy == "T-SAFTE":
            group = f"T-SAFTE_n_up_{args.n_updates}_prop_{config['prop']}_pw_{config['pos_weight']}"
            config.update({"train_then_freeze": True, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False})
        elif strategy == "FG":
            group = f"FG_n_up_{args.n_updates}_pw_{config['pos_weight']}"
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": True,
                           "plain_DeepRC": True})
        elif strategy == "PDRC":
            group = f"PDRC_n_up_{args.n_updates}"
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                           "plain_DeepRC": True})
        else:
            raise "Invalid strategy"

        # Set random seed (will still be non-deterministic due to multiprocessing but weight init will be the same)
        torch.manual_seed(seeds[args.idx])
        np.random.seed(seeds[args.idx])

        run = wandb.init(project="HUNTTEST", group=group, reinit=True)  # , tags=config["tag"])
        run.name = f"results_idx_{str(args.idx)}"  # config["run"] +   # += f"_ideal_{config['ideal']}"
        # DeepRC_PlainW_StanData, Explore_wFPs

        wandb.config.update(args)
        wandb.config.update(config)

        print("Dataloaders with lengths: ",
              ",\n".join([f"{str(name)}: {len(loader)}" for name, loader in dl_dict.items()]))
        print("Stage1: ", len(stage1_dataloader))

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
                                       n_output_features=task_definition.get_n_output_features(), n_layers=1,
                                       n_units=32)
        # Combine networks to DeepRC network
        model = DeepRC(max_seq_len=37, sequence_embedding_network=sequence_embedding_network,
                       attention_network=attention_network,
                       output_network=output_network,
                       consider_seq_counts=False, n_input_features=20, add_positional_information=True,
                       sequence_reduction_fraction=config["sequence_reduction_fraction"],
                       reduction_mb_size=config["reduction_mb_size"], device=device,
                       forced_attention=config["forced_attention"]).to(device=device)
        #
        # Train DeepRC model
        #
        print("training")
        train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
              trainingset_eval_dataloader=trainingset_eval, learning_rate=args.learning_rate,
              early_stopping_target_id='disease_status',  # Get model that performs best for this task
              validationset_eval_dataloader=validationset_eval, stage1_dataloader=stage1_dataloader,
              logger=logger, n_updates=args.n_updates,
              evaluate_at=args.evaluate_at, device=device, results_directory=f"{root_dir}{results_dir}",
              prop=config["prop"],
              log_training_stats_at=args.log_training_stats_at,  # Here our results and trained models will be stored
              train_then_freeze=config["train_then_freeze"], staged_training=config["staged_training"],
              plain_DeepRC=config["plain_DeepRC"], log=False)

        # logger.log_stats(model=model, device=device, step=args.n_updates)

        #
        # Evaluate trained model on testset
        #
        if with_test:
            scores, sequence_scores = evaluate(model=model, dataloader=testset_eval, task_definition=task_definition,
                                               device=device)
            wandb.run.summary.update(scores["binary_target_1"])
            wandb.run.summary.update(sequence_scores["sequence_class"])
            print(f"Test scores:\n{scores}")
        wandb.finish()
