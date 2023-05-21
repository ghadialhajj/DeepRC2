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
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling, log_sequence_count_scaling
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
                    type=int, default=int(6e4))
# type=int, default=int(100))
parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping. '
                                          'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(5e3))
# type=int, default=int(10))
parser.add_argument('--log_training_stats_at', help='Log training stats every `log_training_stats_at` updates. '
                                                    'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(2e3))
# type=int, default=int(25))
parser.add_argument('--kernel_size', help='Size of 1D-CNN kernels (=how many sequence characters a CNN kernel spans).'
                                          'Default: 9',
                    type=int, default=9)
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
                    type=int, default=2)

args = parser.parse_args()
# Set computation device
device_name = "cuda:0"  # + str(int((args.ideal + args.idx)%2))
with_test = False
device = torch.device(device_name)

seeds = [92, 9241, 5149, 41, 720, 813, 485, 85, 74]

# root_dir = "/home/ghadi/PycharmProjects/DeepRC2/deeprc"
root_dir = "/storage/ghadia/DeepRC2/deeprc"
# root_dir = "/itf-fi-ml/home/ghadia/DeepRC2/deeprc"
dataset_type = "emerson"
# root_dir = "/itf-fi-ml/shared/users/ghadia/deeprc"
# root_dir = "/fp/homes01/u01/ec-ghadia/DeepRC2/deeprc"
# root_dir = "/cluster/work/projects/ec35/ec-ghadia/"
base_results_dir = "/results/singletask_cnn/ideal"
# , "tag": ["AdHoc1.3.1"]}
# n_20_op_1_po_0.100%25_pu_0
strategies = ["PDRC", "TASTE", "FG", "TE"]  # , "TASTER", "T-SAFTE"]
datasets = ["oldAIRR"]  # "n_600_wr_0.050%_po_100%",  "n_600_wr_0.100%_po_100%",

print("defined variables")


# TE: Train Everthing
# TASTE: Train Attention and Sequence Embedding first, then Train Everything
# T-SAFTE: Train Sequence Embedding and Attention networks first, then Freeze the first part and Train Everything
# FG: Forced Guidance: attention is provided rather than learned (regardless of label trueness)

def generate_indcs(num_reps: int = 683, num_test: int = 120, num_splits: int = 5):
    original_list = list(range(num_reps - num_test))
    total_elements = len(original_list)
    elements_per_list = total_elements // num_splits  # Integer division to get the base number of elements per list
    remainder = total_elements % num_splits  # Get the remainder elements
    lists = [
        original_list[i * elements_per_list + min(i, remainder):(i + 1) * elements_per_list + min(i + 1, remainder)] for
        i in range(num_splits)]
    test_indcs = list(range(num_reps - num_test, num_reps))
    lists.append(test_indcs)
    return lists


if __name__ == '__main__':
    for datastet in datasets:
        print(datastet)
        config = {"sequence_reduction_fraction": 0.1, "reduction_mb_size": int(5e3),
                  "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), "prop": 0.3,
                  "dataset": datastet, "pos_weight_seq": 100, "pos_weight_rep": 1., "Branch": "Emerson",
                  "dataset_type": dataset_type}
        # Append current timestamp to results directory
        results_dir = os.path.join(f"{base_results_dir}_{config['dataset']}", config["timestamp"])

        #
        # Create Task definitions
        #
        # Assume we want to train on 1 main task as binary task at column 'binary_target_1' of our metadata file.
        task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
            BinaryTarget(  # Add binary classification task with sigmoid output function
                column_name='CMV',  # Column name of task in metadata file
                true_class_value='True',  # Entries with value 'True' will be positive class, others will be negative class
                pos_weight=config["pos_weight_rep"],
                # We can up- or down-weight the positive class if the classes are imbalanced
            ),
            Sequence_Target(pos_weight=config["pos_weight_seq"]),
        ]).to(device=device)
        #
        # Get dataset
        #
        split_inds = generate_indcs(683, 120, 5)

        # Get data loaders for training set and training-, validation-, and test-set in evaluation mode (=no random subsampling)
        trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
            task_definition=task_definition,
            metadata_file=f"{root_dir}/datasets/{dataset_type}/{config['dataset']}/metadata.csv",
            metadata_file_column_sep=",",
            n_worker_processes=8,
            repertoiresdata_path=f"{root_dir}/datasets/{dataset_type}/{config['dataset']}/repertoires",
            metadata_file_id_column='filename',
            sequence_column='cdr3_aa',
            sequence_counts_column="duplicate_count",
            sequence_pools_column='matched',
            sequence_labels_column='matched',
            sample_n_sequences=args.sample_n_sequences,
            # sequence_counts_scaling_fn=no_sequence_count_scaling,
            sequence_counts_scaling_fn=log_sequence_count_scaling,
            with_test=with_test,
            split_inds=split_inds,
            cross_validation_fold=5
            # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
        )
        dl_dict = {"trainingset_eval": trainingset_eval, "validationset_eval": validationset_eval}
        if with_test:
            dl_dict.update({"testset_eval": testset_eval})

        logger = Logger(dataloaders=dl_dict, with_FPs=False)

        for strategy in strategies:
            print(strategy)
            if strategy == "TE":
                group = f"TE_n_up_{args.n_updates}_pw_{config['pos_weight_seq']}"
                config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                               "plain_DeepRC": False})
            elif strategy == "TASTE":
                group = f"TASTE_n_up_{args.n_updates}_prop_{config['prop']}_pw_{config['pos_weight_seq']}"
                config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                               "plain_DeepRC": False, "rep_loss_only": False})
            elif strategy == "TASTER":
                group = f"TASTER_n_up_{args.n_updates}_prop_{config['prop']}_pw_{config['pos_weight_seq']}"
                config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                               "plain_DeepRC": False, "rep_loss_only": True})
            elif strategy == "T-SAFTE":
                group = f"T-SAFTE_n_up_{args.n_updates}_prop_{config['prop']}_pw_{config['pos_weight_seq']}"
                config.update({"train_then_freeze": True, "staged_training": True, "forced_attention": False,
                               "plain_DeepRC": False})
            elif strategy == "FG":
                group = f"FG_n_up_{args.n_updates}_pw_{config['pos_weight_seq']}"
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

            run = wandb.init(project="oldEmerson_consider_counts_log", group=f"{group}",
                             reinit=True)  # , tags=config["tag"])
            run.name = f"results_idx_{str(args.idx)}"  # config["run"] +   # += f"_ideal_{config['ideal']}"
            # DeepRC_PlainW_StanData, Explore_wFPs
            # if args.idx == 0:
            #     df_dict = get_sample_reps()
            #     log_reps(df_dict)

            wandb.config.update(args)
            wandb.config.update(config)

            print("Dataloaders with lengths: ",
                  ", ".join([f"{str(name)}: {len(loader)}" for name, loader in dl_dict.items()]))

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
            model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
                           attention_network=attention_network,
                           output_network=output_network,
                           consider_seq_counts=True, n_input_features=20, add_positional_information=True,
                           sequence_reduction_fraction=config["sequence_reduction_fraction"],
                           reduction_mb_size=config["reduction_mb_size"], device=device,
                           forced_attention=config["forced_attention"]).to(device=device)
            #
            # Train DeepRC model
            #
            print("training")
            train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
                  trainingset_eval_dataloader=trainingset_eval, learning_rate=args.learning_rate,
                  early_stopping_target_id='CMV',  # Get model that performs best for this task
                  validationset_eval_dataloader=validationset_eval, logger=logger, n_updates=args.n_updates,
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
                wandb.run.summary.update(scores["CMV"])
                wandb.run.summary.update(sequence_scores["sequence_class"])
                print(f"Test scores:\n{scores}")
            wandb.finish()
