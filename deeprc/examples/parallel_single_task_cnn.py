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
import dill as pkl
import numpy as np
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget, Sequence_Target
from deeprc.dataset_readers import make_dataloaders, log_sequence_count_scaling
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.training import train, evaluate
import wandb
import os
import datetime
from deeprc.utils import Logger
from itertools import product
from deeprc.training import ESException

#
# Get command line arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--n_updates', help='Number of updates to train for. Recommended: int(1e5). Default: int(1e3)',
                    type=int, default=int(5e3))
# type=int, default=int(100))
parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping. '
                                          'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(100))
# type=int, default=int(10))
parser.add_argument('--log_training_stats_at', help='Log training stats every `log_training_stats_at` updates. '
                                                    'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(100))
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
device_name = "cuda:0"
with_test = True
device = torch.device(device_name)

seeds = [92, 9241, 5149, 41, 720, 813, 485, 85, 74]
seed_idx = 0
# root_dir = "/home/ghadi/PycharmProjects/DeepRC2/deeprc"
root_dir = "/storage/ghadia/DeepRC2/deeprc"
base_results_dir = "/results/singletask_cnn/ideal"
strategies = ["TE"]  # , "PDRC", "TASTE" , "FG", "TASTER", "T-SAFTE"]


def get_split_inds(n_folds, cohort, n_tr, n_v):
    split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_separate_test_correct.pkl"
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)["inds"]
        if cohort == 2:
            split_inds = split_inds[-1]
        elif cohort == 1:
            split_inds = split_inds[:-1]
            split_inds = [a for b in split_inds for a in b]
    np.random.seed(0)
    np.random.shuffle(split_inds)
    # split_inds = [split_inds[i * int(len(split_inds) / n_folds): (i + 1) * int(len(split_inds) / n_folds)] for i in
    #               range(n_folds)]
    train_split_inds = split_inds[:n_tr]
    val_split_inds = split_inds[n_tr: n_tr + n_v]
    return [[], train_split_inds, val_split_inds]


if __name__ == '__main__':
    loss_config = {"min_cnt": 1, "normalize": False, "add_in_loss": True}

    config = {"sequence_reduction_fraction": 0.1, "reduction_mb_size": int(5e3),
              "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), "prop": 0.1,
              "dataset": "AIRR", "pos_weight_seq": 100, "pos_weight_rep": 1., "Branch": "Emerson",
              "dataset_type": "emerson_linz"}
    # Append current timestamp to results directory
    results_dir = os.path.join(f"{base_results_dir}_{config['dataset']}", config["timestamp"])
    n_samples = 15
    split_inds = get_split_inds(0, 1, n_samples, 160)
    #
    # Create Task definitions
    #
    strategy = "TE"
    # strategy = "TASTE"
    # strategy = "TASTER"
    # strategy = "PDRC"
    if strategy == "PDRC":
        fpa, fps, wsw, wsi = False, False, False, False
    else:
        fpa, fps, wsw, wsi = True, True, True, False
    task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
        BinaryTarget(  # Add binary classification task with sigmoid output function
            column_name='CMV',  # Column name of task in metadata file
            true_class_value='+',
            # Entries with value 'True' will be positive class, others will be negative class
        ),
        Sequence_Target(pos_weight=config["pos_weight_seq"], weigh_seq_by_weight=wsw, weigh_pos_by_inverse=wsi,
                        normalize=loss_config["normalize"], add_in_loss=loss_config["add_in_loss"],
                        device=device), ]).to(device=device)
    #
    # Get dataset
    #
    # Get data loaders for training set and training-, validation-, and test-set in evaluation mode (=no random subsampling)
    trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
        task_definition=task_definition,
        metadata_file=f"{root_dir}/datasets/{config['dataset_type']}/{config['dataset']}/metadata.csv",
        metadata_file_column_sep=",",
        n_worker_processes=1,
        repertoiresdata_path=f"{root_dir}/datasets/{config['dataset_type']}/{config['dataset']}/repertoires",
        metadata_file_id_column='filename',
        sequence_column='cdr3_aa',
        sequence_counts_column="duplicate_count",
        sequence_labels_column='matched',
        sample_n_sequences=args.sample_n_sequences,
        sequence_counts_scaling_fn=log_sequence_count_scaling,
        with_test=with_test,
        split_inds=split_inds,
        force_pos_in_subsampling=fps,
        min_count=loss_config["min_cnt"]
        # cross_validation_fold=0,
    )

    dl_dict = {"trainingset_eval": trainingset_eval, "validationset_eval": validationset_eval}
    if with_test:
        dl_dict.update({"testset_eval": testset_eval})

    logger = Logger(dataloaders=dl_dict, with_FPs=False)

    n_kernels_list = [8, 16, 32, 64, 128, 256]
    kernel_size_list = [[5, 7, 9][args.idx]]
    # for n_kernels, kernel_size in product(n_kernels_list, kernel_size_list):

    max_aucs = []
    seeds_list = [1, 2]
    for seed in seeds_list:
        # for l1w in [0.1, 0.01, 0.01]:
        #     seed = 1
        n_kernels, kernel_size = 32, 9
        if strategy == "TE":
            group = f"TE_n_up_{args.n_updates}_pw_{config['pos_weight_seq']}"
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": False})
        elif strategy == "TASTE":
            group = f"TASTE_n_up_{args.n_updates}_prop_{config['prop']}_pw_{config['pos_weight_seq']}"
            config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": False})
        elif strategy == "TASTER":
            group = f"TASTER_n_up_{args.n_updates}_prop_{config['prop']}_pw_{config['pos_weight_seq']}"
            config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": True})
        elif strategy == "PDRC":
            group = f"PDRC_n_up_{args.n_updates}"
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                           "plain_DeepRC": True, "rep_loss_only": False})
        else:
            raise "Invalid strategy"
        try:
            # Set random seed (will still be non-deterministic due to multiprocessing but weight init will be the same)
            torch.manual_seed(seed)
            np.random.seed(seed)

            run = wandb.init(project="Correct Indices", group=f"czc_{group}",
                             reinit=True)  # , tags=config["tag"])
            run.name = f"results_idx_{str(seed)}"  # config["run"] +   # += f"_ideal_{config['ideal']}"

            wandb.config.update(args)
            wandb.config.update(config)
            wandb.config.update(loss_config)
            wandb.config.update({"fpa": fpa, "fps": fps, "wsw": wsw, "wsi": wsi, "n_samples": n_samples})
            wandb.config.update({"n_kernels": n_kernels, "kernel_size": kernel_size})

            print("Dataloaders with lengths: ",
                  ", ".join([f"{str(name)}: {len(loader)}" for name, loader in dl_dict.items()]))

            #
            # Create DeepRC Network
            #
            # Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
            sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=kernel_size,
                                                              n_kernels=n_kernels, n_layers=1)
            # Create attention network
            attention_network = AttentionNetwork(n_input_features=n_kernels, n_layers=2, n_units=32)
            # Create output network
            output_network = OutputNetwork(n_input_features=n_kernels,
                                           n_output_features=task_definition.get_n_output_features(), n_layers=1,
                                           n_units=32)
            # Combine networks to DeepRC network
            model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
                           attention_network=attention_network,
                           output_network=output_network,
                           consider_seq_counts=True, consider_seq_counts_after_cnn=False, n_input_features=20,
                           add_positional_information=True,
                           sequence_reduction_fraction=config["sequence_reduction_fraction"],
                           reduction_mb_size=config["reduction_mb_size"], device=device,
                           forced_attention=config["forced_attention"], force_pos_in_attention=fpa
                           ).to(device=device)
            #
            # Train DeepRC model
            #
            max_auc = train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
                            trainingset_eval_dataloader=trainingset_eval, learning_rate=args.learning_rate,
                            early_stopping_target_id='CMV',  # Get model that performs best for this task
                            validationset_eval_dataloader=validationset_eval, logger=logger, n_updates=args.n_updates,
                            evaluate_at=args.evaluate_at, device=device, results_directory=f"{root_dir}{results_dir}",
                            prop=config["prop"],  # l1_weight_decay=l1w,
                            log_training_stats_at=args.log_training_stats_at,
                            # Here our results and trained models will be stored
                            train_then_freeze=config["train_then_freeze"], staged_training=config["staged_training"],
                            plain_DeepRC=config["plain_DeepRC"], log=True, rep_loss_only=config["rep_loss_only"])

            # logger.log_stats(model=model, device=device, step=args.n_updates)
            max_aucs.append(max_auc)
            #
            # Evaluate trained model on testset
            #
            # if with_test:
            #     scores, sequence_scores = evaluate(model=model, dataloader=testset_eval,
            #                                        task_definition=task_definition,
            #                                        device=device)
            #     wandb.run.summary.update(scores["CMV"])
            #     wandb.run.summary.update(sequence_scores["sequence_class"])
            #     print(f"Test scores:\n{scores}")
            wandb.finish()
        except ValueError as ve:
            print("Error: ", ve)
        except ESException as ese:
            print("ESE")
    print("max_aucs: ", max_aucs)
