import argparse

import numpy as np
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget, MulticlassTarget, RegressionTarget, Sequence_Target
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling
from deeprc.architectures import ShallowlRC, SequenceEmbeddingCNN, OutputNetwork
from deeprc.training import train, evaluate
import wandb
import os
import datetime
from deeprc.utils import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--n_updates', help='Number of updates to train for. Recommended: int(1e5). Default: int(1e3)',
                    type=int, default=int(15e3))

parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping. '
                                          'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(5e2))

parser.add_argument('--log_training_stats_at', help='Log training stats every `log_training_stats_at` updates. '
                                                    'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(1e3))

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
                    type=int, default=1)

args = parser.parse_args()

device_name = "cuda:1"
device = torch.device(device_name)

seeds = [92, 9241, 5149, 41, 720, 813, 48525]

root_dir = "/storage/ghadia/DeepRC2/deeprc"
dataset_type = "all_observed10"

base_results_dir = "/results/singletask_cnn/ideal"

strategies = ["PDRC"]

datasets = ["n_600_wr_1.000%_po_100%"]
print("defined variables")

for datastet in datasets:
    print(datastet)
    config = {"sequence_reduction_fraction": 0.1, "reduction_mb_size": int(5e3),
              "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), "prop": 0.2,
              "dataset": datastet, "pos_weight": 100, "Branch": "AdHoc1",
              "dataset_type": dataset_type}

    results_dir = os.path.join(f"{base_results_dir}_{config['dataset']}", config["timestamp"])

    task_definition = TaskDefinition(targets=[
        BinaryTarget(
            column_name='binary_target_1',
            true_class_value='+',
            pos_weight=1.,
        ),
        Sequence_Target(pos_weight=config["pos_weight"]),
    ]).to(device=device)

    trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
        task_definition=task_definition,
        metadata_file=f"{root_dir}/datasets/{dataset_type}/{config['dataset']}/metadata.tsv",
        n_worker_processes=4,
        repertoiresdata_path=f"{root_dir}/datasets/{dataset_type}/{config['dataset']}/repertoires",
        metadata_file_id_column='ID',
        sequence_column='amino_acid',
        sequence_counts_column='templates',
        sequence_labels_column='label',
        sample_n_sequences=args.sample_n_sequences,
        sequence_counts_scaling_fn=no_sequence_count_scaling

    )
    dl_dict = {"trainingset_eval": trainingset_eval, "validationset_eval": validationset_eval,
               "testset_eval": testset_eval}
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

        torch.manual_seed(seeds[args.idx])
        np.random.seed(seeds[args.idx])

        run = wandb.init(project="Knut", group=group, reinit=True)
        run.name = f"results_idx_{str(args.idx)}"

        wandb.config.update(args)
        wandb.config.update(config)

        sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=args.kernel_size,
                                                          n_kernels=args.n_kernels, n_layers=1)

        output_network = OutputNetwork(n_input_features=args.n_kernels,
                                       n_output_features=task_definition.get_n_output_features(), n_layers=1,
                                       n_units=32)

        model = ShallowlRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
                           output_network=output_network,
                           consider_seq_counts=False, n_input_features=20, add_positional_information=True,
                           sequence_reduction_fraction=config["sequence_reduction_fraction"],
                           reduction_mb_size=config["reduction_mb_size"], device=device,
                           forced_attention=config["forced_attention"]).to(device=device)

        print("training")
        train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
              trainingset_eval_dataloader=trainingset_eval, learning_rate=args.learning_rate,
              early_stopping_target_id='binary_target_1',
              validationset_eval_dataloader=validationset_eval, logger=logger, n_updates=args.n_updates,
              evaluate_at=args.evaluate_at, device=device, results_directory=f"{root_dir}{results_dir}",
              prop=config["prop"],
              log_training_stats_at=args.log_training_stats_at,
              train_then_freeze=config["train_then_freeze"], staged_training=config["staged_training"],
              plain_DeepRC=config["plain_DeepRC"], log=False)

        scores, sequence_scores = evaluate(model=model, dataloader=testset_eval, task_definition=task_definition,
                                           device=device)
        wandb.run.summary.update(scores["binary_target_1"])
        wandb.run.summary.update(sequence_scores["sequence_class"])
        print(f"Test scores:\n{scores}")
        wandb.finish()