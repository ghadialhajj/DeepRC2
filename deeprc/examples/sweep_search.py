import argparse
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
import dill as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--device_id', help='Index of the device. Default: 0.',
                    type=int, default=0)
parser.add_argument('--run_id', help='Index of the run. Default: 0.',
                    type=int, default=2)

args = parser.parse_args()
device_name = f"cuda:0"
with_test = False
device = torch.device(device_name)

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# root_dir = "/home/ghadi/PycharmProjects/DeepRC2/deeprc"
root_dir = "/storage/ghadia/DeepRC2/deeprc"
dataset_type = "emerson_linz"
base_results_dir = "/results/singletask_cnn/ideal"
strategies = ["TE"]  # "PDRC",  "TASTE", "FG", "TASTER", , "T-SAFTE"]
dataset = "AIRR"
# dataset = "test_data"

# 2: Define the search space
sweep_configuration = {
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'max_auc_score'},
    'parameters':
        {
            'kernel_size': {'values': [5, 7, 9]},
            'n_kernels': {'values': [8, 16, 32, 64, 128, 256]},
        }
}

config = {"sequence_reduction_fraction": 0.1, "reduction_mb_size": int(5e3),
          "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), "prop": 0.3,
          "dataset": dataset, "pos_weight_seq": 100, "pos_weight_rep": 1., "Branch": "Emerson",
          "dataset_type": dataset_type, "log_training_stats_at": int(2e3), "sample_n_sequences": int(1e4),
          "learning_rate": 1e-4, "n_updates": int(3e4), "evaluate_at": int(1e3)}

# Append current timestamp to results directory
results_dir = os.path.join(f"{base_results_dir}_{config['dataset']}", config["timestamp"])


def get_indices(num_lists: int, with_test: bool = False, n_samples: int = 50):
    split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_separate_test.pkl"
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)["inds"]
    print(len(split_inds))
    assert num_lists < len(split_inds)
    indices_lists = split_inds[:num_lists]
    if with_test:
        indices_lists.append(split_inds[-1])
    # else:
    #     indices_lists.append([])
    indices_lists = [i[:n_samples] for i in indices_lists]
    return indices_lists


# split_inds = get_indices(4, False, None)
split_inds = get_indices(2, False, 50)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Günter's")

group = f"TE_n_up_{config['n_updates']}"
config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
               "plain_DeepRC": False})


# group = f"TASTE_n_up_{config['n_updates']}"
# config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
#                "plain_DeepRC": False})
# elif strategy == "PDRC":
# group = f"PDRC_n_up_{config['n_updates']}"
# config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
#                "plain_DeepRC": True})

# print("Dataloaders with lengths: ",
#       ", ".join([f"{str(name)}: {len(loader)}" for name, loader in dl_dict.items()]))


def main():
    run = wandb.init()
    run.name = f"results_idx_{str(args.run_id)}"
    wandb.config.update(config)
    wandb.config.update(args)
    wandb.config.update(sweep_configuration)
    torch.manual_seed(seeds[args.run_id])
    np.random.seed(seeds[args.run_id])

    task_definition = TaskDefinition(targets=[
        BinaryTarget(column_name='CMV', true_class_value='+', pos_weight=config["pos_weight_rep"]),
        Sequence_Target(pos_weight=config["pos_weight_seq"])]).to(device=device)

    # Get data loaders for training set and training-, validation-, and test-set in evaluation mode
    trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
        task_definition=task_definition,
        metadata_file=f"{root_dir}/datasets/{dataset_type}/{config['dataset']}/metadata.csv",
        metadata_file_column_sep=",", n_worker_processes=6,
        repertoiresdata_path=f"{root_dir}/datasets/{dataset_type}/{config['dataset']}/repertoires",
        metadata_file_id_column='filename', sequence_column='cdr3_aa', sequence_counts_column="duplicate_count",
        sequence_labels_column='matched', sample_n_sequences=config["sample_n_sequences"],
        sequence_counts_scaling_fn=log_sequence_count_scaling, with_test=with_test, split_inds=split_inds,
        cross_validation_fold=1)

    dl_dict = {"trainingset_eval": trainingset_eval, "validationset_eval": validationset_eval}
    if with_test:
        dl_dict.update({"testset_eval": testset_eval})

    logger = Logger(dataloaders=dl_dict, with_FPs=False)

    sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=wandb.config.kernel_size,
                                                      n_kernels=wandb.config.n_kernels, n_layers=1)
    attention_network = AttentionNetwork(n_input_features=wandb.config.n_kernels, n_layers=2, n_units=32)
    output_network = OutputNetwork(n_input_features=wandb.config.n_kernels,
                                   n_output_features=task_definition.get_n_output_features(), n_layers=1, n_units=32)

    model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
                   attention_network=attention_network, output_network=output_network, consider_seq_counts=False,
                   n_input_features=20, add_positional_information=True,
                   sequence_reduction_fraction=config["sequence_reduction_fraction"],
                   reduction_mb_size=config["reduction_mb_size"], device=device,
                   forced_attention=config["forced_attention"]).to(device=device)

    max_auc = train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
                    trainingset_eval_dataloader=trainingset_eval, learning_rate=config["learning_rate"],
                    early_stopping_target_id='CMV', validationset_eval_dataloader=validationset_eval, logger=logger,
                    n_updates=config["n_updates"], evaluate_at=config["evaluate_at"], device=device,
                    results_directory=f"{root_dir}{results_dir}", prop=config["prop"],
                    log_training_stats_at=config["log_training_stats_at"],
                    train_then_freeze=config["train_then_freeze"],
                    staged_training=config["staged_training"], plain_DeepRC=config["plain_DeepRC"], log=False)

    # logger.log_stats(model=model, device=device, step=config.n_updates)
    wandb.log({'max_auc_score': max_auc})
    # if with_test:
    #     scores, sequence_scores = evaluate(model=model, dataloader=testset_eval, task_definition=task_definition,
    #                                        device=device)
    #     wandb.run.summary.update(scores["CMV"])
    #     wandb.run.summary.update(sequence_scores["sequence_class"])
    #     print(f"Test scores:\n{scores}")


# main()

# # Start sweep job.
wandb.agent(sweep_id, function=main)
# print("asdfs")
# for idx in range(3):
#     main(idx)