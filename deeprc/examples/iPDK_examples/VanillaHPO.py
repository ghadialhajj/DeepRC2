import copy
import datetime
import os
import dill as pkl
import numpy as np
import wandb

from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling, create_hdf5_file
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.task_definitions import TaskDefinition, BinaryTarget, Sequence_Target
from deeprc.training import train, evaluate
import torch
from deeprc.utils import Logger, eval_on_test

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n_witnesses', help='Name of the strategy. Default: int(1e3)',
                    type=int, default=50)
parser.add_argument('--hp_name', help='Name of the HP. Default: int(1e3)',
                    type=str, default="n_heads")
parser.add_argument('--n_updates', help='Name of the strategy. Default: int(1e3)',
                    type=int, default=int(1e5))
parser.add_argument('--device', help='GPU ID. Default: 0',
                    type=int, default=0)
args = parser.parse_args()

device_name = f"cuda:{args.device}"
device = torch.device(device_name)

root_dir = "/storage/ghadia/DeepRC2/deeprc"
# root_dir = "/itf-fi-ml/home/ghadia/DeepRC2/deeprc"
base_results_dir = "/results/singletask_cnn/ideal"

HPs = {"n_heads": [1, 16, 64],
       "beta": [0.1, 1.0, 10.0],
       "l2_penalty": [1.0, 0.1, 0.01],
       "n_kernels": [8, 32, 128],
       "kernel_size": [5, 7, 9]}

config = {"sequence_reduction_fraction": 0.1,
          "n_heads": 16,
          "beta": 1.0,
          "l2_penalty": 0.1,
          "n_kernels": 32,
          "kernel_size": 7,
          "reduction_mb_size": int(5e3),
          'strategy': "Vanilla",
          'used_sequence_labels_column': 'is_signal_TPR_20%_FDR_50%',
          'evaluate_at': int(2e2),
          "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
          "dataset": f"HIV/v6/phenotype_burden_{args.n_witnesses}",
          "Branch": "HIV",
          'log_training_stats_at': int(2e2),
          'n_updates': args.n_updates,
          'sample_n_sequences': int(1e4),
          'learning_rate': 1e-4,
          "with_seq_loss": False,
          "mul_att_by_factor": False,
          "factor_as_attention": None,
          "average_pooling": False,
          "l2_lambda": 0,
          "seq_loss_lambda": 0,
          "device": device,
          'fpa': False,
          'fps': False}

config.update(vars(args))

results_dir = os.path.join(f"{base_results_dir}_{config['dataset']}", config["timestamp"])

all_labels_columns = ['is_signal_TPR_5%_FDR_0%', 'is_signal_TPR_5%_FDR_10%', 'is_signal_TPR_5%_FDR_50%',
                      'is_signal_TPR_5%_FDR_80%', 'is_signal_TPR_10%_FDR_0%',
                      'is_signal_TPR_10%_FDR_10%', 'is_signal_TPR_10%_FDR_50%',
                      'is_signal_TPR_10%_FDR_80%', 'is_signal_TPR_20%_FDR_0%',
                      'is_signal_TPR_20%_FDR_10%', 'is_signal_TPR_20%_FDR_50%',
                      'is_signal_TPR_20%_FDR_80%', 'is_signal_TPR_50%_FDR_0%',
                      'is_signal_TPR_50%_FDR_10%', 'is_signal_TPR_50%_FDR_50%',
                      'is_signal_TPR_50%_FDR_80%', 'is_signal_TPR_100%_FDR_0%',
                      'is_signal_TPR_100%_FDR_10%', 'is_signal_TPR_100%_FDR_50%',
                      'is_signal_TPR_100%_FDR_80%']

# read pkl file and save folds as np array
with open(f"{root_dir}/used_inds.pkl", 'rb') as f:
    folds = np.array(pkl.load(f)).tolist()

seeds = [0, 1, 2, 3, 4]

task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
    BinaryTarget(column_name='label_positive', true_class_value='True'),
    Sequence_Target(device=config["device"]), ])  # .to(device=config["device"])

hdf5_file, n_repertoires = create_hdf5_file(
    repertoiresdata_path=f"{root_dir}/datasets/{config['dataset']}/data/simulated_repertoires",
    sequence_column='cdr3_aa',
    sequence_counts_column=None,
    sequence_labels_columns=all_labels_columns)

fold = 0
seed = seeds[fold]
config['fold'] = fold
train_dl, train_eval_dl, val_eval_dl, test_eval_dl = make_dataloaders(
    task_definition=task_definition,
    metadata_file=f"{root_dir}/datasets/{config['dataset']}/data/metadata.csv",
    metadata_file_column_sep=",",
    metadata_file_id_column='filename',
    used_sequence_labels_column=config['used_sequence_labels_column'],
    sample_n_sequences=config['sample_n_sequences'],
    split_inds=folds,
    cross_validation_fold=fold,
    force_pos_in_subsampling=config['fps'],
    hdf5_file=hdf5_file,
    n_repertoires=n_repertoires, )

dl_dict = {"train_eval_dl": train_eval_dl, "val_eval_dl": val_eval_dl, "test_eval_dl": test_eval_dl}
logger = Logger(dataloaders=dl_dict, root_dir=root_dir, strategy=config['strategy'])
best_loss = +np.inf
best_model = None
best_HP = None

hp_set = HPs[args.hp_name]
for hp_index in [0, 2]:
    config[args.hp_name] = hp_set[hp_index]
    print(f"HP: {args.hp_name}, value: {config[args.hp_name]}")
    np.random.seed(seed)
    run = wandb.init(project="HIV - VanillaHPO", group=f"{config['strategy']}", reinit=True, config=config, )
    run.name = f"results_idx_{str(fold)}"
    # Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
    sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=config['kernel_size'],
                                                      n_kernels=config['n_kernels'], n_layers=1)
    # Create attention network
    attention_network = AttentionNetwork(n_input_features=config['n_kernels'], n_layers=2, n_units=32)
    # Create output network
    output_network = OutputNetwork(n_input_features=config['n_kernels'], n_units=32, n_layers=1,
                                   n_output_features=task_definition.get_n_output_features())

    model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
                   attention_network=attention_network, output_network=output_network, training_mode=True,
                   consider_seq_counts=False, n_input_features=20, force_pos_in_attention=config['fpa'],
                   sequence_reduction_fraction=config["sequence_reduction_fraction"],
                   reduction_mb_size=config["reduction_mb_size"], device=device,
                   mul_att_by_factor=config["mul_att_by_factor"], average_pooling=config["average_pooling"],
                   factor_as_attention=config["factor_as_attention"]).to(device=device)

    val_loss = train(model, task_definition=task_definition, trainingset_dataloader=train_dl,
                     trainingset_eval_dataloader=train_eval_dl, learning_rate=config['learning_rate'],
                     early_stopping_target_id='label_positive', validationset_eval_dataloader=val_eval_dl,
                     logger=logger, n_updates=config['n_updates'], evaluate_at=config['evaluate_at'], device=device,
                     results_directory=f"{root_dir}{results_dir}", track_test=False, log=True,
                     log_training_stats_at=config['log_training_stats_at'], testset_eval_dataloader=test_eval_dl,
                     with_seq_loss=config["with_seq_loss"], l2_weight_decay=config["l2_lambda"],
                     seq_loss_lambda=config["seq_loss_lambda"])
    if val_loss < best_loss:
        best_model = copy.deepcopy(model)
        best_loss = val_loss
        best_HP = hyperparams_val

# wandb.run.summary.update({"best_val_loss": best_loss})
# wandb.run.summary.update({"best_HP_val": best_HP})
