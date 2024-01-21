import copy
import datetime
import os
import dill as pkl
import numpy as np
import wandb

from deeprc.dataset_readers import make_dataloaders, create_hdf5_file
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.task_definitions import TaskDefinition, BinaryTarget, Sequence_Target
from deeprc.training import train
import torch

from deeprc.utils import Logger, eval_on_test
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--fold', help='Fold index. Default: 0',
                    type=int, default=0)
parser.add_argument('--n_updates', help='Fold index. Default: 0',
                    type=int, default=int(3e4))
parser.add_argument('--strategy', help='Name of the strategy. Default: int(1e3)',
                    type=str, default='FAE')
parser.add_argument('--device', help='GPU ID. Default: 0',
                    type=int, default=0)

args = parser.parse_args()
device_name = f"cuda:{args.device}"
device = torch.device(device_name)

root_dir = "/storage/ghadia/DeepRC2/deeprc"
# root_dir = "/itf-fi-ml/home/ghadia/DeepRC2/deeprc"
base_results_dir = "/results/singletask_cnn/ideal"
hyperparam_names = {'FAE': "mul_att_by_factor", 'AP': None, 'FE': "factor_as_attention", 'TE': 'seq_loss_lambda',
                    'Vanilla': 'l2_lambda'}
# hyperparams_values_25 = {'factor_as_attention': {0: 100, 1: 20, 2: 20, 3: 20, 4: 20},
#                       'mul_att_by_factor': {0: 100, 1: 20, 2: 500, 3: 20, 4: 500},
#                       'l2_lambda': {0: 0, 1: 0, 2: 0.0001, 3: 0.0001, 4: 0.00001},
#                       'seq_loss_lambda': {0: 1, 1: 0.1, 2: 1, 3: 0.1, 4: 0.1}}

hyperparams_values_50 = {'factor_as_attention': {0: 20, 1: 20, 2: 20, 3: 20, 4: 20},
                         'l2_lambda': {0: 0, 1: 0.0001, 2: 0.0001, 3: 0.0001, 4: 0},
                         'mul_att_by_factor': {0: 500, 1: 500, 2: 500, 3: 500, 4: 500},
                         'seq_loss_lambda': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}}

config = {"sequence_reduction_fraction": 0.1,
          "reduction_mb_size": int(5e3),
          'strategy': args.strategy,  # ['FAE', 'TE', 'FE'],
          'evaluate_at': int(2e2),
          "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
          "dataset": f"HIV/v6/phenotype_burden_50",
          "Branch": "HIV",
          'kernel_size': 9,
          'n_kernels': 32,
          'log_training_stats_at': int(1e2),
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

fold = args.fold
seed = seeds[fold]
hyperparam_name = hyperparam_names[config['strategy']]
hyperparams_val = hyperparams_values_50[hyperparam_name][fold]
config['fold'] = fold

for id in range(len(all_labels_columns)):
    print("id", id)
    used_sequence_labels_column = all_labels_columns[id]
    config['id'] = id
    config['used_sequence_labels_column'] = used_sequence_labels_column
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
        n_repertoires=n_repertoires,
        hdf5_file=hdf5_file)
    dl_dict = {"train_eval_dl": train_eval_dl, "val_eval_dl": val_eval_dl, "test_eval_dl": test_eval_dl}

    if config['strategy'] == "TE":
        config.update({'with_seq_loss': True})
    elif config['strategy'] == "AP":
        config.update({"average_pooling": True})
    logger = Logger(dataloaders=dl_dict, root_dir=root_dir, strategy=config['strategy'])

    config[hyperparam_name] = hyperparams_val

    torch.manual_seed(seed)
    np.random.seed(seed)

    run = wandb.init(project="HIV - HM0.2", group=config['strategy'], reinit=True, config=config)
    run.name = f"results_idx_{str(fold)}"

    # Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
    sequence_embedding_network = SequenceEmbeddingCNN(
        n_input_features=20 + 3, kernel_size=config['kernel_size'],
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
                     logger=logger, n_updates=config['n_updates'], evaluate_at=config['evaluate_at'],
                     device=device, results_directory=f"{root_dir}{results_dir}", track_test=False, log=True,
                     log_training_stats_at=config['log_training_stats_at'],
                     testset_eval_dataloader=test_eval_dl, with_seq_loss=config["with_seq_loss"],
                     l2_weight_decay=config["l2_lambda"], seq_loss_lambda=config["seq_loss_lambda"])

    eval_on_test(task_definition, model, test_eval_dl, logger, device, config['n_updates'])
