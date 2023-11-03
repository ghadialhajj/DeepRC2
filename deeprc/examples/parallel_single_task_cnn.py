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
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget, Sequence_Target
from deeprc.dataset_readers import make_dataloaders, log_sequence_count_scaling_with_boosting, \
    log_sequence_count_scaling_with_positive_increment, no_sequence_count_scaling, \
    plain_log_sequence_count_scaling
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.training import train, evaluate
import wandb
import os
import datetime
from deeprc.utils import Logger, get_split_inds, get_correct_indices, get_cherry_picked_inds, get_original_inds, \
    get_splits_new_emerson
from deeprc.training import ESException

#
# Get command line arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--n_updates', help='Number of updates to train for. Recommended: int(1e5). Default: int(1e3)',
                    type=int, default=int(15e3))
# type=int, default=int(100))
parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping. '
                                          'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(2e2))
# type=int, default=int(10))
parser.add_argument('--log_training_stats_at', help='Log training stats every `log_training_stats_at` updates. '
                                                    'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(1e2))
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
device_name = "cuda:1"
with_test = True
device = torch.device(device_name)

root_dir = "/storage/ghadia/DeepRC2/deeprc"
base_results_dir = "/results/singletask_cnn/ideal"

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

# all_labels_columns = ['is_signal_TPR_20%_FDR_10%']

if __name__ == '__main__':
    loss_config = {"min_cnt": 1, "normalize": False, "add_in_loss": False}
    config = {"sequence_reduction_fraction": 0.1, "reduction_mb_size": int(5e3),
              "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), "prop": 0.02,
              "dataset": "phenotype_burden_40", "pos_weight_seq": 100, "pos_weight_rep": 1., "Branch": "HIV",
              "dataset_type": "HIV", "attention_temperature": 0, "best_pos": None, "best_neg": None,
              "max_factor": None, "consider_seq_counts": False, "consider_seq_counts_after_cnn": False,
              "consider_seq_counts_after_att": False, "consider_seq_counts_after_softmax": False,
              "consider_seq_counts_before_maxpool": False, "add_positional_information": True, "per_for_tmp": 0,
              "non_zeros_only": False}
    # todo: check whether TE performed well after correcting fps, on cohort 2
    # todo: when scaling with PDRC, I didn't use FPS and FPA. Try with them
    results_dir = os.path.join(f"{base_results_dir}_{config['dataset']}", config["timestamp"])
    # strategy = "TE"
    # strategy = "FG"
    # strategy = "TASTE"
    # strategy = "TASTER"
    # strategy = "UniAtt"
    strategy = "F*G*E"
    # strategy = "PDRC"
    # if strategy == "PDRC":
    fpa, fps, wsw, wsi = False, False, False, False
    scaling_fn = no_sequence_count_scaling
    # else:
    #     fpa, fps, wsw, wsi = True, True, False, True  # True, True
    #     scaling_fn = log_sequence_count_scaling_with_positive_increment
    config.update({"scaling_fn": scaling_fn})
    max_aucs = []
    seeds_list = [0, 1, 2]
    seed = 2
    # for seed in seeds_list:
    for used_sequence_labels in all_labels_columns:
        config.update({"used_sequence_labels": used_sequence_labels})
        n_kernels, kernel_size = 32, 9
        task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
            BinaryTarget(column_name='label_positive', true_class_value='True'),
            Sequence_Target(pos_weight=config["pos_weight_seq"], weigh_seq_by_weight=wsw, weigh_pos_by_inverse=wsi,
                            normalize=loss_config["normalize"], add_in_loss=loss_config["add_in_loss"],
                            device=device), ]).to(device=device)
        #
        trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
            task_definition=task_definition,
            metadata_file=f"{root_dir}/datasets/{config['dataset_type']}/{config['dataset']}/data/metadata.csv",
            metadata_file_column_sep=",",
            n_worker_processes=4,
            repertoiresdata_path=f"{root_dir}/datasets/{config['dataset_type']}/{config['dataset']}/data/simulated_repertoires",
            metadata_file_id_column='filename',
            sequence_column='cdr3_aa',
            sequence_labels_columns=['is_signal_TPR_5%_FDR_0%', 'is_signal_TPR_5%_FDR_10%', 'is_signal_TPR_5%_FDR_50%',
                                     'is_signal_TPR_5%_FDR_80%', 'is_signal_TPR_10%_FDR_0%',
                                     'is_signal_TPR_10%_FDR_10%', 'is_signal_TPR_10%_FDR_50%',
                                     'is_signal_TPR_10%_FDR_80%', 'is_signal_TPR_20%_FDR_0%',
                                     'is_signal_TPR_20%_FDR_10%', 'is_signal_TPR_20%_FDR_50%',
                                     'is_signal_TPR_20%_FDR_80%', 'is_signal_TPR_50%_FDR_0%',
                                     'is_signal_TPR_50%_FDR_10%', 'is_signal_TPR_50%_FDR_50%',
                                     'is_signal_TPR_50%_FDR_80%', 'is_signal_TPR_100%_FDR_0%',
                                     'is_signal_TPR_100%_FDR_10%', 'is_signal_TPR_100%_FDR_50%',
                                     'is_signal_TPR_100%_FDR_80%'],
            used_sequence_labels_column=used_sequence_labels,
            sample_n_sequences=args.sample_n_sequences,
            sequence_counts_column=None,
            sequence_counts_scaling_fn=no_sequence_count_scaling,
            non_zeros_only=config["non_zeros_only"],
            with_test=with_test,
            force_pos_in_subsampling=fps,
            min_count=loss_config["min_cnt"],
            max_factor=config["max_factor"]
        )
        dl_dict = {"trainingset_eval": trainingset_eval, "validationset_eval": validationset_eval}
        if with_test:
            dl_dict.update({"testset_eval": testset_eval})
        logger = Logger(dataloaders=dl_dict, with_FPs=False)

        if strategy == "TE":
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": False, "mul_att_by_factor": False})
        elif strategy == "TASTE":
            config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": False, "mul_att_by_factor": False})
        elif strategy == "TASTER":
            config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": True, "mul_att_by_factor": False})
        elif strategy == "FG":
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": True,
                           "plain_DeepRC": True, "rep_loss_only": False, "mul_att_by_factor": False})
        elif strategy == "PDRC":
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                           "plain_DeepRC": True, "rep_loss_only": False, "mul_att_by_factor": False})
        elif strategy == "UniAtt":
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                           "plain_DeepRC": True, "rep_loss_only": False, "mul_att_by_factor": 10,
                           "uniform_attention": True})  # currently not used
        elif strategy == "F*G*E":
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                           "plain_DeepRC": True, "rep_loss_only": False, "mul_att_by_factor": 500})
        elif strategy == "TEOR":
            config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": True, "mul_att_by_factor": False})
        else:
            raise "Invalid strategy"
        try:
            # Set random seed (will still be non-deterministic due to multiprocessing but weight init will be the same)
            torch.manual_seed(seed)
            np.random.seed(seed)

            run = wandb.init(project="HIV", group=f"{strategy}", reinit=True)
            run.name = f"results_idx_{str(seed)}"

            wandb.config.update(args)
            wandb.config.update(config)
            wandb.config.update(loss_config)
            wandb.config.update(
                {"fpa": fpa, "fps": fps, "wsw": wsw, "wsi": wsi})
            wandb.config.update({"n_kernels": n_kernels, "kernel_size": kernel_size})

            print("Dataloaders with lengths: ",
                  ", ".join([f"{str(name)}: {len(loader)}" for name, loader in dl_dict.items()]))

            # Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
            sequence_embedding_network = SequenceEmbeddingCNN(
                n_input_features=20 + 3 * config["add_positional_information"], kernel_size=kernel_size,
                n_kernels=n_kernels, n_layers=1)
            # Create attention network
            attention_network = AttentionNetwork(n_input_features=n_kernels, n_layers=2, n_units=32)
            # Create output network
            output_network = OutputNetwork(n_input_features=n_kernels,
                                           n_output_features=task_definition.get_n_output_features(), n_layers=1,
                                           n_units=32)
            model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
                           attention_network=attention_network, output_network=output_network,
                           consider_seq_counts=config["consider_seq_counts"], n_input_features=20,
                           consider_seq_counts_after_cnn=config["consider_seq_counts_after_cnn"],
                           add_positional_information=config["add_positional_information"], training_mode=True,
                           consider_seq_counts_after_att=config["consider_seq_counts_after_att"],
                           consider_seq_counts_after_softmax=config["consider_seq_counts_after_softmax"],
                           consider_seq_counts_before_maxpool=config["consider_seq_counts_before_maxpool"],
                           sequence_reduction_fraction=config["sequence_reduction_fraction"],
                           reduction_mb_size=config["reduction_mb_size"], device=device,
                           forced_attention=config["forced_attention"], force_pos_in_attention=fpa,
                           temperature=config["attention_temperature"], mul_att_by_factor=config["mul_att_by_factor"],
                           per_for_tmp=config["per_for_tmp"]).to(device=device)

            max_auc = train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
                            trainingset_eval_dataloader=trainingset_eval, learning_rate=args.learning_rate,
                            early_stopping_target_id='label_positive', validationset_eval_dataloader=validationset_eval,
                            logger=logger, n_updates=args.n_updates, evaluate_at=args.evaluate_at, device=device,
                            results_directory=f"{root_dir}{results_dir}", prop=config["prop"],
                            log_training_stats_at=args.log_training_stats_at,
                            train_then_freeze=config["train_then_freeze"], staged_training=config["staged_training"],
                            plain_DeepRC=config["plain_DeepRC"], log=True, rep_loss_only=config["rep_loss_only"],
                            config=config, loss_config=loss_config)

            # logger.log_stats(model=model, device=device, step=args.n_updates)
            max_aucs.append(max_auc)
            #
            # Evaluate trained model on testset
            #
            if with_test:
                assert not model.training_mode, "Model is in training mode!"
                scores, sequence_scores = evaluate(model=model, dataloader=testset_eval,
                                                   task_definition=task_definition,
                                                   device=device)
                wandb.run.summary.update(scores["label_positive"])
                wandb.run.summary.update(sequence_scores["sequence_class"])
                print(f"Test scores:\n{scores}")
            wandb.finish()
        except ValueError as ve:
            print("Error: ", ve)
        except ESException as ese:
            print("ESE")
    print("max_aucs: ", max_aucs)
