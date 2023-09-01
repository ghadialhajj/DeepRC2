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
from deeprc.dataset_readers import make_dataloaders, log_sequence_count_scaling, no_sequence_count_scaling, \
    plain_log_sequence_count_scaling
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
                    type=int, default=int(50))
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

# root_dir = "/home/ghadi/PycharmProjects/DeepRC2/deeprc"
root_dir = "/storage/ghadia/DeepRC2/deeprc"
base_results_dir = "/results/singletask_cnn/ideal"


def get_split_inds(n_folds, cohort, n_tr, n_v, seed):
    split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_separate_test_correct.pkl"
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)["inds"]
        if cohort == 2:
            split_inds = split_inds[-1]
        elif cohort == 1:
            split_inds = split_inds[:-1]
            split_inds = [a for b in split_inds for a in b]
    np.random.seed(seed)
    np.random.shuffle(split_inds)
    # split_inds = [split_inds[i * int(len(split_inds) / n_folds): (i + 1) * int(len(split_inds) / n_folds)] for i in
    #               range(n_folds)]
    train_split_inds = split_inds[:n_tr]
    val_split_inds = split_inds[n_tr: n_tr + n_v]
    return [[], train_split_inds, val_split_inds]


def get_cherry_picked_inds(cohort: int = 1, n_t: int = 30, n_v: int = 160, best_pos: bool = True,
                           best_neg: bool = True):
    split_file = "/storage/ghadia/DeepRC2/deeprc/datasets/splits_used_in_paper/CMV_separate_test_correct.pkl"
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)["inds"]
        train_split_inds = []
        if cohort == 2:
            split_inds = split_inds[-1]
            # Perfect indices: High for pos, low for neg
            # train_split_inds = [731, 704, 753, 762, 730, 715, 686, 691, 758, 703, 710, 669, 709, 712, 713, 718, 723,
            #                     728, 683, 687]
            # High for pos and neg
            # train_split_inds = [731, 704, 753, 762, 730, 715, 686, 691, 758, 703, 772, 692, 700, 705, 706, 734, 666,
            #                     696, 750, 754]
            # Low for pos and High for neg
            # train_split_inds = [697, 680, 720, 708, 732, 768, 773, 745, 748, 749, 772, 692, 700, 705, 706, 734, 666,
            #                     696, 750, 754]
            # Low for pos and for neg
            # train_split_inds = [697, 680, 720, 708, 732, 768, 773, 745, 748, 749, 710, 669, 709, 712, 713, 718, 723,
            #                     728, 683, 687]
            positive = [779, 756, 746, 744, 774, 721, 737, 724, 719, 765, 729, 783, 781, 773, 740, 696, 754, 671, 697,
                        768, 760, 668, 741, 755, 687, 682, 673, 718, 717, 782, 763, 677, 690, 753, 748, 708, 716, 732,
                        704, 726, 699, 714, 707, 723, 679, 772, 665, 731, 739, 694, 752]

            negative = [734, 747, 778, 759, 749, 693, 758, 745, 775, 735, 761, 685, 769, 725, 757, 751, 713, 736, 715,
                        733, 777, 720, 722, 701, 750, 728, 730, 712, 669, 706, 705, 703, 695, 764, 738, 689, 684, 770,
                        743, 678, 672, 666, 784, 762, 766, 767, 771, 776, 780, 688, 667, 670, 674, 675, 676, 680, 681,
                        683, 686, 742, 691, 692, 698, 700, 702, 709, 710, 711, 727]

        elif cohort == 1:
            split_inds = split_inds[:-1]
            split_inds = [a for b in split_inds for a in b]
            positive = [307, 161, 473, 208, 236, 343, 508, 386, 28, 551, 231, 215, 233, 213, 542, 63, 528, 283, 235,
                        415, 290, 462, 380, 648, 295, 568, 658, 664, 437, 83, 164, 168, 596, 26, 399, 35, 111, 650, 418,
                        94, 326, 287, 306, 285, 172, 641, 554, 11, 34, 105, 15, 87, 126, 237, 574, 495, 144, 260, 385,
                        67, 409, 222, 467, 365, 254, 275, 362, 345, 459, 268, 55, 299, 464, 518, 604, 185, 108, 615, 54,
                        25, 639, 524, 463, 396, 192, 81, 112, 361, 581, 656, 57, 494, 229, 557, 273, 546, 389, 412, 378,
                        419, 424, 586, 321, 649, 127, 265, 88, 196, 489, 247, 174, 477, 152, 342, 301, 86, 141, 68, 188,
                        328, 398, 182, 522, 202, 391, 388, 527, 445, 417, 513, 349, 82, 2, 427, 241, 240, 334, 250, 189,
                        223, 123, 16, 532, 548, 550, 622, 274, 590, 592, 539, 53, 602, 319, 194, 316, 584, 486, 576,
                        538, 535, 92, 322, 133, 654, 110, 523, 259, 176, 272, 52, 30, 140, 382, 659, 565, 148, 93, 158,
                        500, 488, 482, 428, 267, 509, 99, 340, 177, 187, 280, 264, 210, 198, 333, 411, 6, 404, 376, 125,
                        446, 351, 405, 390, 327, 184, 318, 291, 485, 438, 453, 569, 497, 616, 470, 332, 600, 620, 452,
                        635, 599, 136, 163, 493, 206, 491, 638, 634, 400, 121, 323, 618, 570, 149, 142, 607, 201, 628,
                        309, 567, 41, 512, 245, 506, 487, 360, 119, 324, 128, 170, 359, 337, 19, 367, 255, 277, 279,
                        162, 315, 450, 263, 344, 253]
            negative = [230, 167, 529, 101, 221, 348, 471, 269, 393, 49, 329, 138, 43, 218, 186, 431, 284, 249, 154,
                        104, 226, 37, 31, 66, 39, 595, 292, 354, 79, 439, 100, 276, 481, 124, 645, 89, 132, 545, 180,
                        45, 78, 14, 643, 147, 651, 243, 251, 534, 338, 454, 289, 238, 220, 183, 61, 559, 636, 346, 114,
                        96, 363, 353, 314, 246, 358, 414, 242, 179, 191, 613, 190, 59, 606, 173, 134, 571, 256, 209,
                        225, 262, 563, 395, 469, 547, 435, 429, 294, 451, 553, 536, 603, 579, 281, 311, 543, 560, 293,
                        562, 258, 228, 217, 203, 504, 505, 530, 106, 401, 368, 605, 117, 625, 97, 8, 341, 633, 614, 644,
                        377, 27, 193, 20, 22, 42, 65, 36, 159, 153, 271, 150, 200, 216, 310, 352, 76, 21, 588, 227, 219,
                        171, 165, 122, 120, 71, 270, 46, 303, 38, 373, 653, 631, 413, 379, 577, 433, 461, 566, 514, 525,
                        540, 519, 516, 502, 483, 544, 422, 143, 476, 197, 442, 432, 199, 248, 297, 317, 330, 331, 312,
                        339, 239, 169, 369, 298, 139, 137, 420, 449, 80, 484, 32, 13, 10, 531, 3, 549, 647, 597, 558,
                        47, 661, 623, 90, 91, 181, 73, 64, 578, 40, 29, 113, 58, 151, 499, 537, 160, 72, 145, 443, 448,
                        77, 205, 479, 583, 69, 23, 541, 617, 166, 335, 175, 421, 455, 116, 475, 12, 498, 336, 304, 430,
                        109, 533, 266, 261, 356, 257, 252, 102, 95, 610, 425, 407, 406, 392, 384, 372, 496, 350, 521,
                        300, 572, 288, 601, 608, 440, 612, 232, 619, 0, 207, 5, 9, 24, 178, 50, 51, 129, 98, 564, 573,
                        646, 1, 85, 74, 44, 56, 103, 107, 18, 115, 118, 135, 157, 552, 282, 561, 296, 302, 305, 347,
                        355, 370, 492, 371, 460, 403, 447]

    n_per_class = int(n_t / 2)
    if best_pos:
        train_split_inds.extend(positive[:n_per_class])
    else:
        train_split_inds.extend(positive[-n_per_class:])
    if best_neg:
        train_split_inds.extend(negative[-n_per_class:])
    else:
        train_split_inds.extend(negative[:n_per_class])
    np.random.seed(0)
    np.random.shuffle(train_split_inds)
    val_split_inds = [x for x in split_inds if x not in train_split_inds]
    np.random.shuffle(val_split_inds)
    val_split_inds = val_split_inds[:n_v]
    return [[], train_split_inds, val_split_inds]


if __name__ == '__main__':
    loss_config = {"min_cnt": 1, "normalize": False, "add_in_loss": True}
    config = {"sequence_reduction_fraction": 0.01, "reduction_mb_size": int(5e3),
              "timestamp": datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), "prop": 0.2,
              "dataset": "AIRR", "pos_weight_seq": 100, "pos_weight_rep": 1., "Branch": "Emerson",
              "dataset_type": "emerson_linz_2", "attention_temperature": 1, "best_pos": None, "best_neg": None,
              "max_factor": 150}

    results_dir = os.path.join(f"{base_results_dir}_{config['dataset']}", config["timestamp"])
    # strategy = "TE"
    # strategy = "FG"
    # strategy = "TASTE"
    strategy = "TASTER"
    # strategy = "PDRC"
    if strategy == "PDRC":
        fpa, fps, wsw, wsi = False, False, False, False
    else:
        fpa, fps, wsw, wsi = True, True, False, "old"  # True, True

    max_aucs = []
    seeds_list = [0, 1, 2]
    for seed in seeds_list:
        n_kernels, kernel_size = 32, 9
        n_samples = 40
        cohort = 2
        split_inds = get_split_inds(0, cohort, n_samples, 60, seed)
        # split_inds = get_cherry_picked_inds(cohort=2, n_v=50, n_t=n_samples, best_pos=config["best_pos"]
        #                                     best_neg=config["best_neg"])

        task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
            BinaryTarget(column_name='CMV', true_class_value='+'),
            Sequence_Target(pos_weight=config["pos_weight_seq"], weigh_seq_by_weight=wsw, weigh_pos_by_inverse=wsi,
                            normalize=loss_config["normalize"], add_in_loss=loss_config["add_in_loss"],
                            device=device), ]).to(device=device)
        #
        trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
            task_definition=task_definition,
            metadata_file=f"{root_dir}/datasets/{config['dataset_type']}/{config['dataset']}/metadata.csv",
            metadata_file_column_sep=",",
            n_worker_processes=4,
            repertoiresdata_path=f"{root_dir}/datasets/{config['dataset_type']}/{config['dataset']}/repertoires",
            metadata_file_id_column='filename',
            sequence_column='cdr3_aa',
            sequence_counts_column="duplicate_count",
            sequence_labels_column='matched',
            sample_n_sequences=args.sample_n_sequences,
            sequence_counts_scaling_fn=plain_log_sequence_count_scaling, # if strategy == "PDRC" else log_sequence_count_scaling,
            with_test=with_test,
            split_inds=split_inds,
            force_pos_in_subsampling=fps,
            min_count=loss_config["min_cnt"],
            max_factor=config["max_factor"]
        )

        dl_dict = {"trainingset_eval": trainingset_eval, "validationset_eval": validationset_eval}
        if with_test:
            dl_dict.update({"testset_eval": testset_eval})
        logger = Logger(dataloaders=dl_dict, with_FPs=False)

        if strategy == "TE":
            group = f"TE_n_up_{args.n_updates}"
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": False})
        elif strategy == "TASTE":
            group = f"TASTE_n_up_{args.n_updates}_prop_{config['prop']}"
            config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": False})
        elif strategy == "TASTER":
            group = f"TASTER_n_up_{args.n_updates}_prop_{config['prop']}"
            config.update({"train_then_freeze": False, "staged_training": True, "forced_attention": False,
                           "plain_DeepRC": False, "rep_loss_only": True})
        elif strategy == "FG":
            group = f"FG_n_up_{args.n_updates}"
            config.update({"train_then_freeze": False, "staged_training": False, "forced_attention": True,
                           "plain_DeepRC": True, "rep_loss_only": False})
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

            run = wandb.init(project="Simulation",
                             group=f"{group}_{n_samples}",
                             # group=f"{group}_{n_samples}_boost_all_+_max_not_in_forward",
                             # group=f"{group}_{n_samples}_scale_y_by_5_nobp_wsi",
                             # group=f"{group}_{n_samples}_temp_{config['attention_temperature']}_oval_bp_otrain_max",
                             # group=f"{group}_boost_positives_oval_20_temp_{config['attention_temperature']}_nobp",
                             # group=f"{group}_boost_pos_lp_hn_wLRS@>0.7_to-5"
                             reinit=True)  # , tags=config["tag"])
            run.name = f"results_idx_{str(seed)}"  # config["run"] +   # += f"_ideal_{config['ideal']}"

            wandb.config.update(args)
            wandb.config.update(config)
            wandb.config.update(loss_config)
            wandb.config.update({"fpa": fpa, "fps": fps, "wsw": wsw, "wsi": wsi, "n_samples": n_samples})
            wandb.config.update({"n_kernels": n_kernels, "kernel_size": kernel_size, "cohort": cohort})

            print("Dataloaders with lengths: ",
                  ", ".join([f"{str(name)}: {len(loader)}" for name, loader in dl_dict.items()]))

            # Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
            sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=kernel_size,
                                                              n_kernels=n_kernels, n_layers=1)
            # Create attention network
            attention_network = AttentionNetwork(n_input_features=n_kernels, n_layers=2, n_units=32)
            # Create output network
            output_network = OutputNetwork(n_input_features=n_kernels,
                                           n_output_features=task_definition.get_n_output_features(), n_layers=1,
                                           n_units=32)
            model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
                           attention_network=attention_network,
                           output_network=output_network,
                           consider_seq_counts=True, consider_seq_counts_after_cnn=False, n_input_features=20,
                           add_positional_information=True, training_mode=True, consider_seq_counts_after_att=False,
                           sequence_reduction_fraction=config["sequence_reduction_fraction"],
                           reduction_mb_size=config["reduction_mb_size"], device=device,
                           forced_attention=config["forced_attention"], force_pos_in_attention=fpa,
                           temperature=config["attention_temperature"]).to(device=device)

            max_auc = train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
                            trainingset_eval_dataloader=trainingset_eval, learning_rate=args.learning_rate,
                            early_stopping_target_id='CMV', validationset_eval_dataloader=validationset_eval,
                            logger=logger, n_updates=args.n_updates, evaluate_at=args.evaluate_at, device=device,
                            results_directory=f"{root_dir}{results_dir}", prop=config["prop"],
                            log_training_stats_at=args.log_training_stats_at,
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
