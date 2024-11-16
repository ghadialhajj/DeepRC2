import numpy as np
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling, make_test_dataloader
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.training import evaluate
import re
import os
import pickle

device = torch.device("cuda:0")
task_definition = TaskDefinition(
    targets=[BinaryTarget(column_name='ML_class', true_class_value='T1D', pos_weight=1.)]).to(device=device)

res_dict = {}
best_ids = [18, 21, 21, 21, 21]
# todo use the same split as before
# todo split cohort 1 into tr+val, build a logreg morel based on the predictions of the 5 DeepRC models, then test it
# on cohort 2+3
# generate 1298 integers from 0 to 1297, shuffle them and split them into 5 folds
split_inds = np.random.RandomState(seed=0).permutation(1298)
split_inds = np.array_split(split_inds, 5)

dl_config = {"task_definition": task_definition,
             "metadata_file_id_column": 'filename',
             "metadata_file_column_sep": ',',
             "sequence_column": 'amino_acid',
             "sequence_counts_column": 'templates',
             "sample_n_sequences": 10000,
             "sequence_counts_scaling_fn": no_sequence_count_scaling,
             "with_test": False}

# trainingset, trainingset_eval, *_ = make_dataloaders(
#     metadata_file="/storage/ghadia/DeepRC2/deeprc/datasets/T1D/cohort1_wws.csv",
#     repertoiresdata_path="/storage/ghadia/DeepRC2/deeprc/datasets/T1D/cohort_1",
#     split_inds=split_inds,
#     **dl_config)

main_dir = f"/storage/ghadia/DeepRC2/results/T1D_AC_clip"

testset_eval = make_test_dataloader(metadata_file="/storage/ghadia/DeepRC2/deeprc/datasets/T1D/cohorts2_3.csv",
                                    repertoiresdata_path="/storage/ghadia/DeepRC2/deeprc/datasets/T1D/cohorts2_3",
                                    split_inds=list(range(857)),
                                    **dl_config
                                    )

for fold in range(5):
    # if fold != 4:
    #     continue
    n_kernels = 64 if fold else 32
    sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=5, n_kernels=n_kernels,
                                                      n_layers=1)
    attention_network = AttentionNetwork(n_input_features=n_kernels, n_layers=2, n_units=32)
    output_network = OutputNetwork(n_input_features=n_kernels,
                                   n_output_features=task_definition.get_n_output_features(), n_layers=1, n_units=64)
    model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
                   attention_network=attention_network, output_network=output_network, consider_seq_counts=False,
                   n_input_features=20, add_positional_information=True, sequence_reduction_fraction=0.1,
                   reduction_mb_size=int(5e4), device=device).to(device=device)

    pattern = r'Validation loss: (\d+\.\d+)'

    best_idx = best_ids[fold]
    # filename = f"{main_dir}/log_f{fold}.txt"
    # with open(filename, 'r') as file:
    #     # read a list of lines into data
    #     data = file.readlines()
    #     best_idx = min([(float(re.search(pattern, line)[1]), idx) for idx, line in enumerate(data) if
    #                     "Validation loss" in line])[1]

    model_dir = f"/storage/ghadia/DeepRC2/results/T1D_AC_clip/fold{fold}/idx{best_idx}/checkpoint/"
    file = [file_name for file_name in os.listdir(model_dir) if file_name.endswith('.tar.gzip')][0]
    checkpoint = torch.load(model_dir + file)
    model.load_state_dict(checkpoint['model'])

    tr_sample_ids, tr_raw_outputs, tr_targets = evaluate(model=model, dataloader=testset_eval,
                                                         task_definition=task_definition, device=device)

    res_dict[fold] = (tr_sample_ids, tr_raw_outputs, tr_targets)

with open(f'{main_dir}/folds_outputs23.pkl', 'wb') as f:
    pickle.dump(res_dict, f)

# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)
