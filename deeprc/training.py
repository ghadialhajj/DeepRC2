# -*- coding: utf-8 -*-
"""
Training and evaluation of DeepRC model

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os
from itertools import chain

import matplotlib
import numpy as np
import torch
from tqdm import tqdm
from widis_lstm_tools.utils.collection import TeePrint, SaverLoader, close_all
from deeprc.task_definitions import TaskDefinition
import wandb
from deeprc.utils import Logger, evaluate


class ESException(Exception):
    def __init__(self, ):
        super().__init__()
        self.additional_info = "Early Stop Exception"


def train(model: torch.nn.Module, task_definition: TaskDefinition, early_stopping_target_id: str,
          trainingset_dataloader: torch.utils.data.DataLoader, trainingset_eval_dataloader: torch.utils.data.DataLoader,
          validationset_eval_dataloader: torch.utils.data.DataLoader, logger: Logger,
          testset_eval_dataloader: torch.utils.data.DataLoader,
          results_directory: str = "results", n_updates: int = int(1e5), show_progress: bool = True,
          load_file: str = None, device: torch.device = torch.device('cuda:1'),
          num_torch_threads: int = 4, learning_rate: float = 1e-4, l1_weight_decay: float = 0,
          l2_weight_decay: float = 0, log_training_stats_at: int = int(1e2), evaluate_at: int = int(5e3),
          ignore_missing_target_values: bool = True, seq_loss_lambda: float = 1.0,
          with_seq_loss: bool = False, log: bool = True,
          track_test: bool = False):
    """Train a DeepRC model on a given dataset on tasks specified in `task_definition`
     
     Model with lowest validation set loss on target `early_stopping_target_id` will be taken as final model (=early
     stopping). Model performance on validation set will be evaluated every `evaluate_at` updates.
     Trained model, logfile, and tensorboard files will be stored in `results_directory`.
    
    See `deeprc/examples/` for examples.
    
    Parameters
    ----------
    model: torch.nn.Module
         deeprc.architectures.DeepRC or similar model as PyTorch module
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
    early_stopping_target_id: str
        ID of task in TaskDefinition object to use for early stopping.
    trainingset_dataloader: torch.utils.data.DataLoader
         Data loader for training
    trainingset_eval_dataloader: torch.utils.data.DataLoader
         Data loader for evaluation on training set (=no random subsampling)
    validationset_eval_dataloader: torch.utils.data.DataLoader
         Data loader for evaluation on validation set (=no random subsampling).
         Will be used for early-stopping.
    results_directory: str
         Directory to save checkpoint of best trained model, logfile, and tensorboard files in
    n_updates: int
         Number of updates to train for
    show_progress: bool
         Show progressbar?
    load_file: str
         Path to load checkpoint of previously saved model from
    device: torch.device
         Device to use for computations. E.g. `torch.device('cuda:0')` or `torch.device('cpu')`.
         Currently, only devices which support 16 bit float are supported.
    num_torch_threads: int
         Number of parallel threads to allow PyTorch
    learning_rate: float
         Learning rate for adam optimizer
    l1_weight_decay: float
         l1 weight decay factor. l1 weight penalty will be added to loss, scaled by `l1_weight_decay`
    l2_weight_decay: float
         l2 weight decay factor. l2 weight penalty will be added to loss, scaled by `l2_weight_decay`
    log_training_stats_at: int
         Write current training statistics to tensorboard every `log_training_stats_at` updates
    evaluate_at: int
         Evaluate model on training and validation set every `evaluate_at` updates.
         This will also check for a new best model for early stopping.
    ignore_missing_target_values: bool
         If True, missing target values will be ignored for training. This can be useful if auxiliary tasks are not
         available for all samples but might increase the computation time per update.
    """

    os.makedirs(results_directory, exist_ok=True)

    # Read config file and set up results folder
    logfile = os.path.join(results_directory, 'log.txt')
    checkpointdir = os.path.join(results_directory, 'checkpoint')
    os.makedirs(checkpointdir, exist_ok=True)

    # Print all outputs to logfile and terminal
    tee_print = TeePrint(logfile)
    tprint = tee_print.tee_print

    try:
        # Set up PyTorch and numpy random seeds
        torch.set_num_threads(num_torch_threads)

        # Send model to device
        model.to(device)

        # Get optimizer (eps needs to be at about 1e-4 to be numerically stable with 16 bit float)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight_decay, eps=1e-4)

        # Define the learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.1)

        # Create a checkpoint dictionary with objects we want to have saved and loaded if needed
        state = dict(model=model, optimizer=optimizer, update=0, best_validation_loss=np.inf)

        # Setup the SaverLoader class to save/load our checkpoint dictionary to files or to RAM objects
        saver_loader = SaverLoader(save_dict=state, device=device, save_dir=checkpointdir,
                                   n_savefiles=1,  # keep only the latest checkpoint
                                   n_inmem=1  # save checkpoint only in RAM
                                   )

        # Load previous checkpoint dictionary, if load_file is specified
        if load_file is not None:
            state.update(saver_loader.load_from_file(loadname=load_file, verbose=True))
            tprint(f"Loaded checkpoint from file {load_file}")
        update, best_validation_loss = state['update'], state['best_validation_loss']

        # Save checkpoint dictionary to RAM object
        saver_loader.save_to_ram(savename=str(update))
        #
        # Start training
        #
        log_scores(device, early_stopping_target_id, logger,
                   model, task_definition, tprint, trainingset_eval_dataloader,
                   update, validationset_eval_dataloader, testset_eval_dataloader)
        try:
            tprint("Training model...")
            update_progess_bar = tqdm(total=n_updates, disable=not show_progress, position=0,
                                      desc=f"loss={np.nan:6.4f}")
            second_phase = False
            while update < n_updates:
                for data in trainingset_dataloader:
                    if update == 500:
                        evaluate_at = 500

                    # Get samples as lists
                    targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence, pools_per_sequence, sample_ids = data
                    with torch.no_grad():
                        targets, inputs, sequence_lengths, sequence_counts, sequence_labels, sequence_pools, n_sequences, sequence_attentions = \
                            model.reduce_and_stack_minibatch(
                                targets, inputs, sequence_lengths, counts_per_sequence, labels_per_sequence,
                                pools_per_sequence)
                    # Reset gradients
                    optimizer.zero_grad()
                    assert model.training_mode, "Model is not in training mode!"
                    # Calculate predictions from reduced sequences,
                    logit_outputs, attention_outputs, _ = model(inputs_flat=inputs,
                                                                sequence_lengths_flat=sequence_lengths,
                                                                n_sequences_per_bag=n_sequences,
                                                                sequence_labels=sequence_labels)

                    # Calculate losses
                    pred_loss = task_definition.get_loss(raw_outputs=logit_outputs, targets=targets,
                                                         ignore_missing_target_values=ignore_missing_target_values)
                    l1reg_loss = (torch.mean(torch.stack([p.abs().float().mean() for p in model.parameters()])))
                    if with_seq_loss:
                        attention_loss = task_definition.get_sequence_loss(attention_outputs.squeeze(), sequence_labels)
                    else:
                        with torch.no_grad():
                            attention_loss = task_definition.get_sequence_loss(attention_outputs.squeeze(),
                                                                               sequence_labels)

                    if with_seq_loss:
                        loss = pred_loss + l1reg_loss * l1_weight_decay + attention_loss * seq_loss_lambda
                    else:
                        loss = pred_loss + l1reg_loss * l1_weight_decay

                    with torch.no_grad():
                        total_loss = pred_loss + l1reg_loss * l1_weight_decay + attention_loss * seq_loss_lambda
                    # Perform update
                    loss.backward()
                    optimizer.step()

                    # scheduler.step()
                    update += 1
                    update_progess_bar.update()
                    update_progess_bar.set_description(desc=f"loss={total_loss.item():6.4f}", refresh=True)

                    if update % log_training_stats_at == 0:
                        # Add to tensorboard
                        if log:
                            group = 'training/'
                            # Loop through tasks and add losses to tensorboard
                            pred_losses = task_definition.get_losses(raw_outputs=logit_outputs, targets=targets)
                            pred_losses = pred_losses.mean(dim=1)  # shape: (n_tasks, tr_samples, 1) -> (n_tasks, 1)
                            for task_id, task_loss in zip(task_definition.get_task_ids(), pred_losses):
                                wandb.log({f"{group}{task_id}_loss": task_loss}, step=update)  # loss per target
                            # wandb.log({f"{group}total_task_loss": pred_loss}, step=update)  # sum losses over targets
                            wandb.log({f"{group}l1reg_loss": l1reg_loss}, step=update)
                            wandb.log({f"{group}attention_loss": attention_loss}, step=update)
                            wandb.log({f"{group}total_loss": total_loss},
                                      step=update)  # sum losses over targets + l1 + att.

                            group = 'gradients/'
                            cnn_weights = torch.norm(model.sequence_embedding.network[0].weight)
                            seq_grad = torch.norm(list(model.sequence_embedding.parameters())[0].grad).cpu().numpy()
                            att_grad = torch.norm(list(model.attention_nn.parameters())[0].grad).cpu().numpy()
                            out_grad = torch.norm(list(model.output_nn.parameters())[0].grad).cpu().numpy()

                            wandb.log({f"{group}sequence_embedding_grad_mean": seq_grad}, step=update)
                            wandb.log({f"{group}attention_nn_grad_mean": att_grad}, step=update)
                            wandb.log({f"{group}output_nn_grad_mean": out_grad}, step=update)
                            wandb.log({f"{group}cnn_weights": cnn_weights}, step=update)
                            wandb.log({f"{group}learning_rate": optimizer.param_groups[0]["lr"]}, step=update)

                    # Calculate scores and loss on training set and validation set
                    if update % evaluate_at == 0 or update == n_updates:
                        # if update in [10, 100, 1000, 5000, 10000, 15000, 20000] or update == n_updates:
                        scores, scoring_loss = log_scores(device, early_stopping_target_id, logger,
                                                          model, task_definition, tprint, trainingset_eval_dataloader,
                                                          update, validationset_eval_dataloader,
                                                          testset_eval_dataloader, track_test=track_test)

                        # If we have a new best loss on the validation set, we save the model as new best model
                        if best_validation_loss > scoring_loss:
                            best_validation_loss = scoring_loss
                            tprint(f"New best validation loss for {early_stopping_target_id}: {scoring_loss}")
                            # Save current state as RAM object
                            state['update'] = update
                            state['best_validation_loss'] = scoring_loss
                            # Save checkpoint dictionary with currently best model to RAM
                            saver_loader.save_to_ram(savename=str(update))
                            # This would save to disk every time a new best model is found, which can be slow
                            # saver_loader.save_to_file(filename=f'best_so_far_u{update}.tar.gzip')

                    if update >= n_updates:
                        break

            update_progess_bar.close()
        finally:
            # In any case, save the current model and best model to a file
            saver_loader.save_to_file(filename=f'lastsave_u{update}.tar.gzip')
            state.update(saver_loader.load_from_ram())  # load best model so far
            saver_loader.save_to_file(filename=f'best_u{state["update"]}.tar.gzip')
            model.training_mode = False
            print('Finished Training!')

    finally:
        close_all()  # Clean up
    return best_validation_loss


def log_scores(device, early_stopping_target_id, logger, model, task_definition, tprint,
               trainingset_eval_dataloader, update, validationset_eval_dataloader, testset_eval_dataloader,
               track_test=False):
    classes_dict = task_definition.__sequence_targets__[0].classes_dict
    model.training_mode = False
    print("  Calculating training score...")
    scores, sequence_scores = evaluate(model=model,
                                       dataloader=trainingset_eval_dataloader,
                                       task_definition=task_definition,
                                       device=device, logger=None, step=None, log_stats=False)
    print(f" ...done!")
    tprint(f"[training_inference] u: {update:07d}; scores: {scores};")
    group = 'training_inference/'
    for task_id, task_scores in scores.items():
        [wandb.log({f"{group}{task_id}/{score_name}": score}, step=update)
         for score_name, score in task_scores.items()]
    curves = sequence_scores['sequence_class'].pop('curves')
    for task_id, task_scores in sequence_scores.items():
        [wandb.log({f"{group}{task_id}/{score_name}": score}, step=update)
         for score_name, score in task_scores.items()]
        for id, curve in curves.items():
            wandb.log({f"{group}{task_id}/PRC_{classes_dict[id]}": wandb.Image(curve.figure_),
                       f"{group}{task_id}/AP_{classes_dict[id]}": curve.average_precision,
                       f"{group}{task_id}/ranAP_{classes_dict[id]}": curve.prevalence_pos_label}, step=update)

    print("  Calculating validation score...")
    scores, sequence_scores = evaluate(model=model, dataloader=validationset_eval_dataloader,
                                       task_definition=task_definition, device=device, logger=logger, step=update,
                                       log_stats=True)
    scoring_loss = scores[early_stopping_target_id]['loss']

    tprint(f"[validation] u: {update:07d}; scores: {scores};")
    group = 'validation/'
    for task_id, task_scores in scores.items():
        [wandb.log({f"{group}{task_id}/{score_name}": score}, step=update)
         for score_name, score in task_scores.items()]
    curves = sequence_scores['sequence_class'].pop('curves')
    for task_id, task_scores in sequence_scores.items():
        [wandb.log({f"{group}{task_id}/{score_name}": score}, step=update)
         for score_name, score in task_scores.items()]
        for id, curve in curves.items():
            wandb.log({f"{group}{task_id}/PRC_{classes_dict[id]}": wandb.Image(curve.figure_),
                       f"{group}{task_id}/AP_{classes_dict[id]}": curve.average_precision,
                       f"{group}{task_id}/ranAP_{classes_dict[id]}": curve.prevalence_pos_label}, step=update)

    if track_test:
        print("  Calculating test score...")
        scores, sequence_scores = evaluate(model=model, dataloader=testset_eval_dataloader,
                                           task_definition=task_definition, device=device, logger=None, step=None,
                                           log_stats=False)
        tprint(f"[test] u: {update:07d}; scores: {scores};")
        group = 'test/'
        for task_id, task_scores in scores.items():
            [wandb.log({f"{group}{task_id}/{score_name}": score}, step=update)
             for score_name, score in task_scores.items()]
        curves = sequence_scores['sequence_class'].pop('curves')
        for task_id, task_scores in sequence_scores.items():
            [wandb.log({f"{group}{task_id}/{score_name}": score}, step=update)
             for score_name, score in task_scores.items()]
            for id, curve in curves.items():
                wandb.log({f"{group}{task_id}/PRC_{classes_dict[id]}": wandb.Image(curve.figure_),
                           f"{group}{task_id}/AP_{classes_dict[id]}": curve.average_precision,
                           f"{group}{task_id}/ranAP_{classes_dict[id]}": curve.prevalence_pos_label}, step=update)

    model.training_mode = True
    matplotlib.pyplot.close()
    return scores, scoring_loss
