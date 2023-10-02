# -*- coding: utf-8 -*-
"""
Network architectures

See `deeprc/examples/` for examples.

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import numpy as np
import torch
import torch.nn as nn
import torch.jit as jit
from typing import List


# TODO: correct datatypes of added attributes
def compute_position_features(max_seq_len, sequence_lengths, dtype=np.float16):
    """Compute position features for sequences of lengths `sequence_lengths`, given the maximum sequence length
    `max_seq_len`.
    """
    sequences = np.zeros((max_seq_len + 1, max_seq_len, 3), dtype=dtype)
    half_sequence_lengths = np.asarray(np.ceil(sequence_lengths / 2.), dtype=np.int)
    for i in range(len(sequences)):
        sequence, seq_len, half_seq_len = sequences[i], sequence_lengths[i], half_sequence_lengths[i]
        sequence[:seq_len, -1] = np.abs(0.5 - np.linspace(1.0, 0, num=seq_len)) * 2.
        sequence[:half_seq_len, -3] = sequence[:half_seq_len, -1]
        sequence[half_seq_len:seq_len, -2] = sequence[half_seq_len:seq_len, -1]
        sequence[:seq_len, -1] = 1. - sequence[:seq_len, -1]
    return sequences


def compute_position_features2(max_seq_len, sequence_lengths, dtype=np.float16):
    """Compute position features for sequences of lengths `sequence_lengths`, given the maximum sequence length
    `max_seq_len`.
    """
    print("went through compute_position_features2")
    sequences = np.random.rand(max_seq_len + 1, max_seq_len, 3).astype(dtype)
    sequences /= np.sum(sequences, axis=2, keepdims=True)
    return sequences


class SequenceEmbeddingCNN(nn.Module):
    def __init__(self, n_input_features: int, kernel_size: int = 9, n_kernels: int = 32, n_layers: int = 1):
        """Sequence embedding using 1D-CNN (`h()` in paper)
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features per sequence position
        kernel_size : int
            Size of 1D-CNN kernels
        n_kernels : int
            Number of 1D-CNN kernels in each layer
        n_layers : int
            Number of 1D-CNN layers
        """
        super(SequenceEmbeddingCNN, self).__init__()
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.n_layers = n_layers

        if self.n_layers <= 0:
            raise ValueError(f"Number of layers n_layers must be > 0 but is {self.n_layers}")

        # CNN layers
        network = []
        for i in range(self.n_layers):
            conv = nn.Conv1d(in_channels=n_input_features, out_channels=self.n_kernels, kernel_size=self.kernel_size,
                             bias=True)
            conv.weight.data.normal_(0.0, np.sqrt(1 / np.prod(conv.weight.shape)))
            network.append(conv)
            network.append(nn.SELU(inplace=True))
            n_input_features = self.n_kernels

        self.network = torch.nn.Sequential(*network)

    def forward(self, inputs, sequence_counts, *args, **kwargs):
        """Apply sequence embedding CNN to inputs in NLC format.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_sequences, n_sequence_positions, n_input_features).
        
        Returns
        ---------
        max_conv_acts: torch.Tensor
            Sequences embedded to tensor of shape (n_sequences, n_kernels)
        """
        inputs = torch.transpose(inputs, 1, 2)  # NLC -> NCL
        # Apply CNN
        conv_acts = self.network(inputs)
        # Take maximum over sequence positions (-> 1 output per kernel per sequence)
        with torch.no_grad():
            if sequence_counts is not None:
                first_half = conv_acts[:, :-5, :].clone()  # Use clone() to avoid inplace operation
                scaled_first_half = first_half * sequence_counts.view(-1, 1, 1)
                conv_acts[:, :-5, :] = scaled_first_half
        max_conv_acts, _ = conv_acts.max(dim=-1)
        return max_conv_acts


class SequenceEmbeddingLSTM(nn.Module):
    def __init__(self, n_input_features: int, n_lstm_blocks: int = 32, n_layers: int = 1, lstm_kwargs: dict = None):
        """Sequence embedding using LSTM network (`h()` in paper) with `torch.nn.LSTM`
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        n_lstm_blocks : int
            Number of LSTM blocks in each LSTM layer
        n_layers : int
            Number of LSTM layers
        lstm_kwargs : dict
            Parameters to be passed to `torch.nn.LSTM`
        """
        super(SequenceEmbeddingLSTM, self).__init__()
        self.n_lstm_blocks = n_lstm_blocks
        self.n_layers = n_layers
        if lstm_kwargs is None:
            lstm_kwargs = {}
        self.lstm_kwargs = lstm_kwargs

        if self.n_layers <= 0:
            raise ValueError(f"Number of layers n_layers must be > 0 but is {self.n_layers}")

        # LSTM layers
        network = []
        for i in range(self.n_layers):
            lstm = nn.LSTM(input_size=n_input_features, hidden_size=self.n_lstm_blocks, **lstm_kwargs)
            network.append(lstm)
            n_input_features = self.n_lstm_blocks

        self.network = torch.nn.Sequential(*network)

    def forward(self, inputs, sequence_lengths, *args, **kwargs):
        """Apply sequence embedding LSTM network to inputs in NLC format.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_sequences, n_sequence_positions, n_input_features).
        
        Returns
        ---------
        max_conv_acts: torch.Tensor
            Sequences embedded to tensor of shape (n_sequences, n_kernels)
        """
        inputs = torch.transpose(inputs, 0, 1)  # NLC -> LNC
        output, (hn, cn) = self.network(inputs)
        output = output[sequence_lengths.long() - 1, torch.arange(output.shape[1], dtype=torch.long)]
        return output


class AttentionNetwork(nn.Module):
    def __init__(self, n_input_features: int, n_layers: int = 2, n_units: int = 32):
        """Attention network (`f()` in paper) as fully connected network.
         Currently only implemented for 1 attention head and query.
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        n_layers : int
            Number of attention layers to compute keys
        n_units : int
            Number of units in each attention layer
        """
        super(AttentionNetwork, self).__init__()
        self.n_attention_layers = n_layers
        self.n_units = n_units

        fc_attention = []
        for _ in range(self.n_attention_layers):
            att_linear = nn.Linear(n_input_features, self.n_units)
            att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
            fc_attention.append(att_linear)
            fc_attention.append(nn.SELU())
            n_input_features = self.n_units

        att_linear = nn.Linear(n_input_features, 1)
        att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
        fc_attention.append(att_linear)
        self.attention_nn = torch.nn.Sequential(*fc_attention)

    def forward(self, inputs):
        """Apply single-head attention network.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_sequences, n_input_features)
        
        Returns
        ---------
        attention_weights: torch.Tensor
            Attention weights for sequences as tensor of shape (n_sequences, 1)
        """
        attention_weights = self.attention_nn(inputs)
        return attention_weights


class OutputNetwork(nn.Module):
    def __init__(self, n_input_features: int, n_output_features: int = 1, n_layers: int = 1, n_units: int = 32):
        """Output network (`o()` in paper) as fully connected network
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        n_output_features : int
            Number of output features
        n_layers : int
            Number of layers in output network (in addition to final output layer)
        n_units : int
            Number of units in each attention layer
        """
        super(OutputNetwork, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units

        output_network = []
        for _ in range(self.n_layers - 1):
            o_linear = nn.Linear(n_input_features, self.n_units)
            o_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(o_linear.weight.shape)))
            output_network.append(o_linear)
            output_network.append(nn.SELU())
            n_input_features = self.n_units

        o_linear = nn.Linear(n_input_features, n_output_features)
        o_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(o_linear.weight.shape)))
        output_network.append(o_linear)
        self.output_nn = torch.nn.Sequential(*output_network)

    def forward(self, inputs):
        """Apply output network to `inputs`.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (tr_samples, n_input_features).
        
        Returns
        ---------
        prediction: torch.Tensor
            Prediction as tensor of shape (tr_samples, n_output_features).
        """
        predictions = self.output_nn(inputs)
        return predictions


class DeepRC(nn.Module):
    def __init__(self, max_seq_len: int, n_input_features: int = 20,
                 sequence_embedding_network: torch.nn.Module = SequenceEmbeddingCNN(
                     n_input_features=20 + 3, kernel_size=9, n_kernels=32, n_layers=1),
                 attention_network: torch.nn.Module = AttentionNetwork(
                     n_input_features=32, n_layers=2, n_units=32),
                 output_network: torch.nn.Module = OutputNetwork(
                     n_input_features=32, n_output_features=1, n_layers=0, n_units=32),
                 sequence_embedding_as_16_bit: bool = True,
                 consider_seq_counts: bool = True, consider_seq_counts_after_cnn=False,
                 add_positional_information: bool = True, consider_seq_counts_after_att=False,
                 consider_seq_counts_after_softmax=False, consider_seq_counts_before_maxpool=False,
                 sequence_reduction_fraction: float = 0.1, reduction_mb_size: int = 5e4,
                 device: torch.device = torch.device('cuda:0'), forced_attention: bool = True,
                 force_pos_in_attention=True, training_mode: bool = True, temperature: float = 0.01,
                 mul_att_by_label: bool = False, per_for_tmp: float = 0.75):
        """DeepRC network as described in paper
        
        Apply `.reduce_and_stack_minibatch()` to reduce number of sequences by `sequence_reduction_fraction`
        based on their attention weights and stack/concatenate the bags to a minibatch.
        Then apply `.forward()` to the minibatch to compute the predictions.
        
        Reduction of sequences per bag is performed using minibatches of `reduction_mb_size` sequences to compute the
        attention weights.
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        max_seq_len
            Maximum sequence length to expect. Used for pre-computation of position features.
        n_input_features : int
            Number of input features per sequence position (without position features).
            E.g. 20 for 20 different AA characters.
        sequence_embedding_network
            Sequence embedding network (`h()` in paper).
        attention_network
            Attention network (`f()` in paper).
        output_network
            Output network (`o()` in paper).
        sequence_embedding_as_16_bit : bool
            Compute attention weights using 16bit precision? (Recommended if supported by hardware.)
        consider_seq_counts : bool
            Scale inputs by sequence counts? If False, sequence count information will be ignored.
        add_positional_information : bool
            Add position features to input sequence? Will add 3 position features per sequence position.
        sequence_reduction_fraction : float
            Sequences in a bag are ranked based on attention weights and reduced to the top
             `sequence_reduction_fraction*n_seqs_per_bag` sequences.
             `sequence_reduction_fraction` to be in range [0, 1].
        reduction_mb_size : int
            Reduction of sequences per bag is performed using minibatches of `reduction_mb_size` sequences to compute
             the attention weights.
        device : torch.device
            Device to perform computations on
        forced_attention : bool
            If True, the model will use att_weights=1 for positive sequences and att_weights=0 for negative sequences.
            Otherwise, attention weights will be computed using the attention network.
        """
        super(DeepRC, self).__init__()
        self.n_input_features = n_input_features
        self.max_seq_len = max_seq_len
        self.device = device
        self.consider_seq_counts = consider_seq_counts
        self.consider_seq_counts_after_cnn = consider_seq_counts_after_cnn
        self.consider_seq_counts_after_att = consider_seq_counts_after_att
        self.consider_seq_counts_after_softmax = consider_seq_counts_after_softmax
        self.consider_seq_counts_before_maxpool = consider_seq_counts_before_maxpool
        self.add_positional_information = add_positional_information
        self.sequence_reduction_fraction = sequence_reduction_fraction
        self.reduction_mb_size = int(reduction_mb_size)
        self.forced_attention = forced_attention
        self.force_pos_in_attention = force_pos_in_attention
        self.training_mode = training_mode
        self.temperature = temperature
        self.per_for_tmp = per_for_tmp
        self.mul_att_by_label = mul_att_by_label

        # sequence embedding network (h())
        if sequence_embedding_as_16_bit:
            self.embedding_dtype = torch.float16
            self.sequence_embedding = sequence_embedding_network.to(device=device, dtype=self.embedding_dtype)
        else:
            self.embedding_dtype = torch.float
            self.sequence_embedding = sequence_embedding_network

        # Attention network (f())
        self.attention_nn = attention_network

        # Output network (o)
        self.output_nn = output_network

        # Pre-compute position features for all possible sequence lengths
        position_features = compute_position_features(max_seq_len=max_seq_len,
                                                      sequence_lengths=np.arange(max_seq_len + 1))
        self.position_features = torch.from_numpy(position_features).to(device=device,
                                                                        dtype=self.embedding_dtype).detach()

    def reduce_and_stack_minibatch(self, targets, sequences_of_indices, sequence_lengths, sequence_counts,
                                   sequence_labels):
        """ Apply attention-based reduction of number of sequences per bag and stacked/concatenated bags to minibatch.
        
        Reduces sequences per bag `d_k` to top `d_k*sequence_reduction_fraction` important sequences,
        sorted descending by importance based on attention weights.
        Reduction is performed using minibatches of `reduction_mb_size` sequences.
        Bags are then stacked/concatenated to one minibatch.
        
        Parameters
        ----------
        targets: list of torch.Tensor
            Labels of bags as list of tensors of shapes (n_classes,)
        sequences_of_indices: list of torch.Tensor
            AA indices of bags as list of int8 tensors of shape (n_sequences, n_sequence_positions) = (d_k, d_l)
        sequence_lengths: list of torch.Tensor
            Sequences lengths of bags as tensors of dtype torch.long and shape (n_sequences,) = (d_k,)
        sequence_labels: list of torch.Tensor
            Sequences labels of bags as tensors of dtype torch.long and shape (n_sequences,) = (d_k,)
        sequence_counts: list of torch.Tensor
            Sequences counts per bag as tensors of shape (n_sequences,) = (d_k,).
            The sequences counts are the log(max(counts, 1)).
        
        Returns
        ----------
        mb_targets: list of torch.Tensor
            Labels of bags as tensor of shape (tr_samples, n_classes)
        mb_reduced_inputs: torch.Tensor
            Top `n_sequences*network_config['sequence_reduction_fraction']` important sequences per bag,
            as tensor of shape (tr_samples*n_reduced_sequences, n_input_features, n_sequence_positions),
            where `n_reduced_sequences=n_sequences*network_config['sequence_reduction_fraction']`
        mb_reduced_sequence_lengths: torch.Tensor
            Sequences lengths of `reduced_inputs` per bag as tensor of dtype torch.long and shape
            (tr_samples*n_reduced_sequences,),
            where `n_reduced_sequences=n_sequences*network_config['sequence_reduction_fraction']`
        mb_reduced_sequence_labels: torch.Tensor
            Sequences labels of `reduced_inputs` per bag as tensor of dtype torch.long and shape
            (tr_samples*n_reduced_sequences,),
            where `n_reduced_sequences=n_sequences*network_config['sequence_reduction_fraction']`
        mb_n_sequences: torch.Tensor
            Number of sequences per bag as tensor of dtype torch.long and shape (tr_samples,)
        """
        with torch.no_grad():
            # Move tensors to device
            sequences_of_indices = [t.to(self.device) for t in sequences_of_indices]
            max_mb_seq_len = max(t.max() for t in sequence_lengths)
            sequence_lengths = [t.to(self.device) for t in sequence_lengths]
            sequence_counts = [t.to(self.device) for t in sequence_counts]
            sequence_labels = [t.to(self.device) for t in sequence_labels]

            # TODO: potentially optimize using parallelization
            # Compute features (turn into 1-hot sequences and add position features)
            inputs_list = [
                self.__compute_features__(sequence_of_indices, sequence_lengths, max_mb_seq_len, counts_per_sequence)
                for sequence_of_indices, sequence_lengths, counts_per_sequence
                in zip(sequences_of_indices, sequence_lengths, sequence_counts)]

            # Reduce number of sequences (apply __reduce_sequences_for_bag__ separately to all bags in mb)
            reduced_inputs, reduced_sequence_lengths, reduced_sequence_counts, reduced_sequence_labels, reduced_sequence_attentions = \
                list(zip(*[self.__reduce_sequences_for_bag__(inp, sequence_lengths, sequence_counts, sequence_labels)
                           for inp, sequence_lengths, sequence_counts, sequence_labels
                           in zip(inputs_list, sequence_lengths, sequence_counts, sequence_labels)]))

            # Stack bags in minibatch to tensor
            mb_targets = torch.stack(targets, dim=0).to(device=self.device)
            mb_reduced_sequence_lengths = torch.cat(reduced_sequence_lengths, dim=0)
            mb_reduced_sequence_labels = torch.cat(reduced_sequence_labels, dim=0)
            mb_reduced_sequence_counts = torch.cat(reduced_sequence_counts, dim=0)
            mb_reduced_sequence_attentions = torch.cat(reduced_sequence_attentions, dim=0)
            mb_reduced_inputs = torch.cat(reduced_inputs, dim=0)
            mb_n_sequences = torch.tensor([len(rsl) for rsl in reduced_sequence_lengths], dtype=torch.long,
                                          device=self.device)

        return mb_targets, mb_reduced_inputs, mb_reduced_sequence_lengths, mb_reduced_sequence_counts, \
            mb_reduced_sequence_labels, mb_n_sequences, mb_reduced_sequence_attentions

    def forward(self, inputs_flat, sequence_lengths_flat, n_sequences_per_bag,
                sequence_counts, sequence_labels, sequence_attentions):
        """ Apply DeepRC (see Fig.2 in paper)
        
        Parameters
        ----------
        inputs_flat: torch.Tensor
            Concatenated bags as input of shape
            (tr_samples*n_sequences_per_bag, n_sequence_positions, n_input_features)
        sequence_lengths_flat: torch.Tensor
            Sequence lengths
            (tr_samples*n_sequences_per_bag, 1)
        sequence_labels_flat: torch.Tensor
            Sequence labels
            (tr_samples*n_sequences_per_bag, 1)
        n_sequences_per_bag: torch.Tensor
            Number of sequences per bag as tensor of dtype torch.long and shape (tr_samples,)
        
        Returns
        ----------
        predictions: torch.Tensor
            Prediction for bags of shape (tr_samples, n_outputs)
        """
        # if not is_training:
        #     weights = torch.softmax(sequence_attentions / self.temperature, 0)
        #     inputs_flat = inputs_flat * weights[:, None, None]
        #
        # Get sequence embedding h() for all bags in mb (shape: (d_k, d_v))
        counts = sequence_counts if self.consider_seq_counts_before_maxpool else None
        mb_emb_seqs = self.sequence_embedding(inputs_flat, sequence_counts=counts,
                                              sequence_lengths=sequence_lengths_flat).to(dtype=torch.float32)

        if self.consider_seq_counts_after_cnn:
            mb_emb_seqs = mb_emb_seqs * sequence_counts.unsqueeze(1)
        # Calculate attention weights f() before softmax function for all bags in mb (shape: (d_k, 1))
        if self.forced_attention:
            mb_attention_weights = sequence_labels[:, None]
        else:
            mb_attention_weights = self.attention_nn(mb_emb_seqs)

        # Compute representation per bag (N times shape (d_v,))
        mb_emb_reps_after_attention = []
        start_i = 0
        for n_seqs in n_sequences_per_bag:
            # Get sequence embedding h() for single bag (shape: (n_sequences_per_bag, d_v))
            attention_weights = mb_attention_weights[start_i:start_i + n_seqs]
            # Get attention weights for single bag (shape: (n_sequences_per_bag, 1))
            emb_seqs = mb_emb_seqs[start_i:start_i + n_seqs]
            if self.consider_seq_counts_after_att:
                emb_seqs = emb_seqs * sequence_counts[start_i:start_i + n_seqs].reshape(-1, 1)
            if self.mul_att_by_label:
                attention_weights = attention_weights * torch.softmax(sequence_labels, 0)
            # Calculate attention activations (softmax over n_sequences_per_bag) (shape: (n_sequences_per_bag, 1))
            if self.per_for_tmp != 0:  # not self.training_mode and
                attention_weights = attention_weights  # / self.get_temp(sequence_labels)  # self.temperature
            attention_weights = torch.softmax(attention_weights, dim=0)
            if self.consider_seq_counts_after_softmax:
                attention_weights = attention_weights * sequence_counts[start_i:start_i + n_seqs].reshape(-1, 1)
            # Apply attention weights to sequence features (shape: (n_sequences_per_bag, d_v))
            emb_reps_after_attention = emb_seqs * attention_weights
            # Compute weighted sum over sequence features after attention (format: (d_v,))
            mb_emb_reps_after_attention.append(emb_reps_after_attention.sum(dim=0))
            start_i += n_seqs

        # Stack representations of bags (shape (N, d_v))
        emb_reps_after_attention = torch.stack(mb_emb_reps_after_attention, dim=0)

        # Calculate predictions (shape (N, n_outputs))
        predictions = self.output_nn(emb_reps_after_attention)

        return predictions, mb_attention_weights, emb_reps_after_attention

    def get_temp(self, labels):
        assert not (self.temperature != 0 and self.per_for_tmp != 0), ("temperature and per_for_tmp cannot be set at "
                                                                       "the same time")
        if self.temperature == 0 and self.per_for_tmp == 0:
            return 1
        if self.temperature != 0:
            temp = self.temperature
        elif self.per_for_tmp != 0:
            num_pos = torch.sum(labels)
            if num_pos == 0:
                return 1
            else:
                temp = 1 / torch.log(self.per_for_tmp * (len(labels) - num_pos) / num_pos / (1 - self.per_for_tmp))
                if temp == 0:
                    temp = 1
        return temp

    def __compute_features__(self, sequence_char_indices, sequence_lengths, max_mb_seq_len, counts_per_sequence):
        """Compute one-hot sequence features + position features with shape (n_sequences, sequence_length, n_features)
        from sequence indices
        """
        n_features = self.n_input_features + 3 * self.add_positional_information
        # Send indices to device
        sequence_char_indices = sequence_char_indices.to(dtype=torch.long, device=self.device)
        sequence_lengths = sequence_lengths.to(dtype=torch.long, device=self.device)
        # Only send sequence counts to device, if using sequence counts
        # counts_copy = np.copy(counts_per_sequence.detach().cpu().numpy())
        if self.consider_seq_counts:
            counts_per_sequence = counts_per_sequence.to(dtype=self.embedding_dtype, device=self.device)
        # Allocate tensor for one-hot sequence features + position features
        features_one_hot_shape = (sequence_char_indices.shape[0], max_mb_seq_len, n_features)
        features_one_hot_padded = torch.zeros(size=features_one_hot_shape, dtype=self.embedding_dtype,
                                              device=self.device)
        # Set one-hot sequence features
        features_one_hot = features_one_hot_padded[:, :sequence_char_indices.shape[1]]
        features_one_hot = features_one_hot.reshape((-1, n_features))
        features_one_hot[torch.arange(features_one_hot.shape[0]), sequence_char_indices.reshape((-1))] = 1.
        # Set padded sequence-parts to 0 (sequence_char_indices == -1 marks the padded positions)
        features_one_hot[sequence_char_indices.reshape((-1)) == -1, -1] = 0.
        features_one_hot = features_one_hot.reshape((sequence_char_indices.shape[0], sequence_char_indices.shape[1],
                                                     n_features))
        features_one_hot_padded[:, :sequence_char_indices.shape[1], :] = features_one_hot
        # Scale by sequence counts
        if self.consider_seq_counts:
            # scale the first 15 dimensions of the third dimension of features_one_hot_padded by counts_per_sequence
            # features_one_hot_padded[:, :, :15] = features_one_hot_padded[:, :, :15] * counts_per_sequence[:, None, None]
            features_one_hot_padded = features_one_hot_padded * counts_per_sequence[:, None, None]
        # Add position information
        if self.add_positional_information:
            features_one_hot_padded[:, :sequence_char_indices.shape[1], -3:] = \
                self.position_features[sequence_lengths, :sequence_char_indices.shape[1]]
        # Perform normalization to std=1
        features_one_hot_padded = features_one_hot_padded / features_one_hot_padded.std()
        return features_one_hot_padded

    def __reduce_sequences_for_bag__(self, inputs, sequence_lengths, sequence_counts, sequence_labels):
        """ Reduces sequences to top `n_sequences*sequence_reduction_fraction` important sequences,
        sorted descending by importance based on attention weights.
        Reduction is performed using minibatches of `reduction_mb_size` sequences.

        Parameters
        ----------
        inputs: torch.Tensor
            Input of shape (n_sequences, n_input_features, n_sequence_positions) = (d_k, 20+3, d_l)
        sequence_lengths: torch.Tensor
            Sequences lengths as tensor of dtype torch.long and shape (n_sequences,) = (d_k,)
        sequence_labels: torch.Tensor
            Sequences labels as tensor of dtype torch.long and shape (n_sequences,) = (d_k,)

        Returns
        ----------
        reduced_inputs: torch.Tensor
            Top `n_sequences*sequence_reduction_fraction` important sequences,
            sorted descending by importance as tensor of shape
            (n_reduced_sequences, n_sequence_positions, n_input_features),
            where `n_reduced_sequences=n_sequences*sequence_reduction_fraction`
        reduced_sequence_lengths: torch.Tensor
            Sequences lengths of `reduced_inputs` as tensor of dtype torch.long and shape (n_reduced_sequences,),
            where `n_reduced_sequences=n_sequences*sequence_reduction_fraction`
        reduced_sequence_labels: torch.Tensor
            Sequences labels of `reduced_inputs` as tensor of dtype torch.long and shape (n_reduced_sequences,),
            where `n_reduced_sequences=n_sequences*sequence_reduction_fraction`
        """
        if self.sequence_reduction_fraction <= 1.0:
            # Get number of sequences to reduce to
            n_reduced_sequences = int(sequence_lengths.shape[0] * self.sequence_reduction_fraction)
            # Get number of sequence-minibatches for reduction
            n_mbs = int(np.ceil(inputs.shape[0] / self.reduction_mb_size))
            mb_is = torch.arange(start=0, end=n_mbs, dtype=torch.int)

            # Calculate attention weights for sequences (loop over minibatch of sequences)
            attention_acts = torch.jit.annotate(List[torch.Tensor], [])
            for mb_i in mb_is.unbind(dim=0):
                # Get inputs for current minibatch
                inputs_mb = inputs[mb_i * self.reduction_mb_size:(mb_i + 1) * self.reduction_mb_size].to(
                    device=self.device,
                    dtype=self.embedding_dtype)
                sequence_labels_mb = sequence_labels[
                                     mb_i * self.reduction_mb_size:(mb_i + 1) * self.reduction_mb_size].to(
                    device=self.device, dtype=torch.long)

                sequence_lengths_mb = sequence_lengths[
                                      mb_i * self.reduction_mb_size:(mb_i + 1) * self.reduction_mb_size].to(
                    device=self.device, dtype=torch.long)

                sequence_counts_mb = sequence_lengths[
                                     mb_i * self.reduction_mb_size:(mb_i + 1) * self.reduction_mb_size].to(
                    device=self.device, dtype=torch.long)

                counts = sequence_counts_mb if self.consider_seq_counts_before_maxpool else None
                # Get sequence embedding (h())
                emb_seqs = self.sequence_embedding(inputs_mb, sequence_counts=counts,
                                                   sequence_lengths=sequence_lengths_mb).to(dtype=torch.float32)

                # Calculate attention weights before softmax (f())
                if self.forced_attention:
                    attention_acts.append(sequence_labels_mb)
                else:
                    attention_acts.append(self.attention_nn(emb_seqs).squeeze(dim=-1))

            # Concatenate attention weights for all sequences
            attention_acts = torch.cat(attention_acts, dim=0)

            # Get indices of k sequences with highest attention weights
            _, used_sequences = torch.topk(attention_acts, n_reduced_sequences, dim=0, largest=True, sorted=True)

            if self.force_pos_in_attention and self.training_mode:
                pos_seq_inds = torch.nonzero(sequence_labels)[:, 0]
                used_sequences = list(set(used_sequences.tolist()).union(pos_seq_inds.tolist()))
                used_sequences = torch.tensor(used_sequences)

            # Get top k sequences and sequence lengths
            reduced_inputs = inputs[used_sequences.to(device=self.device)].detach().to(device=self.device,
                                                                                       dtype=self.embedding_dtype)
            reduced_sequence_lengths = \
                sequence_lengths[used_sequences.to(device=self.device)].detach().to(device=self.device,
                                                                                    dtype=self.embedding_dtype)
            reduced_sequence_labels = \
                sequence_labels[used_sequences.to(device=self.device)].detach().to(device=self.device,
                                                                                   dtype=self.embedding_dtype)
            reduced_sequence_counts = \
                sequence_counts[used_sequences.to(device=self.device)].detach().to(device=self.device,
                                                                                   dtype=self.embedding_dtype)
            reduced_sequence_attentions = \
                attention_acts[used_sequences.to(device=self.device)].detach().to(device=self.device,
                                                                                  dtype=self.embedding_dtype)
        else:
            with torch.no_grad():
                reduced_inputs = inputs.detach().to(device=self.device, dtype=self.embedding_dtype)
                reduced_sequence_lengths = sequence_lengths.detach().to(device=self.device, dtype=self.embedding_dtype)
                reduced_sequence_labels = sequence_labels.detach().to(device=self.device, dtype=self.embedding_dtype)
                reduced_sequence_counts = sequence_counts.detach().to(device=self.device, dtype=self.embedding_dtype)
                reduced_sequence_attentions = torch.ones(len(sequence_counts)).to(device=self.device,
                                                                                  dtype=self.embedding_dtype)

        return reduced_inputs, reduced_sequence_lengths, reduced_sequence_counts, reduced_sequence_labels, reduced_sequence_attentions
