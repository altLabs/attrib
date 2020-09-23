"""
Custom pytorch functions.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import numpy as np
import pandas as pd
import pickle
import os
import sys
import sentencepiece as sp
import datetime
from math import ceil, floor
from sklearn.preprocessing import OneHotEncoder
np.random.seed(44)
torch.manual_seed(44)


ATTRIB_TRAIN_X = "../../../data/tts/train_x_no_nan.pkl"
ATTRIB_TRAIN_Y = "../../../data/tts/y_train_ord.pkl"
ATTRIB_VAL_X = "../../../data/tts/val_x_no_nan.pkl"
ATTRIB_VAL_Y = "../../../data/tts/y_val_ord.pkl"


dna_to_int = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "N": 5,
    "a": 1,
    "c": 2,
    "g": 3,
    "t": 4,
    "n": 5,
}


def get_attrib_paths():
    d = {
        "TRAIN_X": os.path.abspath(ATTRIB_TRAIN_X),
        "TRAIN_Y": os.path.abspath(ATTRIB_TRAIN_Y),
        "VAL_X": os.path.abspath(ATTRIB_VAL_X),
        "VAL_Y": os.path.abspath(ATTRIB_VAL_Y),
    }
    return d


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels in Numpy."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


class myLSTM(nn.Module):

    def __init__(self, params):
        """
        Params are:
        dropout_layers: a list of dropout probabilities after 
                        each layer. Should be the same size as 
                        linear_layer_sizes. If zero or None,
                        no dropout is applied.
        activation: "relu", "elu", or "selu" to be applied to all
                    dense activations
        vocab_size: the number of dimensions in the 1-hot encoding
        num_lstm_hidden: number of hidden units in lstm
        num_lstm_layers: number of stacked layers in lstm. See 
                        https://pytorch.org/docs/stable/nn.html#lstm
                        for details.
        bidir: True or False to process bidirectionally
        other_features_size: the length of the metadata vector
                            supplied to forward().
        linear_layer_sizes: list of number of hidden units in
                            intermediate layers between the lstm and
                            final linear projection.
        num_classes: the number of classes to classify.

        """
        super().__init__()

        sys.stdout.flush()
        try:
            self.my_device = params['my_device']
        except:
            self.my_device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = params
        if not self.config['include_metadata']:
            self.config['other_features_size'] = 0
        if params['activation'] == 'relu':
            self.activation = F.relu
        elif params['activation'] == 'selu':
            self.activation = F.selu
        elif params['activation'] == 'elu':
            self.activation = F.elu

        self.embedding = None
        self.lstm_input_dim = self.config['vocab_size']
        if self.config['embed_dim']:
            self.embedding = nn.Embedding(
                self.config['vocab_size'], self.config['embed_dim'],
                padding_idx=0
            )
            self.lstm_input_dim = self.config['embed_dim']

        # Inputs in form (batch_size, seq_length, feature_dim)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.config['num_lstm_hidden'],
            num_layers=self.config['num_lstm_layers'],
            batch_first=True,
            bidirectional=self.config['bidir'],
        )
        self.lstm_output_size = self.config[
            'num_lstm_hidden'] + (self.config['num_lstm_hidden'] * self.config['bidir'])

        self.linear1 = nn.Linear(
            self.config['other_features_size'] +
            self.lstm_output_size,  # input size
            self.config['linear_layer_sizes'][0]  # output dim
        )
        if len(self.config['linear_layer_sizes']) > 1:
            self.other_linears = nn.ModuleList([
                nn.Linear(
                    self.config['linear_layer_sizes'][i],
                    self.config['linear_layer_sizes'][i + 1]
                ) for i, _ in enumerate(self.config['linear_layer_sizes'][:-1])
            ])
        self.logit_output = nn.Linear(
            self.config['linear_layer_sizes'][-1],
            self.config['num_classes']
        )

    def forward(self, x):
        """
        Expects named args as follows
        seq: (batch_size, max_seq_len, vocab_size OR embed_dim) 
              padded 1-hot encoded sequence batch, sorted by length in
              descending order from the top of batch
        seq_len: (batch_size) vector of the sequence lengths
        metadata: (batch_size, other_features_size) 1-hot encoded metadata vector
        """

        seq_len, seq, metadata = x
        seq = seq.to(self.my_device)
        metadata = metadata.to(self.my_device)
        if self.config['embed_dim']:
            # We expect that the sequence is int-encoded if it is intended for
            # embedding.
            seq = self.embedding(seq)
        seqs = nn.utils.rnn.pack_padded_sequence(
            seq, seq_len, batch_first=True)

        sys.stdout.flush()
        output, (ht, ct) = self.lstm(seqs)

        sys.stdout.flush()

        # ht shaped (num_layers*directions, batch_size, hidden_size)
        if self.config['bidir']:
            temp = tuple(ht[-2:])
            last_h = torch.cat(temp, -1)

        else:
            last_h = ht[-1]
        if self.config['include_metadata']:
            dense_input = torch.cat([last_h, metadata], 1)
        else:
            dense_input = last_h

        hidden = self.linear1(dense_input)
        hidden = self.activation(hidden)
        if len(self.config['linear_layer_sizes']) > 1:
            for i, layer in enumerate(self.other_linears):
                if self.config['dropout_layers'][i]:

                    hidden = F.dropout(
                        hidden,
                        self.config['dropout_layers'][i],
                        training=self.training
                    )
                hidden = self.other_linears[i](hidden)
                hidden = self.activation(hidden)
        output = self.logit_output(hidden)
        return output

class myLSTMOutputHidden(nn.Module):

    def __init__(self, params):
        """
        IGNORES THE INCLUDE METADATA PARAM TO FINETUNE
        Params are:
        dropout_layers: a list of dropout probabilities after 
                        each layer. Should be the same size as 
                        linear_layer_sizes. If zero or None,
                        no dropout is applied.
        activation: "relu", "elu", or "selu" to be applied to all
                    dense activations
        vocab_size: the number of dimensions in the 1-hot encoding
        num_lstm_hidden: number of hidden units in lstm
        num_lstm_layers: number of stacked layers in lstm. See 
                        https://pytorch.org/docs/stable/nn.html#lstm
                        for details.
        bidir: True or False to process bidirectionally
        other_features_size: the length of the metadata vector
                            supplied to forward().
        linear_layer_sizes: list of number of hidden units in
                            intermediate layers between the lstm and
                            final linear projection.
        num_classes: the number of classes to classify.

        """
        super().__init__()
        sys.stdout.flush()
        try:
            self.my_device = params['my_device']
        except:
            self.my_device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = params
        if not self.config['include_metadata']:
            self.config['other_features_size'] = 0
        if params['activation'] == 'relu':
            self.activation = F.relu
        elif params['activation'] == 'selu':
            self.activation = F.selu
        elif params['activation'] == 'elu':
            self.activation = F.elu
        self.metadata_len = 39 # WARNING, HARDCODED

        try:
            self.backprop = params['backprop']
        except:
            self.backprop = True
        self.embedding = None
        self.lstm_input_dim = self.config['vocab_size']
        if self.config['embed_dim']:
            self.embedding = nn.Embedding(
                self.config['vocab_size'], self.config['embed_dim'],
                padding_idx=0
            )
            self.lstm_input_dim = self.config['embed_dim']

        # Inputs in form (batch_size, seq_length, feature_dim)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.config['num_lstm_hidden'],
            num_layers=self.config['num_lstm_layers'],
            batch_first=True,
            bidirectional=self.config['bidir'],
        )
        self.lstm_output_size = self.config[
            'num_lstm_hidden'] + (self.config['num_lstm_hidden'] * self.config['bidir'])

        self.linear1 = nn.Linear(
            self.config['other_features_size'] +
            self.lstm_output_size,  # input size
            self.config['linear_layer_sizes'][0]  # output dim
        )
        if len(self.config['linear_layer_sizes']) > 1:
            self.other_linears = nn.ModuleList([
                nn.Linear(
                    self.config['linear_layer_sizes'][i],
                    self.config['linear_layer_sizes'][i + 1]
                ) for i, _ in enumerate(self.config['linear_layer_sizes'][:-1])
            ])
        
    def forward(self, x):
        """
        Expects named args as follows
        seq: (batch_size, max_seq_len, vocab_size OR embed_dim) 
              padded 1-hot encoded sequence batch, sorted by length in
              descending order from the top of batch
        seq_len: (batch_size) vector of the sequence lengths
        metadata: (batch_size, other_features_size) 1-hot encoded metadata vector
        """
        seq_len, seq, metadata = x
        seq = seq.to(self.my_device)
        metadata = metadata.to(self.my_device)
        if self.config['embed_dim']:
            # We expect that the sequence is int-encoded if it is intended for
            # embedding.
            seq = self.embedding(seq)
        seqs = nn.utils.rnn.pack_padded_sequence(
            seq, seq_len, batch_first=True)
        sys.stdout.flush()
        output, (ht, ct) = self.lstm(seqs)
        sys.stdout.flush()

        # ht shaped (num_layers*directions, batch_size, hidden_size)
        if self.config['bidir']:
            temp = tuple(ht[-2:])
            last_h = torch.cat(temp, -1)
        else:
            last_h = ht[-1]
        if self.config['include_metadata']:
            dense_input = torch.cat([last_h, metadata], 1)
        else:
            dense_input = last_h

        hidden = self.linear1(dense_input)
        hidden = self.activation(hidden)
        if len(self.config['linear_layer_sizes']) > 1:
            for i, layer in enumerate(self.other_linears):
                if self.config['dropout_layers'][i]:
                    # print(f"{self.training}")
                    hidden = F.dropout(
                        hidden,
                        self.config['dropout_layers'][i],
                        training=self.training
                    )
                hidden = self.other_linears[i](hidden)
                hidden = self.activation(hidden)
        if not self.backprop:
            hidden.detach_()
        hidden = torch.cat([hidden, metadata], 1)
        return hidden

class NVCNN(nn.Module):
    """
    Implement the architecture from Nielsen and Voigt (2018)
    """
    def __init__(self,params):
        """
        Params are:
        vocab_size: the number of dimensions in the 1-hot encoding
        lr: learning rate
        filter_number: number of filters (in NV, 1-512)
        filter_len: length of filters (in NV, 1-48)
        num_dense_nodes: size of dense layer after filters
        input_len: length of input (batch_size, vocab_size, input_len)
        num_classes: Number of classes to distinguish between
        """
        super().__init__()
        self.config = params
        
        try:
            self.my_device = self.config['my_device']
        except:
            self.my_device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        
        """
        NV uses padding = "same" to preserve the input and output size of conv.
        We can do the same as follows:
        Lout = Lin + 2 * padding - (filter_len-1) 
        set Lout = Lin and solve for padding:
        padding = (filter_len - 1) / 2
        """
        if self.config['filter_len'] % 2 == 1:
            padding = int((self.config['filter_len'] - 1) / 2)
        else:
            padding = int(ceil((self.config['filter_len'] - 1) / 2)), int(floor((self.config['filter_len'] - 1) / 2))

        # This will be followed by a flatten in the forward pass
        self.ConvBlock = nn.Sequential(
            nn.ConstantPad1d(padding,0),
            # Shapes: Input (N,Cin, Lin_padded), output (N,Cout,Lin)
            nn.Conv1d(int(self.config['vocab_size']),
                      int(self.config['filter_number']),
                      int(self.config['filter_len'])),
            nn.ReLU(),
            # NV pool over a vocab_size == dna_bp_length window
            nn.MaxPool1d(int(self.config['input_len'])))

        # copied equation from:
        # https://pytorch.org/docs/stable/nn.html#maxpool1d
        convblock_flat_output_size = int(self.config['filter_number'] * (
            (
                (self.config['input_len'] - self.config['input_len'])
                / self.config['input_len']
            ) + 1
        ))
        self.DenseLogits = nn.Sequential(
            nn.BatchNorm1d(convblock_flat_output_size),
            # but must be convblock_flat_output_size
            nn.Linear(convblock_flat_output_size, self.config['num_dense_nodes']),
            nn.ReLU(),
            nn.BatchNorm1d(self.config['num_dense_nodes']),
            nn.Linear(self.config['num_dense_nodes'], self.config['num_classes'])
            )       


    def forward(self, x):
        """
        Apply the network to a Nielsen-formatted sequence with a flatten step.
        """
        x = x.to(self.my_device)
        x = self.ConvBlock(x)
        x = x.view(x.size(0), -1) # Flatten

        # Softmax will be computed outside the module, if desired
        logits = self.DenseLogits(x)
        return logits 

class NVAttribDataset(data.Dataset):
    def __init__(self, x_pickle_path, y_pickle_path, vocabulary={}, max_len=8000):
        """
        Paths to pickle files of a data frame (x_pickle_path)
        with a sequence column and other columns assumed to be
        metadata. y_pickle path should be a Series of ints 
        corresponding to the class label. Sequences will be
        truncated and arrange as described in Nielsen and Voigt (2018)
        """
        
        self.dna_to_int = {
                "A": 1,
                "C": 2,
                "G": 3,
                "T": 4,
                "N": 0,
                "a": 1,
                "c": 2,
                "g": 3,
                "t": 4,
                "n": 0,
            }
        self.complement = {
                "A": "T",
                "C": "G",
                "G": "C",
                "T": "A",
                "N": "N",
                "a": "t",
                "c": "g",
                "g": "c",
                "t": "a",
                "n": "n",
            }
        x = pickle.load(
            open(x_pickle_path, "rb")
        )
        self.seqs = x['sequence'].values
        self.max_len = max_len
        self.len = len(self.seqs)
        self.y = pickle.load(
            open(y_pickle_path, "rb")
        ).astype(np.int64)
        assert len(self.y) == self.len, "Lengths don't match"
        # Hardcoded!! Based on vocab above
        self.min_int = 0
        self.max_int = 4
        self.vocab_size = self.max_int + 1

    def __getitem__(self,index):
        """
        Returns a (max_len * 2 + 48, vocab_size) matrix
        randomly subsampled if longer than max_len
        """
        seq = self.seqs[index]
        x = self.to_nielsen_seq(seq,self.max_len)
        return (np.asarray(x,dtype=np.float32),self.y[index])

    def one_hot(self,s):
        """
        Converts sequence of integers into one-hot enncoded numpy array
        """
        s = np.array(s)
        b = np.zeros((s.size, self.vocab_size))
        b[np.arange(s.size),s] = 1
        return b
        
    def to_nielsen_seq(self,s, l):
        """
        From methods of Nielsen and Voigt, 2018    
        https://www.nature.com/articles/s41467-018-05378-z
        1. If shorter than l bp, pad with Ns.
        2. If longer, Nielsen says 'truncate', but not how. I will
        randomly subsample on the fly.
        3. Append 48 Ns
        4. Take the reverse complement of original and append.
        5. One-hot encode with no padding character
        """
        # Ensure that there are no special characters
        assert set(s) <= set("ACGTacgtNn")
        s = list(s)
        if len(s) > l:
            start = np.random.randint(len(s)-l)
            s = s[start:start + l]
        elif len(s) == l:
            s = s
        else:
            s = s + (["N"] * (l - len(s)))

        s_comp = [self.complement[bp] for bp in s]
        s_full = s + (["N"] * 48) + s_comp
        s = [self.dna_to_int[bp] for bp in s_full]
        assert len(s) == (l*2 + 48), "Error selecting subsequence {}".format(s)
        # Transposition to get in pytorch Batch, channel, length format
        return self.one_hot(s).T

    def __len__(self):
        return self.len

    
class AttribDataset(data.Dataset):

    def __init__(self, x_pickle_path, y_pickle_path, vocabulary={}, max_len=512):
        """
        Paths to pickle files of a data frame (x_pickle_path)
        with a sequence column and other columns assumed to be
        metadata. y_pickle path should be a Series of ints 
        corresponding to the class label. Sequences above max_len
        will be randomly subsampled on the fly.
        """
        x = pickle.load(
            open(x_pickle_path, "rb")
        )
        self.seqs = x['sequence'].values
        self.metadata = x.iloc[:, 1:].values
        self.vocabulary = vocabulary
        self.max_len = max_len
        assert len(self.seqs) == len(self.metadata), "Lengths don't match"
        self.len = len(self.seqs)
        self.y = pickle.load(
            open(y_pickle_path, "rb")
        ).astype(np.int64)
        assert len(self.y) == self.len, "Lengths don't match"

    def __getitem__(self, index):
        """
        Returns a tuple of sequence, metadata, label.
        Sequence in a variable length sequence of integers
        randomly subsampled if longer than max_len
        """
        seq = self.seqs[index]
        seq = self._to_int(seq)
        if len(seq) > self.max_len:
            start = np.random.randint(len(seq) - self.max_len)
            seq = seq[start:start + self.max_len]
        return (seq, self.metadata[index], self.y[index])

    def _to_int(self, seq):
        return [self.vocabulary[c] for c in seq]

    def __len__(self):
        return self.len


class BPEAttribDataset(AttribDataset):
    """
    Uses a sentencpiece model to tokenize the sequences. If the model file name contains 'sp', it
    will sample from the top 10 sequences unless force_no_sampling is True. 
    """

    def __init__(self, x_pickle_path, y_pickle_path, vocabulary={}, max_len=512, modelpath=None, force_no_sampling=False, force_sampling=False):
        super().__init__(x_pickle_path, y_pickle_path,
                         vocabulary=vocabulary, max_len=max_len)
        #assert os.path.isfile(model_path), "Must provide valid path to model file"
        self.modelpath = modelpath
        if self.modelpath:
            self.processor = sp.SentencePieceProcessor()
            self.processor.Load(self.modelpath)
        self.force_no_sampling = force_no_sampling
        self.force_sampling = force_sampling

    def _to_int(self, seq):
        if self.modelpath is None:
            seq = super()._to_int(seq)
        else:
            if self.force_no_sampling:
                seq = self.processor.EncodeAsIds(seq)
            elif self.force_sampling:
                seq = self._sample_from_sp(seq)
            else:
                __, filename = os.path.split(self.modelpath)
                if 'sp' in filename:
                    seq = self._sample_from_sp(seq)
                else:
                    seq = self.processor.EncodeAsIds(seq)
        return seq

    def _sample_from_sp(self, seq):
        return self.processor.SampleEncodeAsIds(seq, -1, 0.1)


def length_sort_pad_onehot(batch, vocab_size=5):
    """
    Batch is a tuple of vectors of seq, metadata and label
    returned by get_item. Computes sequence lengths, pads, 
    and one_hot encodes.
    Returns (seq_len, seq, metadata), label
    """
    seq, metadata, label = zip(*batch)
    seq_len = [len(s) for s in seq]
    max_len = max(seq_len)
    sorted_list = sorted(
        zip(seq_len, seq, metadata, label),  # List of tuples
        key=lambda x: -x[0]  # sort by seq length
    )
    seq_len, seq, metadata, label = zip(*sorted_list)
    # make a (batch_size, seq_len) tensor
    padded_seqs = np.zeros((len(seq), max_len), dtype=np.int32)
    for i, s in enumerate(seq):
        padded_seqs[i, :seq_len[i]] = s
    padded_seqs = np.array(padded_seqs) - 1
    one_hots = np.apply_along_axis(
        lambda x: indices_to_one_hot(x, vocab_size), 1, padded_seqs)
    assert one_hots.shape[1] == max_len, "Sequence dimension screwed"
    assert one_hots.shape[2] == vocab_size, "Feature dimension screwed"
    return ((
        seq_len,
        torch.tensor(one_hots, dtype=torch.float32),
        torch.tensor(metadata, dtype=torch.float32)), torch.LongTensor(label))



def length_sort_pad_int(batch, vocab_size=5):
    """
    Batch is a tuple of vectors of seq, metadata and label
    returned by get_item. Computes sequence lengths, pads, 
    and one_hot encodes.
    Returns (seq_len, seq, metadata), label
    """
    seq, metadata, label = zip(*batch)
    seq_len = [len(s) for s in seq]
    max_len = max(seq_len)
    sorted_list = sorted(
        zip(seq_len, seq, metadata, label),  # List of tuples
        key=lambda x: -x[0]  # sort by seq length
    )
    seq_len, seq, metadata, label = zip(*sorted_list)
    # make a (batch_size, seq_len) tensor
    padded_seqs = np.zeros((len(seq), max_len), dtype=np.int32)
    for i, s in enumerate(seq):
        padded_seqs[i, :seq_len[i]] = s
    padded_seqs = np.array(padded_seqs)
    return ((
        seq_len,
        torch.LongTensor(padded_seqs),
        torch.tensor(metadata, dtype=torch.float32)), torch.LongTensor(label))


def get_attrib_data(batch_size, num_workers=0, max_len=512, TRAIN_X=ATTRIB_TRAIN_X, TRAIN_Y=ATTRIB_TRAIN_Y, VAL_X=ATTRIB_VAL_X, VAL_Y=ATTRIB_VAL_Y, **kwargs):
    """
    Loads attribution data into memory. Makes DataLader objects with sensible defaults. 
    For higher num_workers, loads data in parallel across cores. See pytorch docs for details.
    Returns: tuple(train_loader, val_loader).
    """
    train_data = AttribDataset(
        TRAIN_X,
        TRAIN_Y,
        vocabulary=dna_to_int,
        max_len=max_len
    )
    val_data = AttribDataset(
        VAL_X,
        VAL_Y,
        vocabulary=dna_to_int,
        max_len=max_len
    )
    train_loader = data.DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   collate_fn=length_sort_pad_onehot
                                   )
    val_loader = data.DataLoader(val_data,
                                 batch_size=batch_size * 2,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 collate_fn=length_sort_pad_onehot
                                 )
    return (train_loader, val_loader)


def get_attrib_bpe_data(batch_size, num_workers=0, max_len=512, modelpath=None, TRAIN_X=ATTRIB_TRAIN_X, TRAIN_Y=ATTRIB_TRAIN_Y, VAL_X=ATTRIB_VAL_X, VAL_Y=ATTRIB_VAL_Y, **kwargs):
    """
    Loads attribution data into memory. Makes DataLader objects with sensible defaults. 
    For higher num_workers, loads data in parallel across cores. See pytorch docs for details.
    Returns: tuple(train_loader, val_loader).
    """
    train_data = BPEAttribDataset(
        TRAIN_X,
        TRAIN_Y,
        vocabulary=dna_to_int,
        max_len=max_len,
        modelpath=modelpath
    )
    val_data = BPEAttribDataset(
        VAL_X,
        VAL_Y,
        vocabulary=dna_to_int,
        max_len=max_len,
        modelpath=modelpath
    )
    train_loader = data.DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   collate_fn=length_sort_pad_int
                                   )
    val_loader = data.DataLoader(val_data,
                                 batch_size=batch_size * 2,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 collate_fn=length_sort_pad_int
                                 )
    return (train_loader, val_loader)

def get_NV_attrib_data(batch_size, num_workers=0, max_len=8000, TRAIN_X=ATTRIB_TRAIN_X, TRAIN_Y=ATTRIB_TRAIN_Y, VAL_X=ATTRIB_VAL_X, VAL_Y=ATTRIB_VAL_Y, **kwargs):
    """
    Loads attribution data into memory. Makes DataLader objects with sensible defaults. 
    For higher num_workers, loads data in parallel across cores. See pytorch docs for details.
    Returns: tuple(train_loader, val_loader).
    """
    train_data = NVAttribDataset(
        TRAIN_X,
        TRAIN_Y,
        vocabulary=dna_to_int,
        max_len=max_len
    )
    val_data = NVAttribDataset(
        VAL_X,
        VAL_Y,
        vocabulary=dna_to_int,
        max_len=max_len
    )
    train_loader = data.DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                                 )
    val_loader = data.DataLoader(val_data,
                                 batch_size=batch_size * 2,
                                 shuffle=True,
                                 num_workers=num_workers,
                                                 )
    return (train_loader, val_loader)


def accuracy(out, y):
    """
    Returns classification accuracy between out predictions in logit form and
    the ground-truth integer class y.
    """
    preds = torch.argmax(out, dim=1)
    return (preds == y).float().mean()


def loss_batch(model, loss_func, x, y, opt=None):
    """
    Computes loss and backpropagates if optimizer is provided,
    returning loss and the batch size. If optimizer is not provided,
    computes loss and accuracy. 
    Returns with opt: tuple(loss, batch_size)
    Returns without opt: tuple(accuracy, loss, batch_size)
    """
    predictions = model(x)
    loss = loss_func(predictions, y)
    a = accuracy(predictions, y)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return float(a), loss.item(), len(x)


def fit(steps, model, loss_func, opt, train_loader, val_loader, log_interval=100, global_step=None, verbose=False, device='cpu'):
    train_iter = iter(train_loader)
    sum_train_loss = 0
    sum_train_accuracy = 0
    sum_nums = 0
    for i in range(steps):
        model.train()
        x, y = train_iter.next()
        y = y.to(device)
        train_accuracy, train_loss, nums = loss_batch(
            model, loss_func, x, y, opt)
        if (i % log_interval == 0) and verbose:
            print(f"Step {global_step + i}, training loss {train_loss}")
        sum_train_loss += (train_loss * nums)
        sum_train_accuracy += (train_accuracy * nums)
        sum_nums += nums
    avg_train_loss = sum_train_loss / sum_nums
    avg_train_accuracy = sum_train_accuracy / sum_nums
    model.eval()
    with torch.no_grad():
        accuracies, losses, nums = zip(
            *[loss_batch(model, loss_func, x, y.to(device)) for x, y in val_loader]
        )
        val_accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

    if global_step is not None:
        global_step += steps
        steps = global_step
    return avg_train_loss, avg_train_accuracy, val_loss, val_accuracy, steps


def torch_load(filepath, ModelClass, OptimizerClass, get_data_fn, device="cpu"):
    """
    Given a filepath, a class definition ModelClass, and an
    optimizer definition OptimizerClass as well as a function get_data_fn
    which takes a batch_size and returns a tuple of (train_loader, val_loader).
    Makes assumptions about what is contained in the params state.

    Pass device to load the model onto a particular device.
    NOTE that this reinstatiates train_loader and val_loader, which may not be
    the desired behavior w.r.t randomization (untested).

    Should be used as the reciprocal of torch_save().
    Returns: tuple(model, loss_function, optimizer, train_loader, val_loader, params, global_step)
    """
    state = torch.load(filepath)
    params = state['params']
    model = ModelClass(params)
    model.to(device)
    opt = OptimizerClass(model.parameters(), lr=params['lr'])
    model.load_state_dict(state['state_dict'])
    opt.load_state_dict(state['optimizer_state'])
    train_loader, val_loader = get_data_fn(**params)
    return model, state['loss_func'], opt, train_loader, val_loader, params, state['global_step']


def torch_save(filepath, model, loss_func, opt, params, global_step):
    """
    Saves state to filepath. Note that the exact train_loader and val_loader
    instances are NOT SAVED. This may cause an issue with sampling. Untested.
    Returns: None
    """
    state = {
        'state_dict': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'loss_func': loss_func,
        'params': params,
        'global_step': global_step
    }
    torch.save(state, filepath)


def torch_save_with_data(filepath, model, loss_func, opt, train_loader, val_loader, params, global_step):
    """
    Saves state to filepath. Note this pickles train_loader, val_loader which contain the entire
    train and val datasets. It is not recommended atm.
    Returns: None
    """
    state = {
        'state_dict': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'loss_func': loss_func,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'params': params,
        'global_step': global_step
    }
    torch.save(state, filepath)


def torch_load_with_data(filepath, ModelClass, OptimizerClass):
    """
    Reciprocal of torch_save_with_data. See associated warnings there.
    Return behaves like torch_load.
    """
    state = torch.load(filepath)
    model = ModelClass(state['params'])
    opt = OptimizerClass(model.parameters(), lr=params['lr'])
    model.load_state_dict(state['state_dict'])
    opt.load_state_dict(state['optimizer_state'])
    return model, state['loss_func'], opt, state['train_loader'], state["val_loader"], state['params'], state['global_step']


class RaylessTrainable:
    def __init__(self,
                 config,
                 ModelClass,
                 loss_fn,
                 OptClass,
                 get_data_fn,
                 train_step_period=300,
                 ):
        self.config = config
        try:
            self.device = self.config['my_device']
        except:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            self.config['my_device'] = self.device
        self.ModelClass = ModelClass
        self.OptClass = OptClass

        sys.stdout.flush()
        self.model = self.ModelClass(self.config)

        sys.stdout.flush()
        self.model.to(device=self.device)
        self.loss_fn = loss_fn
        self.optim = self.OptClass(
            self.model.parameters(), lr=self.config['lr'])
        self.get_data_fn = get_data_fn
        self.train_loader, self.val_loader = get_data_fn(**self.config)
        self.global_step = 0
        self.train_step_period = train_step_period
        self.ident = datetime.datetime.now().strftime("%d_%b_%Y_%I_%M_%S_%f%p")

    def _train(self):
        # Run your training op for n iterations
        # print(f"IN train, device is {self.device}")
        components = self.model, self.loss_fn, self.optim, self.train_loader, self.val_loader

        train_loss, train_accuracy, val_loss, val_accuracy, self.global_step = fit(
            self.train_step_period, *components, global_step=self.global_step, device=self.device)
        #print("Finish train step")
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "neg_val_loss": -1 * val_loss,
            "global_step": self.global_step
        }
        return metrics

    def _stop(self):
        self.model = None
        self.train_loader = None
        self.val_loader = None

    def _save(self, checkpoint_dir):
        """
        """
        filepath = os.path.join(checkpoint_dir, 'model' + str(self.global_step)+ '.pkl')
        torch_save(filepath, self.model, self.loss_fn,
                           self.optim, self.config, self.global_step)
        return filepath

    def _restore(self, checkpoint_path):
        """
        """
        print(f"IN restore, device is {self.device}")
        self.model, self.loss_fn, self.optim, self.train_loader, self.val_loader, self.config, self.global_step = torch_load(
            checkpoint_path, self.ModelClass, self.OptClass, self.get_data_fn, device=self.device)
