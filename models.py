import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class spam_lstm(nn.Module):
    """LSTM for Spam Classification."""
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 num_layers=3,
                 hidden_size=128,
                 dropout=0.5):
        """
        The constructor for spam_lstm class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            input_size (int): an int representing the RNN input size.

            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300

            num_layers (int): Number of layers of LSTM. Default: 2

            hidden_size (int): Size of hidden states. Default: 128

            dropout (float): Dropout rate. Default: 0.5
        """

        super().__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embed = nn.Embedding.from_pretrained(pretrained_embedding,
                                                        freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embed = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=self.embed_dim,
                                        padding_idx=0,
                                        max_norm=5.0)
        # LSTM Network
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.embed_dim, hidden_size, num_layers=num_layers, batch_first=True,
                        dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(self.hidden_size, 1)
        # self.sig = nn.Sigmoid()
        # self.relu = nn.ReLU()
        


    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embed(input_ids).float()

        # Apply LSTM to input, Output shape: (b, max_len, hidden_size)
        output, _ = self.rnn(x_embed)
        x = self.fc(output)

        return x