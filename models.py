from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torch.nn as nn


from preprocess import tdData

class LogReg():
    """Logistic Regression for Spam classification"""
    """NB for Spam classification"""
    def __init__(self, data_dir, use_tfidf=True):
        
        dataset = tdData(data_dir)

        self.train_x = dataset.context
        self.train_y = dataset.label_list

        if use_tfidf:
            self.vectorizer = TfidfVectorizer()
        else:
            self.vectorizer = CountVectorizer()
        
        self.classifier = LogisticRegression()
    
    
    def fit(self):
        """
        Must call this function to fit the model before inference.

        Perform Tf-idf on text document
        Return:
            sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        
        matrix = self.vectorizer.fit_transform(self.train_x)
        self.classifier.fit(matrix, self.train_y)
        print(self.classifier.score(matrix, self.train_y))

    def test(self, test_data_dir):

        test_dataset = tdData(test_data_dir)
        test_x = self.vectorizer.transform(test_dataset.context)
        test_y = test_dataset.label_list
        pred = self.classifier.predict(test_x)
        print(classification_report(test_y, pred))


class kNN():
    """NB for Spam classification"""
    def __init__(self, data_dir, use_tfidf=True):
        
        dataset = tdData(data_dir)

        self.train_x = dataset.context
        self.train_y = dataset.label_list

        if use_tfidf:
            self.vectorizer = TfidfVectorizer()
        else:
            self.vectorizer = CountVectorizer()
        
        self.classifier = KNeighborsClassifier()
    
    
    def fit(self):
        """
        Must call this function to fit the model before inference.

        Perform Tf-idf on text document
        Return:
            sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        
        matrix = self.vectorizer.fit_transform(self.train_x)
        self.classifier.fit(matrix, self.train_y)
        print(self.classifier.score(matrix, self.train_y))

    def test(self, test_data_dir):

        test_dataset = tdData(test_data_dir)
        test_x = self.vectorizer.transform(test_dataset.context)
        test_y = test_dataset.label_list
        pred = self.classifier.predict(test_x)
        print(classification_report(test_y, pred))


class naive_bayes():
    """NB for Spam classification"""
    def __init__(self, data_dir, use_tfidf=True):
        
        dataset = tdData(data_dir)

        self.train_x = dataset.context
        self.train_y = dataset.label_list

        if use_tfidf:
            self.vectorizer = TfidfVectorizer()
        else:
            self.vectorizer = CountVectorizer()
        self.nb = MultinomialNB()
    
    
    def fit(self):
        """
        Must call this function to fit the model before inference.

        Perform Tf-idf on text document
        Return:
            sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        
        matrix = self.vectorizer.fit_transform(self.train_x)
        self.nb.fit(matrix, self.train_y)
        print(self.nb.score(matrix, self.train_y))

    def test(self, test_data_dir):

        test_dataset = tdData(test_data_dir)
        test_x = self.vectorizer.transform(test_dataset.context)
        test_y = test_dataset.label_list
        pred = self.nb.predict(test_x)
        print(classification_report(test_y, pred))


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
        if not pretrained_embedding is None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embed = nn.Embedding.from_pretrained(pretrained_embedding,
                                                        freeze=freeze_embedding)
            print("Using pretrained vectors...")
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
        # fc takes in tensor of shape (..., hidden_size)
        x = self.fc(output[:,-1,:])

        return x
