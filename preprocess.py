import torch
from torch.utils import data
import numpy as np
import os.path
import io
import os
import glob
import pdb
import string
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')

# These IDs are reserved.
PAD_INDEX = 0
UNK_INDEX = 1
SOS_INDEX = 2
EOS_INDEX = 3

MAX_SENT_LENGTH = 100


def txt2list(filename, max_len=None):
    """
    Remove all punctuations and stopwords using ntlk
    Return:
    - context: a list of strings
    """
    context = []
    with io.open(filename, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            # Remove punctuations
            line = [char for char in line if char not in string.punctuation]
            line = ''.join(line)
            # Remove stop words
            context.extend([word.lower() for word in line.strip().split()
                            if word not in stopwords.words('english')])
            if not max_len is None:
                if len(context) >= max_len:
                    break

    return context


def clean_text(filename):
    """
    Remove all punctuations and stopwords using ntlk
    Return:
    - context: a list of strings
    """
    context = ""
    with io.open(filename, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            # Remove punctuations
            line = [char for char in line if char not in string.punctuation]
            line = ''.join(line)
            # Remove stop words
            word_list = [word.lower() for word in line.strip().split()
                            if word not in stopwords.words('english')]
            context += ' '.join(word_list) +' '

    return context


def build_vocab(vocab_dict, context):
    """
    Build the vocabulary if not given word embeddings
    """

    for word in context:
        try:
            vocab_dict[word] += 1
        except:
            vocab_dict[word] = 0


class tdData(object):
    """
    Dataset class for tfidf
    """
    def __init__(self, data_dir):

        ham_path = os.path.join(data_dir, "ham/")
        spam_path = os.path.join(data_dir, "spam/")

        good_mails = glob.glob(ham_path+"*.txt")
        bad_mails = glob.glob(spam_path+"*.txt")

        self.context = []
        self.label_list = []

        for filename in good_mails:

            context = clean_text(filename)

            # add the content of a single email to dataset
            self.context.append(filename)
            self.label_list.append(1)

        for filename in bad_mails:

            context = clean_text(filename)

            self.context.append(filename)
            self.label_list.append(0)
        
        self.label_list = np.array(self.label_list)



class WholeData(data.Dataset):
    def __init__(self, data_dir, src_vocab=None, max_len=MAX_SENT_LENGTH, use_max_len=False):
        # Should use all context

        self.max_len = max_len
        ham_path = os.path.join(data_dir, "ham/")
        spam_path = os.path.join(data_dir, "spam/")

        good_mails = glob.glob(ham_path+"*.txt")
        bad_mails = glob.glob(spam_path+"*.txt")

        

        vocab_dict = {}
        self.context = []
        self.label_list = []

        for filename in good_mails:

            if use_max_len:
                context = txt2list(filename, max_len)
            else:
                context = txt2list(filename)

            if not use_max_len:
                self.max_len = max(self.max_len, len(context))

            if src_vocab is None:
                build_vocab(vocab_dict, context)
    
            # add the content of a single email to dataset
            self.context.append(context)
            self.label_list.append(1)

        for filename in bad_mails:
            if use_max_len:
                context = txt2list(filename, max_len)
            else:
                context = txt2list(filename)

            if not use_max_len:
                self.max_len = max(self.max_len, len(context))
            
            if src_vocab is None:
                build_vocab(vocab_dict, context)

            self.context.append(context)
            self.label_list.append(0)
        
        if not src_vocab is None:
            self.vocab = src_vocab
        else:
            self.vocab = vocab_dict.keys()

        # pad_index is reserved at 0, unk_index is reserved at 1
        self.src_v2id = {v : i+2 for i, v in enumerate(self.vocab)}
        self.src_v2id['<pad>'] = PAD_INDEX
        self.src_v2id['<unk>'] = UNK_INDEX 
        self.src_id2v = {val : key for key, val in self.src_v2id.items()}
        # print(len(self.src_v2id))        


    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        text = self.context[index]
        label = self.label_list[index]

        sent_id = [PAD_INDEX for _ in range(self.max_len)]
        for i, w in enumerate(text):
            if i >= self.max_len:
                break
            if w not in self.vocab:
                w = '<unk>'
            sent_id[i] = self.src_v2id[w]

        # Return context length?
        return torch.tensor(sent_id), label



