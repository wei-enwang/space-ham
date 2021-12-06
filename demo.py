from nltk.corpus import words
import torch
import torch.nn as nn
import joblib

import models
import utils
from preprocess import clean_string, WholeData


PAD_INDEX = 0
max_len = 100

model_filename = "./output/balancew2v_lstmhid128.pt"
train_data_dir = "./data/enron1/"

test_message = "Subject: we will make america great again"


vocab = set([str.lower() for str in words.words()])

train_dataset = WholeData(train_data_dir, src_vocab=vocab, use_max_len=True, max_len=max_len)
context = clean_string(test_message)
w2idx = train_dataset.src_v2id

embed = utils.load_pretrained_vectors(w2idx, "fastText/crawl-300d-2M.vec")
embed = torch.tensor(embed)

model = models.spam_lstm(pretrained_embedding=embed, dropout=0.5)
model.load_state_dict(torch.load(model_filename))
model.eval()
#import pdb; pdb.set_trace()

sent_id = [PAD_INDEX for _ in range(max_len)]
for i, w in enumerate(context):
    if i >= max_len:
        break
    if w not in train_dataset.vocab:
        w = '<unk>'
    sent_id[i] = train_dataset.src_v2id[w]

sent_id = torch.unsqueeze(torch.tensor(sent_id),dim=0)

with torch.no_grad():
    pred = nn.Sigmoid()(model(sent_id))
    pred = torch.squeeze(pred)

    if pred > 0.5:
        print("This is a ham email message.")
    else:
        print("This is a spam email message.")
    
    print(f"The probability of this email being a spam is {1-pred}.")
