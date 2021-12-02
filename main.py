import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils import data
from preprocess import WholeData

import models
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda"   # use gpu whenever you can!

seed = 32
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# use one dataset for now
train_data_dir = "./data/enron1/"
test_data_dir = "./data/enron2/"
output_dir = "./output/"

# hyperparameters
batch_size = 128
dropout = 0.1
learning_rate = 1e-3
epochs = 30

train_dataset = WholeData(train_data_dir)
test_dataset = WholeData(test_data_dir)

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_dataloader = data.DataLoader(test_dataset)
w2idx = train_dataset.src_v2id

embed = utils.load_pretrained_vectors(w2idx, "fastText/crawl-300d-2M.vec")
embed = torch.tensor(embed)

model = models.spam_lstm(pretrained_embedding=embed, dropout=dropout).to(device)
loss_fn = nn.BCEWithLogitsLoss().to(device)
opt = Adam(model.parameters(), lr=learning_rate)

utils.train_full_test_once(train_dataloader, test_dataloader, model, loss_fn, optimizer=opt, epochs=epochs, print_every=1, img_dir=output_dir)

torch.save(model.state_dict(), output_dir+"w2v_lstm.pt")
