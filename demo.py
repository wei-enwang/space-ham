import torch
import torch.nn as nn
import joblib
from preprocess import PAD_INDEX, clean_string

PAD_INDEX = 0

model_filename = "./output/balancew2v_lstm.pt"

test_message = "Subject: we will make america great again"

context = clean_string(test_message)


model = torch.load(model_filename)
model.eval()


sent_id = [PAD_INDEX for _ in range(model.max_len)]
for i, w in enumerate(context):
    if i >= model.max_len:
        break
    if w not in model.vocab:
        w = '<unk>'
    sent_id[i] = model.src_v2id[w]

sent_id = torch.tensor(sent_id)

with torch.no_grad():
    pred = nn.Sigmoid()(model(sent_id))
    pred = torch.squeeze(pred)

    if pred > 0.5:
        print("This is a ham email message.")
    else:
        print("This is a spam email message.")
    
    print(f"The probability of this email being a spam is {1-pred}.")
