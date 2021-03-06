import joblib
import numpy as np

import models


seed = 32
np.random.seed(seed)

# use one dataset for now
train_data_dir = "./data/enron1/"
test_data_dir = "./data/enron2/"
output_dir = "./output/"


model = models.LogReg(train_data_dir, use_tfidf=False)
model.fit()
model.test(test_data_dir)

# joblib.dump(model, output_dir+"_log_reg.pkl")
