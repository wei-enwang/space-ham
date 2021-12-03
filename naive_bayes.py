import joblib
import numpy as np

import models


seed = 32
np.random.seed(seed)

# use one dataset for now
train_data_dir = "./data/enron1/"
test_data_dir = "./data/enron2/"
output_dir = "./output/"


model = models.naive_bayes(train_data_dir)
model.fit()
model.test(test_data_dir)

joblib.dump(model, output_dir+"tfidf_nb.pkl")