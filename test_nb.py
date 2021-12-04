import joblib

# use one dataset for now
test_data_dir = "./data/enron3/"
output_dir = "./output/"

model = joblib.load(output_dir+"tfidf_nb.pkl")
model.test(test_data_dir)

import pdb; pdb.set_trace()
