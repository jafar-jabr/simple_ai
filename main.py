import numpy as np
from sklearn.naive_bayes import GaussianNB

from prepare_data import PreparedData

training_set = PreparedData.get_prepared_data()
# print(training_set)
raw_train_features = training_set['features']
raw_train_labels = training_set['labels']
train_features = np.array(raw_train_features)
train_labels = np.array(raw_train_labels)

scenario = {
        "last_work_shift": 2,
        "yesterday_shift": 1,
        "preferred_shift_1": True,
        "preferred_shift_2": True,
        "what_we_need": 1
}

sample_weight = [
        10,
        20,
        1,
        1,
        30
]

predict_set = [PreparedData.prepare_features(scenario)]

clf = GaussianNB()
# clf.fit(train_features, train_labels)
clf.partial_fit(train_features, train_labels, np.unique(train_labels))
GaussianNB(priors=None, var_smoothing=1e-09)
predict = clf.predict(predict_set)
decision = "shift_"+str(predict[0])
print(decision)
