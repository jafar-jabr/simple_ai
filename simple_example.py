import numpy as np
from sklearn.naive_bayes import GaussianNB

samples = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

labels = np.array(['A', 'A', 'A', 'B', 'B', 'B'])

clf = GaussianNB()

clf.fit(samples, labels)


print(clf.predict([[1, -1]]))
#
# clf_pf = GaussianNB()
# clf_pf.partial_fit(X, Y, np.unique(Y))
#
# GaussianNB(priors=None, var_smoothing=1e-09)
# print(clf_pf.predict([[2, 1]]))
