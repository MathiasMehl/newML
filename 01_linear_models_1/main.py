import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC


# A small utility we shall need later:
def maxpos(A):
    '''
    Takes an n x k array A, and returns 1-dim n array where the i'th
    entry is the index of column in A where the i'th row of A has its
    maximal value (application: turns a probabilitiy distribution over
    k classes for n instances into a single prediction)
    '''
    return np.argmax(A, axis=1)


irisdata = load_iris()

features = [0,1,2,3]
features_train, features_test, labels_train, labels_test = train_test_split(irisdata.data[:, features],
                                                                                irisdata.target,
                                                                                test_size=0.30, train_size=0.70)

onehotclasses = np.zeros((irisdata.target.size, 3))
for i in range(irisdata.target.size):
    onehotclasses[i][irisdata.target[i]] = 1.0

irislinreg = LinearRegression()
irislinreg.fit(irisdata.data[:, features], onehotclasses)

Z = irislinreg.predict(irisdata.data)
pred_labels = maxpos(Z)
print(pred_labels)

print("Accuracy train: {}".format(accuracy_score(irisdata.target, pred_labels)))