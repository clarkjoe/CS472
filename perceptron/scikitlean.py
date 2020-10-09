from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

import numpy as np

import sys
sys.path.append('../')
from tools import arff
from perceptron import PerceptronClassifier
# arff_file = "data_banknote_authentication.arff"
# arff_file = "dataset_1.arff"
# arff_file = "dataset_2.arff"
arff_file = "voting_data.arff"
mat = arff.Arff(arff_file)
np_mat = mat.data
data = mat[:,:-1]
labels = mat[:,-1].reshape(-1,1)

#### Make Classifier ####
X_train, X_test, y_train, y_test = train_test_split(data, labels)
clf = Perceptron()
clf.fit(X_train, np.ndarray.flatten(y_train))

train_score = clf.score(X_train, np.ndarray.flatten(y_train))
test_score = clf.score(X_test, np.ndarray.flatten(y_test))

print('Test accuracy: {}'.format(test_score))
print('Train accuracy: {}'.format(train_score))