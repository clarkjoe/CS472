import sys
sys.path.append('../')
from tools import arff
from perceptron import PerceptronClassifier
# arff_file = "data_banknote_authentication.arff"
arff_file = "dataset_1.arff"
# arff_file = "dataset_2.arff"
mat = arff.Arff(arff_file)
np_mat = mat.data
data = mat[:,:-1]
labels = mat[:,-1].reshape(-1,1)

#### Make Classifier ####
P2Class = PerceptronClassifier(lr=0.1,shuffle=False,deterministic=10)
P2Class.fit(data,labels)
Accuracy = P2Class.score(data,labels)
print("Accuray = [{:.2f}]".format(Accuracy))
print("Final Weights =",P2Class.get_weights())


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from sklearn import svm

x = data[:,0]
y = data[:,1]
labels_flat = np.ndarray.flatten(labels)
weights = P2Class.get_weights()

plt.scatter(x,y, c=labels_flat)

dec_x = [0, (-weights[-1])/weights[1]]
dec_y = [(-weights[-1])/weights[0], 0]


fig = plt.figure()
plt.scatter(x,y, c=labels_flat)

if (sum(dec_x) == 0 and sum(dec_y) == 0):
    print('HERE')
    dec_x = [-1,1]
    dec_y = [-1,1]

print(dec_x[0], dec_y[0])
print(dec_x[1], dec_y[1])
plt.plot(dec_x, dec_y)
plt.ylabel('Y')
plt.xlabel('X')
plt.title('Linearly Separable Data')
plt.show()