import sys
sys.path.append('../')
from tools import arff
from perceptron import PerceptronClassifier
mat = arff.Arff('linsep2nonorigin.arff')
data = mat.data[:, 0:-1]
labels = mat.data[:, -1].reshape(-1, 1)
PClass = PerceptronClassifier(lr=0.1, shuffle=False, deterministic=10)
PClass.fit(data, labels)
Accuracy = PClass.score(data, labels)
weights = PClass.get_weights()
print("Accuracy = [{:.2f}]".format(Accuracy))
print("Final Weights =", PClass.get_weights())