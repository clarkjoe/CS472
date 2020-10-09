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
P2Class = PerceptronClassifier(None,lr=0.1,shuffle=True)
X_train, y_train, X_test, y_test = P2Class._train_test_split(data, labels, 70)
P2Class.fit(X_train,y_train)

misclassifications = P2Class.misclassifications
print('Train split accuracy: {}'.format(P2Class.score(X_train, y_train)))
print('Test split accuracy: {}'.format(P2Class.score(X_test, y_test)))
print("Final Weights =",P2Class.get_weights())
for i in range(len(misclassifications)):
    print('epoch: {}, {}'.format(i+1, misclassifications[i]))

# Accuracy = P2Class.score(data,labels)
# print("Accuray = [{:.2f}]".format(Accuracy))