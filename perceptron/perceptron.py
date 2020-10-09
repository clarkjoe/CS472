import numpy as np
import time
from random import Random
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, deterministic, lr=.1, shuffle=True):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.epochs_wo_improv_stop_criteria = 10

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        # adding the bias
        np_X = np.array([np.append(x, 1) for x in X])
        np_y = np.array(y)
        self.initialize_weights(len(X[0])+1) if initial_weights is None else initial_weights

        # print(np_X)
        # print(np_y)
        # print(self.weights)

        epochs = 0
        epochs_stop_criteria = self.deterministic if self.deterministic is not None else np.math.inf

        # print(epochs_stop_criteria)

        epochs_wo_improv = 0
        best_accuracy = 0

        self.misclassifications = []

        while (epochs <= epochs_stop_criteria and epochs_wo_improv <= self.epochs_wo_improv_stop_criteria):
            outputs = []

            if (self.shuffle):
                np_X, np_y = self._shuffle_data(np_X,np_y)

            for patt_indx in range(len(np_X)):
                target = np_y[patt_indx][0]
                pattern = np_X[patt_indx]

                output = 0 if np.sum([pattern[i] * self.weights[i] for i in range(len(pattern))]) < 0 else 1
                outputs.append(output)
                delta_w = self.lr * (target - output) * pattern
                self.weights += delta_w

            cur_accuracy = self.accuracy(outputs, np_y)

            # print('Outputs: ', outputs)
            # print('Target: ', np.ndarray.flatten(np_y))

            epochs += 1
            if (cur_accuracy == 1):
                break
            elif (cur_accuracy > best_accuracy + 0.015):
                epochs_wo_improv = 0
                best_accuracy = cur_accuracy
            else:
                epochs_wo_improv += 1
            
            self.misclassifications.append(1-cur_accuracy)

        print(epochs)
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        np_X = np.array([np.append(x, 1) for x in X])
        nets = np.dot(np_X, self.weights.T)
        return np.where(nets > 0,1,0)

    def accuracy(self, predict, truth):
        return np.sum(predict == np.ndarray.flatten(truth)) / len(truth)


    def initialize_weights(self, weight_length):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """

        self.weights = np.zeros(weight_length)

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        return self.accuracy(self.predict(X), y)

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        n_elem = X.shape[0]
        indices = np.random.permutation(n_elem)
        return X[indices], y[indices]

    def _train_test_split(self, X, y, perct):
        rand_array = np.random.rand(X.shape[0])
        split = rand_array < np.percentile(rand_array, perct)

        X_train = X[split]
        y_train = y[split]
        X_test =  X[~split]
        y_test = y[~split]

        return X_train, y_train, X_test, y_test


    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
