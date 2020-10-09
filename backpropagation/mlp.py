import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,lr=.1, momentum=0, shuffle=True,hidden_layer_widths=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent 
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLPClassifier(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        # print(X)
        # print(y)
        # X_np = np.array(X)
        X_np = np.array([np.append(x, 1) for x in X])
        y_np = np.array(y).flatten()
        # print(X_np)
        # print(y_np)

        num_input_nodes = len(X_np[0]) - 1
        num_output_nodes = len(np.unique(y_np))

        self.hidden_layer_widths = [num_input_nodes * 2] if self.hidden_layer_widths is None else self.hidden_layer_widths
        self.initial_weights = self.initialize_weights(num_input_nodes, num_output_nodes) if not initial_weights else initial_weights

        self.forward_propogate(X_np[0])
        self.backward_propogate(X_np[0], y_np[0])

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass

    def initialize_weights(self, num_input_nodes, num_output_nodes):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        # print(num_input_nodes, num_output_nodes)
        # print(self.hidden_layer_widths)
        weights = list()

        # print(num_input_nodes)
        # print(num_output_nodes)

        initial_hidden_layer = np.array([np.array([np.random.uniform(-1,1) for i in range(num_input_nodes + 1)]) for i in range(self.hidden_layer_widths[0])])
        weights.append(initial_hidden_layer)
        for i in range(1, len(self.hidden_layer_widths)):
            # hidden_layer = np.array([np.array([np.random.uniform(-1,1) for j in range(self.hidden_layer_widths[i-1] + 1)]) for j in range(self.hidden_layer_widths[i-1])])
            hidden_layer = np.array([np.array([np.random.uniform(-1,1) for j in range(self.hidden_layer_widths[i-1] + 1)]) for j in range(self.hidden_layer_widths[i])])
            weights.append(hidden_layer)
        output_layer = np.array([np.array([np.random.uniform(-1,1) for i in range(self.hidden_layer_widths[-1] + 1)]) for i in range(num_output_nodes)])

        weights.append(output_layer)

        return weights

    def output(self, net):
        # return (lambda net: 1/(1+e^(-net)))(nets)
        return 1/(1+np.exp(-net))

    def gradient(self, output):
        # return (lambda output: output*(1-output)))(outputs)
        return output*(1-output)

    def backward_propogate(self, inputs, target):
        """
        """

        # inputs = np.copy(inputs)

        # output_func = np.vectorize(lambda net: 1/(1+np.exp(-net)))

        # layers_nets = list([np.array([np.sum(np.multiply(inputs, node_weights)) for node_weights in self.initial_weights[0]])])
        # layers_outputs = list(np.array(output_func(layers_nets)))
        # inputs = np.concatenate((layers_outputs[-1], [1]))


        # for i in range(1, len(self.initial_weights)):
        #     layers_nets.append(np.array([np.sum(np.multiply(inputs, node_weights)) for node_weights in self.initial_weights[i]]))
        #     layers_outputs.append(output_func(layers_nets[-1]))
        #     inputs = np.concatenate((layers_outputs[-1], [1]))

        # print(layers_nets)
        # print(layers_outputs)

        sigma_output_fuc = np.vectorize(lambda target, output, gradient: (target - output)*gradient)
        sigma_hidden_fun = np.vectorize(lambda sigmas, weights, gradient: np.sum(np.multiply(sigmas, weights)) * gradient)
        gradient_func = np.vectorize(lambda o: o*(1-o))

        layers_gradients = list(np.array([gradient_func(self.layers_outputs[-1])]))
        layers_sigmas = list(np.array([sigma_output_fuc(target, self.layers_outputs[-1], layers_gradients)]))

        print(layers_gradients)

        # print(len(self.layers_outputs))

        for i in reversed(range(len(self.layers_outputs) - 1)):
            layers_gradients.append(np.array(gradient_func(self.layers_outputs[i])))
            # gradients = gradient_func(self.layers_outputs[i])
            layers_sigmas.append(np.array([sigma_hidden_fun(layers_sigmas[-1], self.layers_outputs[i], gradients) for ]))
        # print(outputs)
        print(layers_gradients)
        # print(sigmas)

    def backward_propogate_output_nodes(self, target):
        """
        """

        output_func = np.vectorize(lambda net: 1/(1+np.exp(-net)))
        gradient_func = np.vectorize(lambda o: o*(1-o))
        pass

    # def forward_propogate(self, inputs):
    #     """
    #     """

    #     inputs = np.copy(inputs)

    #     output_func = np.vectorize(lambda net: 1/(1+np.exp(-net)))
    #     gradient_func = np.vectorize(lambda o: o*(1-o))

    #     # for node_weights in self.initial_weights[0]:
    #     #     print(np.sum(np.multiply(X, node_weights)))

    #     nets = list(np.array([np.sum(np.multiply(inputs, node_weights)) for node_weights in self.initial_weights[0]]))
    #     outputs = output_func(nets)
    #     inputs = np.concatenate((outputs, [1]))

    #     # print(self.initial_weights)

    #     for i in range(1, len(self.initial_weights)):
    #         nets = list(np.array([np.sum(np.multiply(inputs, node_weights)) for node_weights in self.initial_weights[i]]))
    #         outputs = output_func(nets)
    #         inputs = np.concatenate((outputs, [1]))


    #     print(outputs)

    #     # nets = np.array([np.sum(np.multiply(X, node_weights)) for node_weights in self.initial_weights[0]])
    #     # nets = np.array([np.sum(np.multiply(X, node_weights)) for nodes_weights in self.initial_weights for node_weights in nodes_weights])
    #     # print(nets)
    #     pass

    def forward_propogate(self, inputs):
        """
        """

        inputs = np.copy(inputs)

        output_func = np.vectorize(lambda net: 1/(1+np.exp(-net)))

        layers_nets = list([np.array([np.sum(np.multiply(inputs, node_weights)) for node_weights in self.initial_weights[0]])])
        layers_outputs = list(np.array(output_func(layers_nets)))
        inputs = np.concatenate((layers_outputs[-1], [1]))

        for i in range(1, len(self.initial_weights)):
            layers_nets.append(np.array([np.sum(np.multiply(inputs, node_weights)) for node_weights in self.initial_weights[i]]))
            layers_outputs.append(output_func(layers_nets[-1]))
            inputs = np.concatenate((layers_outputs[-1], [1]))
        
        self.layers_nets = layers_nets
        self.layers_outputs = layers_outputs

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        return 0

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        pass
