from mlp import MLPClassifier
from pprint import PrettyPrinter

pp = PrettyPrinter()

bp = MLPClassifier(hidden_layer_widths=[1])
# bp = MLPClassifier()
X = [[1,2]]
y = [[0]]
bp.fit(X, y)
# bp.initialize_weights(1, 1)
initial_weights = bp.initial_weights

# pp.pprint(initial_weights)