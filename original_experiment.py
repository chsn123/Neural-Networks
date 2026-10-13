"""Reproduce the initial 3-input, 3-output forward-pass experiment."""

import numpy as np

from neural_network import sigmoid


inputs = np.array([0.7, 0.3, 0.1])
weights = np.array([
    [0.5, 0.2, -0.9],
    [0.2, 0.7, 0.8],
    [-0.5, 0.6, -0.1],
])
biases = np.array([0.2, -0.8, 0.1])
targets = np.array([1.0, 0.0, 1.0])

# Each row of weights represents one neuron's incoming weights.
weighted_sums = weights @ inputs + biases
outputs = sigmoid(weighted_sums)
squared_error = np.sum((targets - outputs) ** 2)

print("Outputs:", outputs)
print("Sum of squared errors:", squared_error)
