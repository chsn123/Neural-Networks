"""A small fully connected neural network implemented with NumPy.

The model uses one hidden layer, sigmoid activations, binary cross-entropy
loss, and gradient descent. The XOR problem is its demonstration dataset.
"""

import numpy as np


def sigmoid(values):
    """Convert arbitrary real numbers into values between 0 and 1."""
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def binary_cross_entropy(predictions, targets):
    """Return the average loss for binary predictions."""
    predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(
        targets * np.log(predictions)
        + (1.0 - targets) * np.log(1.0 - predictions)
    ))


class NeuralNetwork:
    """A network with an input layer, one hidden layer, and one output."""

    def __init__(self, input_size=2, hidden_size=4, seed=42):
        rng = np.random.default_rng(seed)

        # Weights connect each layer; biases allow each neuron to shift.
        self.W1 = rng.normal(0, 1 / np.sqrt(input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        """Compute hidden activations and the network's predictions."""
        hidden = sigmoid(X @ self.W1 + self.b1)
        output = sigmoid(hidden @ self.W2 + self.b2)
        return hidden, output

    def train(self, X, y, epochs=10000, learning_rate=1.0, print_every=2000):
        """Learn weights and biases using backpropagation."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.W1.shape[0]:
            raise ValueError("X must have shape (samples, input_size).")
        if y.shape != (len(X), 1):
            raise ValueError("y must have shape (samples, 1).")
        if len(X) == 0:
            raise ValueError("Training data must not be empty.")
        if np.any((y < 0) | (y > 1)):
            raise ValueError("Targets must be between 0 and 1.")
        if epochs < 1 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive.")

        for epoch in range(1, epochs + 1):
            # 1. Forward pass: make predictions.
            hidden, output = self.forward(X)

            # 2. Backward pass: calculate how each weight affects the loss.
            # For sigmoid + binary cross-entropy, dL/dz = output - y.
            output_error = output - y
            dW2 = hidden.T @ output_error / len(X)
            db2 = np.mean(output_error, axis=0, keepdims=True)

            hidden_error = (output_error @ self.W2.T) * hidden * (1 - hidden)
            dW1 = X.T @ hidden_error / len(X)
            db1 = np.mean(hidden_error, axis=0, keepdims=True)

            # 3. Gradient descent: update parameters in the opposite
            # direction to the loss gradient.
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            if print_every and (epoch == 1 or epoch % print_every == 0):
                current_loss = binary_cross_entropy(self.forward(X)[1], y)
                print(f"Epoch {epoch:5d} | loss = {current_loss:.6f}")

    def predict_proba(self, X):
        """Return probabilities from 0 to 1."""
        X = np.asarray(X, dtype=float)
        return self.forward(X)[1]

    def predict(self, X):
        """Convert probabilities into labels: 0 or 1."""
        return (self.predict_proba(X) >= 0.5).astype(int)


def main():
    # XOR: output 1 when the inputs are different, otherwise output 0.
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    model = NeuralNetwork(input_size=2, hidden_size=4, seed=42)
    print("Before training:")
    print(np.round(model.predict_proba(X).ravel(), 3))

    model.train(X, y, epochs=10000, learning_rate=1.0)

    print("\nAfter training:")
    for inputs, probability, prediction in zip(
        X.astype(int), model.predict_proba(X).ravel(), model.predict(X).ravel()
    ):
        print(f"{inputs.tolist()} -> probability={probability:.4f}, predicted={prediction}")


if __name__ == "__main__":
    main()
