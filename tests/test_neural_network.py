"""Run using: python -m unittest discover -s tests"""

import unittest

import numpy as np

from neural_network import NeuralNetwork, binary_cross_entropy, sigmoid


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        self.y = np.array([[0], [1], [1], [0]], dtype=float)

    def test_sigmoid_midpoint(self):
        self.assertAlmostEqual(float(sigmoid(np.array([0.0]))[0]), 0.5)

    def test_forward_output_shape(self):
        model = NeuralNetwork(seed=42)
        _, outputs = model.forward(self.X)
        self.assertEqual(outputs.shape, (4, 1))

    def test_learning_reduces_loss_and_solves_xor(self):
        model = NeuralNetwork(seed=42)
        initial_loss = binary_cross_entropy(model.predict_proba(self.X), self.y)
        model.train(self.X, self.y, epochs=10000, learning_rate=1.0, print_every=0)
        final_loss = binary_cross_entropy(model.predict_proba(self.X), self.y)
        self.assertLess(final_loss, initial_loss)
        np.testing.assert_array_equal(model.predict(self.X), self.y.astype(int))


if __name__ == "__main__":
    unittest.main()
