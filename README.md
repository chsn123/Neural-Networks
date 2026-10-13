# Neural Network Experiment

I've been learning about neural networks and wanted to see how they work without using something like PyTorch.

I started with a small bit of code that takes some inputs, weights and biases and calculates an output. This project builds on that by trying to train a network to solve XOR.

XOR is pretty simple: the answer is 1 when the two inputs are different, and 0 when they're the same.

## Files

- `original_experiment.py` - my first experiment with weights, biases and the sigmoid function
- `neural_network.py` - a bigger version with a hidden layer and training
- `tests/` - a few tests to check that the code works

## Running it

Install NumPy:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python neural_network.py
```

It prints the loss while training and shows its predictions afterwards. You can also run `original_experiment.py` to see the earlier version.

## What I'm learning

The main idea I'm trying to understand is how a network changes its weights when it gets an answer wrong. I understand the forward pass better than the backpropagation part at the moment, so that's what I want to work on next.

I used AI help to expand the original experiment into a trainable network, and I'm going through the code to understand how it works.
