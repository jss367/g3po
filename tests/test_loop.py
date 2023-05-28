import torch
from torch import nn

from model import MultiHeadAttention

# ------------hyperparameters---------------- some are in multiple files while I move things around
batch_size = 32  # this is for getting started
input_dimensions = 256
sequence_length = 128
num_heads = 8  # original paper used 8
vocab_size = 30522  # this is for "bert-base-uncased"
# ------------hyperparameters----------------


def estimate_loss(vocab_size):
    """
    The loss after the first iteration should be easy to estimate
    """
    probability_of_correct_word = 1 / vocab_size
    nll = -torch.log(probability_of_correct_word)
    return nll


def test_initial_loss():
    # Set up your model
    model = MultiHeadAttention(input_dimensions, num_heads, vocab_size)
    loss_func = nn.CrossEntropyLoss()

    # Create some random input data
    input = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    target = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))

    # Forward pass and compute loss
    output = model(input)
    loss = loss_func(output.view(-1, vocab_size), target.view(-1))

    # Assert that the loss is approximately what we expect for a randomly initialized model
    assert 10.2 < loss.item() < 10.4


def test_estimate_loss():
    loss = estimate_loss(vocab_size)
    assert 10.2 < loss.item() < 10.4
