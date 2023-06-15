import toml

from g3po.data import get_shakespeare_data

hyperparameters = toml.load("Hyperparameters.toml")

batch_size = hyperparameters["batch_size"]
sequence_length = hyperparameters["sequence_length"]


def test_get_shakespeare_data():
    data = get_shakespeare_data(sequence_length)

    _, batch_y = data
    labels = batch_y.long()
    assert labels.shape == (batch_size, sequence_length)
