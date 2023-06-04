#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import os

import requests
import toml
import torch
from transformers import BertTokenizerFast

hyperparameters = toml.load("Hyperparameters.toml")

batch_size = hyperparameters["batch_size"]


def load_or_download_data():
    filename = "data/shakespeare.txt"

    if os.path.isfile(filename):
        # File already exists, so load it
        with open(filename, "r") as f:
            text = f.read()

    else:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        # File does not exist, so download it
        response = requests.get(url)
        text = response.text

        # Save the file to disk
        with open(filename, "w") as f:
            f.write(text)

    return text


def get_shakespeare_data(sequence_length):
    """
    sequence_length is also called block size

    This returns encoded data
    """
    text = load_or_download_data()

    # now let's tokenize it then we can get embeddings
    # load a pre-trained tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased",
        add_special_tokens=True,
    )

    # vocab_size = len(tokenizer.get_vocab()) # 30522 for "bert-base-uncased"

    # encoded_text = tokenizer(
    #     text, padding="max_length", truncation=True, max_length=sequence_length, return_tensors="pt"
    # )
    # let's tokenize the whole thing
    encoded_text = tokenizer(text, padding="max_length", max_length=sequence_length, return_tensors="pt")

    # convert the tensor to type long
    data = encoded_text.input_ids.squeeze().long()

    train_fraction = 0.9
    train_cutoff = int(train_fraction * len(data))

    train_data = data[:train_cutoff]
    test_data = data[train_cutoff:]

    # now we get a batch from either the train or test data
    def get_batch(split="train"):
        data = train_data if split == "train" else test_data
        # get a random starting point
        start_indices = torch.randint(len(data) - sequence_length, (batch_size,))

        batch_x = torch.stack([data[i : i + sequence_length] for i in start_indices])  # batch_size, sequence_length
        # the target is the next character
        batch_y = torch.stack(
            [data[i + 1 : i + 1 + sequence_length] for i in start_indices]
        )  # batch_size, sequence_length
        return batch_x, batch_y

    batch = get_batch()

    return batch


if __name__ == "__main__":
    sequence_length = 128
    data = get_shakespeare_data(sequence_length)
    print(data)
