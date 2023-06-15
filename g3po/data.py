#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import os

import requests
import toml
import torch
from transformers import BertTokenizerFast

# torch.manual_seed(42)
hyperparameters = toml.load("hyperparameters.toml")

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
    tokenizer = get_tokenizer()

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


def get_tokenizer():
    # load a pre-trained tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased",
        add_special_tokens=True,
    )
    return tokenizer


# def create_shakespeare_tokenizer():
#     """
#     This is specifically for the shakespeare dataset.

#     I'm going to tokenize it at the letter level. It would be interesting to do it at the word level at some point too,
#     but this will make it easier to see if it's working or not.
#     """
#     text = load_or_download_data()

#     chars = tuple(set(text))

#     def encoder(char):
#         return chars.index(char)

#     def decoder(index):
#         return chars[index]


def get_data(type="shakespeare", sequence_length=128):
    if type == "maxi":
        return get_shakespeare_data(sequence_length)
    elif type == "mini":
        return get_shakespeare_data_small(sequence_length)
    else:
        raise ValueError(f"Unknown data type {type}")


def get_vocab_size():
    """
    This is specifically for the shakespeare dataset. Going to need to clean this up a bit.
    """
    text = load_or_download_data()
    chars = tuple(set(text))
    vocab_size = len(chars)
    return vocab_size


def get_shakespeare_data_small(sequence_length):
    """
    This uses the shakespeare data with a smaller tokenizer.

    sequence_length is also called block size and context window
    """
    text = load_or_download_data()

    chars = tuple(set(text))

    def encoder(char):
        return chars.index(char)

    def decoder(index):
        return chars[index]

    split_index = int(len(text) * 0.9)
    train_text = text[:split_index]
    val_text = text[split_index:]

    first_block = torch.tensor([encoder(c) for c in train_text[:sequence_length]])
    # convert all of train_text into a tensor
    train_tensor = torch.tensor([encoder(c) for c in train_text])
    val_tensor = torch.tensor([encoder(c) for c in val_text])

    batch_size = 8

    def get_batch(split):
        """
        Break train or val data into context window-sized chunks
        Randomly pull from there WITHOUT replacement
        Once you've emptied that, that's one epoch.

        But don't have to break it into context sized chunks, could just choose anywhere. It's fine
        """
        data = train_text if split == "train" else val_text
        start_indices = torch.randint(
            0, len(data) - sequence_length, (batch_size,)
        )  # should this be - (context_window + 1) to save room for the label?
        # end_indices = start_indices + context_window + 1 # +1 for label
        context = torch.stack([train_tensor[i : i + sequence_length] for i in start_indices])
        label = torch.stack([train_tensor[i + 1 : i + sequence_length + 1] for i in start_indices])
        return context, label

    x, y = get_batch("train")

    return x, y


if __name__ == "__main__":
    sequence_length = 128
    data = get_shakespeare_data(sequence_length)
    print(data)
