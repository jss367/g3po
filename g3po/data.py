#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import os
from typing import Union

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


def get_tokenizer(type="bert"):
    if type == "bert":
        return get_bert_tokenizer()
    elif type == "mini":
        text = load_or_download_data()
        return MiniTokenizer(text)
    raise ValueError(f"Unknown tokenizer type: {type}")


def get_bert_tokenizer():
    # load a pre-trained tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased",
        add_special_tokens=True,
    )
    return tokenizer


class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars

    def encode(self, char):
        return self.chars.index(char)

    def decode(self, index):
        return self.chars[index]

    def encode_sequence(self, sequence):
        return [self.encode(char) for char in sequence]

    def decode_sequence(self, indices):
        return "".join([self.decode(index) for index in indices])


class MiniTokenizer:
    def __init__(self, text):
        """
        I'm going to sort the set so the tokenizer is deterministic and the tokens are in a nice order. Easier to debug.
        """
        self.chars = tuple(sorted(set(text)))

    def encode(self, sentence, add_special_tokens=False, return_tensors="pt"):
        # Here we're assuming `sentence` is a string of characters.
        encoded = [self.chars.index(char) for char in sentence]
        if return_tensors == "pt":
            encoded = torch.tensor(encoded)
            if torch.cuda.is_available():
                encoded = encoded.to("cuda")
        return encoded

    def decode(self, encoded: Union[list[int], torch.Tensor]) -> str:
        """
        Converts a list or tensor of character indices into a sentence.

        Args:
        encoded: The character indices to decode.

        Returns:
        str: The decoded sentence.
        """
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().tolist()  # If on GPU, move back to CPU and convert to list
        decoded = "".join(self.chars[idx] for idx in encoded)
        return decoded


def get_shakespeare_data(sequence_length):
    """
    sequence_length is also called block size

    This returns encoded data
    """
    text = load_or_download_data()

    # now let's tokenize it then we can get embeddings
    # load a pre-trained tokenizer
    tokenizer = get_tokenizer("bert")

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

    return batch, tokenizer


def get_data(type="shakespeare", sequence_length=128):
    if type == "maxi":
        batch, tokenizer = get_shakespeare_data(sequence_length)
    elif type == "mini":
        batch, tokenizer = get_shakespeare_data_small(sequence_length)
    else:
        raise ValueError(f"Unknown data type {type}")

    return batch, tokenizer


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

    TODO: Remove all this stuff that shouldn't be called each step
    """
    text = load_or_download_data()

    chars = tuple(set(text))
    tokenizer = CharTokenizer(chars)

    split_index = int(len(text) * 0.9)
    train_text = text[:split_index]
    val_text = text[split_index:]

    first_block = torch.tensor([tokenizer.encode(c) for c in train_text[:sequence_length]])
    # convert all of train_text into a tensor
    train_tensor = torch.tensor([tokenizer.encode(c) for c in train_text])
    val_tensor = torch.tensor([tokenizer.encode(c) for c in val_text])

    batch_size = 8  # TODO: This should come from the hyperparameters file

    def get_batch(split):
        """
        Break train or val data into context window-sized chunks
        Randomly pull from there WITHOUT replacement
        Once you've emptied that, that's one epoch.

        But don't have to break it into context sized chunks, could just choose anywhere. It's fine for now
        """
        data = train_text if split == "train" else val_text
        start_indices = torch.randint(
            0, len(data) - sequence_length, (batch_size,)
        )  # should this be - (context_window + 1) to save room for the label?
        # end_indices = start_indices + context_window + 1 # +1 for label
        context = torch.stack([train_tensor[i : i + sequence_length] for i in start_indices])
        label = torch.stack([train_tensor[i + 1 : i + sequence_length + 1] for i in start_indices])
        return context, label

    batch = get_batch("train")

    return batch, tokenizer


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


if __name__ == "__main__":
    sequence_length = 128
    data = get_shakespeare_data(sequence_length)
    print(data)
