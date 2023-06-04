import glob
import os

import torch
from torch import nn
from torch.nn import functional as F

from model import MultiHeadAttention
from tokenizer import get_tokenizer

# ------------hyperparameters----------------
batch_size = 32  # this is for getting started # note this is set in the other file too
input_dimensions = 256
sequence_length = 128
num_heads = 8  # original paper used 8
learning_rate = 3e-5
num_iters = 20
save_interval = 10
vocab_size = 30522  # this is for "bert-base-uncased"
num_tokens_to_generate = 10
# ------------hyperparameters----------------


def find_latest_checkpoint(dir_path):
    """
    I'm going to use ctime to determine which is the latest so I can mess around with file names
    Note: ctime performs differently on Unix vs Windows, but I think this should be OK
    - ctime is creation time on Windows, but change time on Unix
    """
    # Get list of all .pth files in the directory
    list_of_files = glob.glob(os.path.join(dir_path, "*.pth"))
    if not list_of_files:  # If no files found
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def load_latest_model(dir_path, device):
    checkpoint_path = find_latest_checkpoint(dir_path)
    if checkpoint_path is None:
        print("No checkpoint found")
        return None

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MultiHeadAttention(input_dimensions, num_heads, vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dir_path = "./checkpoints"

model = load_latest_model(dir_path, device)

print(model)

test_sentence = "I enjoy walking with my cute dog and"

tokenizer = get_tokenizer()

# Encoding the test sentence
encoded_input = tokenizer.encode(test_sentence, add_special_tokens=False, return_tensors="pt")

# If your model is on GPU, remember to send your input to GPU
if torch.cuda.is_available():
    encoded_input = encoded_input.to("cuda")

response = model.generate(encoded_input, num_tokens_to_generate)

decoded_sequence = tokenizer.decode(response[0])


print(f"Input Sentence: {test_sentence}")
print(f"Decoded Sequence: {decoded_sequence}")
