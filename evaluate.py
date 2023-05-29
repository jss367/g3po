import glob
import os

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizerFast

from model import MultiHeadAttention

# ------------hyperparameters----------------
batch_size = 32  # this is for getting started # note this is set in the other file too
input_dimensions = 256
sequence_length = 128
num_heads = 8  # original paper used 8
learning_rate = 3e-5
num_iters = 20
save_interval = 10
vocab_size = 30522  # this is for "bert-base-uncased"
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


def top_p_sampling(logits: torch.Tensor, top_p=0.9, filter_value=-float("inf")):
    """Applies top-p sampling to logits"""

    # Calculate cumulative probabilities of sorted tokens
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # shape (batch_size, vocab_size)

    # Remove tokens with cumulative probability above the threshold (top_p)
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Set logits of tokens to remove to a large negative value
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[0, indices_to_remove] = filter_value
    return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dir_path = "./checkpoints"  # Replace with your actual directory path

model = load_latest_model(dir_path, device)

print(model)

test_sentence = "I enjoy walking with my cute dog and"

tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased",
    add_special_tokens=True,
)

# Encoding the test sentence
encoded_input = tokenizer.encode(test_sentence, add_special_tokens=False, return_tensors="pt")

# If your model is on GPU, remember to send your input to GPU
if torch.cuda.is_available():
    encoded_input = encoded_input.to("cuda")

# Get model prediction
with torch.no_grad():  # We don't need gradients for inference, so this makes it more memory-efficient
    logits = model(encoded_input)  # the output is of shape (batch_size, sequence_length, vocab_size)
# this means we have a word prediction for each word in the sequence. But we only want the last word

last_word_logits = logits[:, -1, :]  # shape (batch_size, vocab_size)


filtered_logits = top_p_sampling(last_word_logits, top_p=0.9)
probabilities = nn.functional.softmax(filtered_logits, dim=-1)
next_token = torch.multinomial(probabilities, 1)

next_word = tokenizer.decode(next_token[0])


print(f"Input Sentence: {test_sentence}")
print(f"Predicted word: {next_word}")
