import glob
import os

import torch
from torch.nn import functional as F


def tensor_to_text(tensor, tokenizer):
    """
    This function takes a tensor representing a batch of sequences and a tokenizer,
    and converts the tensor into human-readable text using the tokenizer.
    """
    # Convert the tensor to a list of sequences
    sequences = tensor.tolist()

    # Decode each sequence into text
    text = [tokenizer.decode_sequence(seq) for seq in sequences]

    return text


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
