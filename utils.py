import torch
from torch.nn import functional as F


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
