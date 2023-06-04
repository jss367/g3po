"""
The goal here is to build GPT-2:
12-layer, 768-hidden, 12-heads, 117M parameters.
OpenAI GPT-2 English model

Should be about 117M parameters

I was doing it all in a MultiHeadedAttention class, but I will make a single self-attention module

"""
import numpy as np
import toml
import torch
from torch import nn
from torch.nn import functional as F

from utils import find_latest_checkpoint, top_p_sampling

hyperparameters = toml.load("Hyperparameters.toml")


input_dimensions = hyperparameters["input_dimensions"]
num_heads = hyperparameters["num_heads"]
vocab_size = hyperparameters["vocab_size"]


def load_latest_model(dir_path, device):
    checkpoint_path = find_latest_checkpoint(dir_path)
    if checkpoint_path is None:
        print("No checkpoint found")
        return None, None, None  # Also return None for optimizer and start_iter

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MultiHeadAttention(input_dimensions, num_heads, vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Initialize the optimizer and load its state
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_iter = checkpoint["iter"]

    return model, optimizer, start_iter


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    The inputs are all of shape batch_size, num_heads, sequence_length, dimensions_per_head (# B,H,S,C--this is dim per head)

    I think probably einsum isn't the fastest way to do this, but it's the easiest to understand
    """
    matmul_qk = torch.einsum(
        "bhid,bhjd->bhij", query, key
    )  # result is batch_size, num_heads, sequence_length, sequence_length
    d_k = query.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)  # B,H,S,S

    # add the mask here
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax = nn.Softmax(dim=-1)(scaled_attention_logits) # Am I going to need to use this to track the gradients?
    attention_weights = torch.nn.functional.softmax(
        scaled_attention_logits, dim=-1
    )  # batch_size, num_heads, dimensions_per_head, dimensions_per_head
    matmul_qkv = torch.einsum("bhij,bhjd->bhid", attention_weights, value)  # B,H,S,C
    return matmul_qkv


def create_mask(size):
    """
    This is a look-ahead mask (maybe change function name?), so normally it has ones below the diagonal and zeros above the diagonal
    But in this case, we want ones above the diagonal so we can multiple them by float(-inf) and then softmax them
    """
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask.unsqueeze(0).unsqueeze(0)  # this is batch_size, 1, 1, sequence_length


class MultiHeadAttention(nn.Module):
    """
    I think I need to add the layers. Not sure exactly where they should go


    """

    def __init__(self, input_dimensions, num_heads, vocab_size):
        super().__init__()

        assert (
            input_dimensions % num_heads == 0
        ), "The number of input dimensions must be evenly divisible by the number of heads"
        self.input_dimensions = input_dimensions  # this is also known as d_model
        self.num_heads = num_heads
        self.dimensions_per_head = int(input_dimensions / num_heads)  # this is also known as d_k
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(
            vocab_size, input_dimensions
        )  # karpathy has vocab_size, vocab_size -- double check at some point

        self.values = nn.Linear(
            self.input_dimensions, self.input_dimensions, bias=False
        )  # should give it in features and out features. Not sure exactly which is which
        self.keys = nn.Linear(self.input_dimensions, self.input_dimensions, bias=False)  # Karparthy says no bias
        self.queries = nn.Linear(self.input_dimensions, self.input_dimensions, bias=False)

        self.feed_forward = nn.Linear(self.input_dimensions, self.input_dimensions)  # this can happen later

        # Going to need a final linear layer right before softmax (see figure 1 of AIAYN)
        self.final_linear = nn.Linear(self.input_dimensions, self.vocab_size)

    def forward(self, x, mask=None):
        """
        The input here can be batch_size, sequence_length.
        I'll take that can put it through the embeddings and that will add the input dimensions


        The input to this model should be a tensor of shape (batch_size, sequence_length, input_dimensions)

        """
        batch_size, sequence_length = x.shape
        # start by embedding the input
        embedded_input = self.embeddings(x)  # output should be batch_size, sequence_length, input_dimensions (B,S,C)

        # take the input and independently do the query and key multiplication?
        queries = self.queries(embedded_input)  # output should be batch_size, sequence_length, input_dimensions
        keys = self.keys(embedded_input)  # output should be batch_size, sequence_length, input_dimensions
        values = self.values(embedded_input)  # output should be batch_size, sequence_length, input_dimensions

        # reshape all the matrices to have an extra dimension for the heads
        queries = queries.reshape(batch_size, num_heads, sequence_length, self.dimensions_per_head)  # B,H,S,C
        keys = keys.reshape(batch_size, num_heads, sequence_length, self.dimensions_per_head)
        values = values.reshape(batch_size, num_heads, sequence_length, self.dimensions_per_head)

        # now apply the attention
        attention_logits = scaled_dot_product_attention(queries, keys, values, mask=mask)
        # more steps to do here

        # add residual connection
        # x = embedded_input + attention

        # now do layer norm
        # x = nn.LayerNorm(x)

        # now we do the feed forward part
        # ff = self.feed_forward(x)

        # another residual connection
        # x = x + ff

        # layer norm
        # x = nn.LayerNorm(x)

        B, H, S, C = attention_logits.shape

        # First, we reshape to [B, S, H, C]
        outputs = attention_logits.transpose(1, 2).contiguous()  # view operates on contiguous tensors

        # Then, we combine all heads into a single vector for each position
        outputs = outputs.view(B, S, H * C)  # B, S, H*C

        # Finally, we apply a linear transformation to reduce the dimension back to d_model
        outputs = self.feed_forward(outputs)  # B, S, input_dimensions

        outputs = self.final_linear(outputs)  # B, S, vocab_size

        return outputs

    def generate(self, start_sequence, num_tokens):
        """
        Does this need the starting sequence or something?

        Start_sequence is an encoded sequence of tokens
        """

        sequence = start_sequence

        for _ in range(num_tokens):
            # Get model prediction
            with torch.no_grad():  # We don't need gradients for inference, so this makes it more memory-efficient
                logits = self(sequence)  # the output is of shape (batch_size, sequence_length, vocab_size)
            # this means we have a word prediction for each word in the sequence. But we only want the last word

            last_word_logits = logits[:, -1, :]  # shape (batch_size, vocab_size); this seems inefficient

            filtered_logits = top_p_sampling(last_word_logits, top_p=0.9)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)

            # now we can append the token to the sequence
            # append to encoded_input somehow
            sequence = torch.cat((sequence, next_token), dim=1)

        return sequence
