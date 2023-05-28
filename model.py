"""
The goal here is to build GPT-2:
12-layer, 768-hidden, 12-heads, 117M parameters.
OpenAI GPT-2 English model

Should be about 117M parameters

I was doing it all in a MultiHeadedAttention class, but I will make a single self-attention module

"""
import numpy as np
import torch
from pyxtend import struct
from torch import einsum, nn
from transformers import BertTokenizerFast


class GPT(nn.Module):
    pass


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    The inputs are all of shape batch_size, num_heads, sequence_length, dimensions_per_head

    I think probably einsum isn't the fastest way to do this, but it's the easiest to understand
    """
    matmul_qk = torch.einsum(
        "bhsi,bhsj->bhij", query, key
    )  # result is batch_size, num_heads, dimensions_per_head, dimensions_per_head
    d_k = query.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # add the mask here
    if mask is not None:
        scaled_attention_logits += mask * float("-inf")

    # softmax = nn.Softmax(dim=-1)(scaled_attention_logits) # Am I going to need to use this to track the gradients?
    attention_weights = torch.nn.functional.softmax(
        scaled_attention_logits, dim=-1
    )  # batch_size, num_heads, dimensions_per_head, dimensions_per_head
    matmul_qkv = torch.einsum("bhij,bhsj->bhsi", attention_weights, value)
    return matmul_qkv


def create_mask(size):
    mask = torch.tril(torch.ones(size, size))
    return mask


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

        self.embeddings = nn.Embedding(vocab_size, input_dimensions)

        self.values = nn.Linear(
            self.input_dimensions, self.input_dimensions, bias=False
        )  # should give it in features and out features. Not sure exactly which is which
        self.keys = nn.Linear(self.input_dimensions, self.input_dimensions, bias=False)  # Karparthy says no bias
        self.queries = nn.Linear(self.input_dimensions, self.input_dimensions, bias=False)

        # self.feed_forward = nn.Linear(self.input_dimensions, self.input_dimensions) # this can happen later

    def forward(self, x):
        """
        The input here can be batch_size, sequence_length.
        I'll take that can put it through the embeddings and that will add the input dimensions


        The input to this model should be a tensor of shape (batch_size, sequence_length, input_dimensions)
        """
        B, S = x.shape
        # start by embedding the input
        embedded_input = self.embeddings(x)  # output should be batch_size, sequence_length, input_dimensions (B,S,C)

        # take the input and independently do the query and key multiplication?
        queries = self.queries(embedded_input)  # output should be batch_size, sequence_length, input_dimensions
        keys = self.keys(embedded_input)  # output should be batch_size, sequence_length, input_dimensions
        values = self.values(embedded_input)  # output should be batch_size, sequence_length, input_dimensions

        # reshape all the matrices to have an extra dimension for the heads
        queries = queries.reshape(batch_size, num_heads, sequence_length, self.dimensions_per_head)
        keys = keys.reshape(batch_size, num_heads, sequence_length, self.dimensions_per_head)
        values = values.reshape(batch_size, num_heads, sequence_length, self.dimensions_per_head)

        # now apply the attention
        attention = scaled_dot_product_attention(queries, keys, values)
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


