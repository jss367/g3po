from transformers import BertTokenizerFast

from get_data import get_shakespeare_data
from model import MultiHeadAttention, create_mask

# ------------hyperparameters----------------
batch_size = 32  # this is for getting started # note this is set in the other file too
input_dimensions = 256
sequence_length = 128
num_heads = 8  # original paper used 8
# ------------hyperparameters----------------


data = get_shakespeare_data(sequence_length)

batch_x, batch_y = data

# the data comes out of here in tokenized form, still need to do the embedding
# will do the embedding in the forward loop?
vocab_size = 30522  # this is for "bert-base-uncased"

mha = MultiHeadAttention(input_dimensions, num_heads, vocab_size)

mask = create_mask(sequence_length)  # not convinced this is the right place to create the mask

mha.forward(batch_x, mask=mask)

# now let's get the loss
