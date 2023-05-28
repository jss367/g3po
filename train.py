from transformers import BertTokenizerFast
from .model import MultiHeadAttention

# ------------hyperparameters----------------
batch_size = 32  # this is for getting started # note this is set in the other file too
input_dimensions = 256
sequence_length = 128
num_heads = 8  # original paper used 8
# ------------hyperparameters----------------




mha = MultiHeadAttention(input_dimensions, num_heads, vocab_size)
texts = ["this is my test text", "this is another test text"] * 200

# let's get this going for one batch
texts = texts[:batch_size]



# OK, now let's think about putting the encoded text into a model
x = encoded_text.input_ids

print(x.shape)  # this should be batch_size, sequence_length

# Now I need to go through an embedding layer

# but this is a list, not a tensor...
mha.forward(x)
