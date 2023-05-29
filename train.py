import torch
from torch import nn

from get_data import get_shakespeare_data
from model import MultiHeadAttention, create_mask

# ------------hyperparameters----------------
batch_size = 32  # this is for getting started # note this is set in the other file too
input_dimensions = 256
sequence_length = 128
num_heads = 8  # original paper used 8
learning_rate = 3e-5
num_iters = 20
save_interval = 10
# ------------hyperparameters----------------


data = get_shakespeare_data(sequence_length)

batch_x, batch_y = data
labels = batch_y.long()
assert labels.shape == (batch_size, sequence_length)
# the data comes out of here in tokenized form, still need to do the embedding
# will do the embedding in the forward loop?
vocab_size = 30522  # this is for "bert-base-uncased"

mha = MultiHeadAttention(input_dimensions, num_heads, vocab_size)

mask = create_mask(sequence_length)  # not convinced this is the right place to create the mask


# now let's get the loss


optimizer = torch.optim.AdamW(mha.parameters(), lr=learning_rate)

for iter in range(num_iters):
    # forward pass
    outputs = mha(batch_x, mask=mask)

    # compute loss
    # loss = torch.nn.functional.cross_entropy(outputs, batch_y)
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(outputs.view(-1, vocab_size), labels.view(-1))

    # backward pass
    loss.backward()

    # update parameters
    optimizer.step()

    # clear gradients
    optimizer.zero_grad()

    # print loss
    print(f"iter {iter} loss {loss}")

    if iter % save_interval == 0:
        save_dict = {
            "model_state_dict": mha.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "iter": iter,
        }
        torch.save(save_dict, f"checkpoints/model_checkpoint_{iter}.pth")
        print("Model saved at iteration", iter)
