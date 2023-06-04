import os

import toml
import torch
from torch import nn

from data import get_shakespeare_data
from evaluate import run_eval
from model import MultiHeadAttention, create_mask, load_latest_model

hyperparameters = toml.load("hyperparameters.toml")

batch_size = hyperparameters["batch_size"]
eval_interval = hyperparameters["eval_interval"]
input_dimensions = hyperparameters["input_dimensions"]
learning_rate = hyperparameters["learning_rate"]
num_heads = hyperparameters["num_heads"]
num_iters = hyperparameters["num_iters"]
save_interval = hyperparameters["save_interval"]
sequence_length = hyperparameters["sequence_length"]
vocab_size = hyperparameters["vocab_size"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None


# check if model exists and if so, load it
model, optimizer, loaded_iter = load_latest_model(ckpt_dir, device)

if model is None:
    model = MultiHeadAttention(input_dimensions, num_heads, vocab_size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

mask = create_mask(
    sequence_length
)  # not convinced this is the right place to create the mask; this only works if all sequences are the same length
mask = mask.to(device)

for iter in range(num_iters):
    batch_x, batch_y = get_shakespeare_data(sequence_length)
    labels = batch_y.long()

    # Move the data to the device where the model is
    batch_x = batch_x.to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(batch_x, mask=mask)

    # compute loss
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(outputs.view(-1, vocab_size), labels.view(-1))

    # backward pass
    loss.backward()

    # update parameters
    optimizer.step()

    # clear gradients
    optimizer.zero_grad()

    total_iter = iter + loaded_iter

    print(f"{iter=}, {total_iter=}, loss: {loss}")

    if (iter and iter % save_interval == 0) or iter == num_iters - 1:
        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "iter": total_iter,
        }
        torch.save(save_dict, f"checkpoints/model_checkpoint_{total_iter}.pth")
        print("Model saved at iteration", total_iter)

    if total_iter % eval_interval == 0:
        run_eval(model, total_iter)
