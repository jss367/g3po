import logging
import os

import toml
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from g3po.data import get_data
from g3po.evaluate import run_eval, save_eval
from g3po.model import MultiHeadAttention, create_mask, load_latest_model

logging.basicConfig(level=logging.INFO)

config = toml.load("configs/mini.toml")
vocab_size = config["vocab_size"]  # this could be check config and if not calculate it

hyperparameters = toml.load(config["hyperparameters"])

batch_size = hyperparameters["batch_size"]
eval_interval = hyperparameters["eval_interval"]
input_dimensions = hyperparameters["input_dimensions"]
learning_rate = hyperparameters["learning_rate"]
num_heads = hyperparameters["num_heads"]
num_iters = hyperparameters["num_iters"]
save_interval = hyperparameters["save_interval"]
sequence_length = hyperparameters["sequence_length"]

# vocab_size = get_vocab_size()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = config["checkpoint_dir"]
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

loss_func = nn.CrossEntropyLoss()

# Set up Tensorboard
writer = SummaryWriter()

for iter in range(num_iters):
    batch_x, batch_y = get_data(config["dataset"], sequence_length)
    labels = batch_y.long()

    # Move the data to the device where the model is
    batch_x = batch_x.to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(batch_x, mask=mask)

    # compute loss
    loss = loss_func(outputs.view(-1, vocab_size), labels.view(-1))

    # backward pass
    loss.backward()

    # update parameters
    optimizer.step()

    # Write loss to the TensorBoard
    writer.add_scalar("Loss/train", loss.item(), iter)

    # clear gradients
    optimizer.zero_grad()

    total_iter = iter + loaded_iter

    logging.info(f"{iter=}, {total_iter=}, loss: {loss}")

    if (iter and iter % save_interval == 0) or iter == num_iters - 1:
        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "iter": total_iter,
        }
        torch.save(save_dict, f"{ckpt_dir}/model_checkpoint_{total_iter}.pth")
        logging.info("Model saved at iteration %s", total_iter)

    if total_iter and total_iter % eval_interval == 0:
        test_sentence, decoded_sequence = run_eval(model, tokenizer_type=config["tokenizer"])
        save_eval(total_iter, test_sentence, decoded_sequence)

writer.close()
