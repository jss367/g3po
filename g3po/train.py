import logging
import os

import toml
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from g3po.data import get_data
from g3po.evaluate import run_eval, save_eval
from g3po.model import MultiHeadAttention, create_mask, load_latest_model

# Set logging level
logging.basicConfig(level=logging.INFO)

config = toml.load("configs/mini.toml")

hyperparameters = toml.load(config["hyperparameters"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = config["checkpoint_dir"]
os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None

# check if model exists and if so, load it
model, optimizer, loaded_iter = load_latest_model(ckpt_dir, device)

if model is None:
    model = MultiHeadAttention(hyperparameters["input_dimensions"], hyperparameters["num_heads"], config["vocab_size"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["learning_rate"])

mask = create_mask(hyperparameters["sequence_length"]).to(
    device
)  # not convinced this is the right place to create the mask; this only works if all sequences are the same length

# Set up Tensorboard
writer = SummaryWriter()

# Define loss function outside the loop
loss_func = nn.CrossEntropyLoss()

for iter_ in range(hyperparameters["num_iters"]):
    batch_x, batch_y = get_data(config["dataset"], hyperparameters["sequence_length"])
    labels = batch_y.long()

    # Move the data to the device where the model is
    batch_x = batch_x.to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(batch_x, mask=mask)

    # compute loss
    loss = loss_func(outputs.view(-1, config["vocab_size"]), labels.view(-1))

    # backward pass and parameter update
    loss.backward()
    optimizer.step()

    # Write loss to the TensorBoard
    writer.add_scalar("Loss/train", loss.item(), iter_)

    # clear gradients
    optimizer.zero_grad()

    total_iter_ = iter_ + loaded_iter

    logging.info(f"{iter_=}, {total_iter_=}, loss: {loss}")

    if (iter_ and iter_ % hyperparameters["save_interval"] == 0) or iter_ == hyperparameters["num_iters"] - 1:
        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "iter": total_iter_,
        }
        torch.save(save_dict, f"{ckpt_dir}/model_checkpoint_{total_iter_}.pth")
        logging.info(f"Model saved at iteration {total_iter_}")

    if total_iter_ and total_iter_ % hyperparameters["eval_interval"] == 0:
        test_sentence, decoded_sequence = run_eval(model, tokenizer_type=config["tokenizer"])
        save_eval(total_iter_, test_sentence, decoded_sequence)

# Close the TensorBoard writer
writer.close()
