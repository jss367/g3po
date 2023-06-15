import logging
import os

import toml
import torch

from g3po.tokenizer import get_tokenizer

hyperparameters = toml.load("hyperparameters.toml")

batch_size = hyperparameters["batch_size"]
input_dimensions = hyperparameters["input_dimensions"]
learning_rate = hyperparameters["learning_rate"]
num_heads = hyperparameters["num_heads"]
num_iters = hyperparameters["num_iters"]
save_interval = hyperparameters["save_interval"]
sequence_length = hyperparameters["sequence_length"]
vocab_size = hyperparameters["vocab_size"]

num_tokens_to_generate = 10

eval_dir_path = "./evals"
os.makedirs(eval_dir_path, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)


def run_eval(model, tokenizer_type="bert", **kwargs):
    model.eval()

    test_sentence = "\n"
    tokenizer = get_tokenizer(tokenizer_type)

    # Encoding the test sentence
    encoded_input = tokenizer.encode(test_sentence, add_special_tokens=False, return_tensors="pt").to(device)
    encoded_batch = encoded_input.unsqueeze(1)
    response = model.generate(encoded_batch, num_tokens_to_generate)

    decoded_sequence = tokenizer.decode(response[0])

    logging.info(f"Input Sentence: {test_sentence}")
    logging.info(f"Decoded Sequence: {decoded_sequence}")

    return test_sentence, decoded_sequence


def save_eval(iteration, test_sentence, decoded_sequence):
    # Save the evaluation to a file
    with open(f"{eval_dir_path}/eval_at_iter_{iteration}.txt", "w") as f:
        f.write(f"Input Sentence: {test_sentence}\n")
        f.write(f"Decoded Sequence: {decoded_sequence}\n")
