import toml
import torch
from torch import nn
from torch.nn import functional as F

from model import MultiHeadAttention
from tokenizer import get_tokenizer
from utils import load_latest_model

hyperparameters = toml.load("Hyperparameters.toml")

batch_size = hyperparameters["batch_size"]
input_dimensions = hyperparameters["input_dimensions"]
learning_rate = hyperparameters["learning_rate"]
num_heads = hyperparameters["num_heads"]
num_iters = hyperparameters["num_iters"]
save_interval = hyperparameters["save_interval"]
sequence_length = hyperparameters["sequence_length"]
vocab_size = hyperparameters["vocab_size"]

num_tokens_to_generate = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dir_path = "./checkpoints"

model, _, _ = load_latest_model(dir_path, device)

model.eval()

print(model)

test_sentence = "I enjoy walking with my cute dog and"

tokenizer = get_tokenizer()

# Encoding the test sentence
encoded_input = tokenizer.encode(test_sentence, add_special_tokens=False, return_tensors="pt")

# If your model is on GPU, remember to send your input to GPU
if torch.cuda.is_available():
    encoded_input = encoded_input.to("cuda")

response = model.generate(encoded_input, num_tokens_to_generate)

decoded_sequence = tokenizer.decode(response[0])


print(f"Input Sentence: {test_sentence}")
print(f"Decoded Sequence: {decoded_sequence}")
