"""
I'm probably going to move this to be closer to the data so it's easier to use.

Unclear if I should pass the text to the tokenizer upon initialization or not
"""
import torch
from transformers import BertTokenizerFast

from data import load_or_download_data


def get_tokenizer(type="bert"):
    if type == "bert":
        return get_bert_tokenizer()
    elif type == "mini":
        text = load_or_download_data()
        return MiniTokenizer(text)
    raise ValueError(f"Unknown tokenizer type: {type}")


def get_bert_tokenizer():
    # load a pre-trained tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased",
        add_special_tokens=True,
    )
    return tokenizer


class MiniTokenizer:
    def __init__(self, text):
        self.chars = tuple(set(text))

    def encode(self, sentence, add_special_tokens=False, return_tensors="pt"):
        # Here we're assuming `sentence` is a string of characters.
        encoded = [self.chars.index(char) for char in sentence]
        if return_tensors == "pt":
            encoded = torch.tensor(encoded)
            if torch.cuda.is_available():
                encoded = encoded.to("cuda")
        return encoded

    def decode(self, encoded):
        # Assuming `encoded` is a tensor of indices.
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().tolist()  # If on GPU, move back to CPU and convert to list
        decoded = "".join(self.chars[idx] for idx in encoded)
        return decoded
