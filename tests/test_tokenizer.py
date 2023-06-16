import torch

from g3po.data import get_tokenizer


def test_mini_tokenizer():
    mini_tokenizer = get_tokenizer("mini")
    assert len(mini_tokenizer.chars) == 65
    encoded = mini_tokenizer.encode("hello world", add_special_tokens=False, return_tensors="pt")
    assert encoded.shape == (11,)


def test_mini_tokenizer2():
    mini_tokenizer = get_tokenizer("mini")
    encoded = mini_tokenizer.encode("\n", add_special_tokens=False, return_tensors="pt")
    assert encoded == torch.tensor(0)


def test_decode():
    test_tokens = [46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42]
    mini_tokenizer = get_tokenizer("mini")
    assert mini_tokenizer.decode(test_tokens) == "hello world"
