from g3po.tokenizer import get_tokenizer


def test_mini_tokenizer():
    mini_tokenizer = get_tokenizer("mini")
    encoded = mini_tokenizer.encode("hello world", add_special_tokens=False, return_tensors="pt")
    assert encoded.shape == (11,)
