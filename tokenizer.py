from transformers import BertTokenizerFast


def get_tokenizer():
    # load a pre-trained tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased",
        add_special_tokens=True,
    )
    return tokenizer
