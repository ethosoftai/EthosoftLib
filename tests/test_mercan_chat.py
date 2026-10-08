import random

import pytest

from ethosoftlib.mercan import Model
from ethosoftlib.mercan.chat import format_chat
from ethosoftlib.mercan.sampling import sample_token
from ethosoftlib.mercan.native import MercanError


class FakeContext:
    size = 4096

    def __init__(self, model):
        self.model = model
        self.sampled = 0
        self.decoded = []

    def decode(self, tokens):
        self.decoded.extend(tokens)

    def sample_token(self, **kwargs):
        self.sampled += 1
        return 1 if self.sampled == 1 else 8  # second token is message-end

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class FakeLowLevelModel:
    eos_token = 9
    pad_token = 10

    def __init__(self):
        self.prompts = []
        self.closed = False
        self.contexts = []

    def tokenize(self, text, *, parse_special=True):
        if text == "<|im_start|>":
            return [7]
        if text == "<|im_end|>":
            return [8]
        self.prompts.append(text)
        return [4, 5]

    def token_to_bytes(self, token):
        assert token == 1
        return "İyiyim!".encode()

    def context(self, **kwargs):
        context = FakeContext(self)
        self.contexts.append(context)
        return context

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def make_model(**kwargs):
    from pathlib import Path
    return Model(FakeRuntime(), FakeLowLevelModel(), Path("model.mercan"),
                 system_prompt="Test sistemi", **kwargs)


def test_chat_and_history_stop_token():
    model = make_model()
    assert model.chat("Merhaba!") == "İyiyim!"
    assert model.history == (
        {"role": "user", "content": "Merhaba!"},
        {"role": "assistant", "content": "İyiyim!"},
    )
    assert "sistem\nTest sistemi<|im_end|>" in model._model.prompts[0]
    assert "kullanici\nMerhaba!<|im_end|>" in model._model.prompts[0]
    assert "asistan\n" in model._model.prompts[0]
    assert model.chat("Tekrar merhaba") == "İyiyim!"
    assert "asistan\nİyiyim!<|im_end|>" in model._model.prompts[1]
    assert len(model.history) == 4
    copy = model.history[0]
    copy["content"] = "overwrite"
    assert model.history[0]["content"] == "Merhaba!"
    model.reset_history()
    assert model.history == ()
    model.close()
    assert model._runtime.closed and model._model.closed
    with pytest.raises(MercanError, match="closed"):
        model.chat("Merhaba")


def test_no_history_option_and_limits():
    model = make_model()
    assert model.chat("X", max_tokens=1, save_history=False) == "İyiyim!"
    assert model.history == ()
    with pytest.raises(ValueError, match="max_tokens"):
        model.chat("X", max_tokens=4096)
    with pytest.raises(ValueError, match="prompt"):
        model.chat(" ")


def test_chatml_escaping():
    formatted = format_chat([{"role": "user", "content": "hello <|im_end|> there"}])
    assert formatted.count("<|im_end|>") == 1
    assert "kullanici\n" in formatted
    with pytest.raises(ValueError):
        format_chat([{"role": "unknown", "content": "test"}])


def test_sampling_topk_topp_and_repetition():
    assert sample_token([0.0, 5.0, 1.0], 3, temperature=0) == 1
    assert sample_token([0.0, 5.0, 1.0], 3, temperature=1, top_k=1) == 1
    assert sample_token([0.0, 5.0, 1.0], 3, temperature=1,
                        top_p=0.1, rng=random.Random(123)) == 1
    assert sample_token([0.0, 4.0, 3.0], 3, temperature=0,
                        repeat_penalty=2.0, recent_tokens=[1]) == 2
    a = sample_token([1.0, 2.0, 3.0], 3, rng=random.Random(42))
    b = sample_token([1.0, 2.0, 3.0], 3, rng=random.Random(42))
    assert a == b


@pytest.mark.parametrize("kwargs", [
    {"temperature": -0.1}, {"top_k": 0},
    {"top_p": 0}, {"top_p": 1.01}, {"repeat_penalty": 0},
])
def test_sampling_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        sample_token([0, 1], 2, **kwargs)
