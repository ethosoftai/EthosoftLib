"""Real original Rust NDSRF004 checks (skip only when binary isn't configured)."""
from os import environ
import pytest

@pytest.mark.skipif(not environ.get("ETHOSOFT_NEDO_LIBRARY"),
                    reason="set ETHOSOFT_NEDO_LIBRARY to compiled Rust cdylib")
def test_exact_original_nedo_tokenizer():
    from ethosoftlib.nedo import Tokenizer, EXPECTED_SHA256
    tok = Tokenizer()
    assert tok.vocab_size == 32000
    assert tok.vocab_sha256 == EXPECTED_SHA256
    samples = [
        "Merhaba, nasılsın?",
        "İstanbul'daki öğrenciler geldi.",
        "Türkçe ş, ğ, ü, ı, ö, ç ve emoji 😀",
        "def greet(x):\n    return x + 1\n",
        "",
    ]
    for sample in samples:
        encoded = tok.encode(sample)
        assert all(0 <= i < 32000 for i in encoded)
        assert tok.decode(encoded) == sample
        assert tok.decode(tok.encode(sample, add_special_tokens=True)) == sample
    assert tok.decode_bytes(tok.encode(b"\xff\x00\x80")) == b"\xff\x00\x80"
