"""True native end-to-end test, skipped unless a built shared library is available."""
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not (os.environ.get("ETHOSOFT_MERCAN_LIBRARY") and
         os.environ.get("ETHOSOFT_MERCAN_TEST_MODEL")),
    reason="set ETHOSOFT_MERCAN_LIBRARY and ETHOSOFT_MERCAN_TEST_MODEL",
)
def test_real_mercan_model_chat():
    from ethosoftlib.mercan import Model

    with Model.from_pretrained(
        Path(os.environ["ETHOSOFT_MERCAN_TEST_MODEL"]),
        library_path=os.environ["ETHOSOFT_MERCAN_LIBRARY"],
        device="cpu",
        n_ctx=4096,
    ) as model:
        response = model.chat(
            "Merhaba, nasılsın?", temperature=0.7,
            max_tokens=16, seed=42,
        )
        assert isinstance(response, str)
        assert len(response) >= 0
        assert len(model.history) == 2
