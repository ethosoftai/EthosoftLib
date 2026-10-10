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
        print("Native model response:", repr(response), flush=True)
        assert response.strip(), "The model returned an empty response"
        assert len(model.history) == 2



@pytest.mark.skipif(
    not (os.environ.get("ETHOSOFT_MERCAN_LIBRARY") and
         os.environ.get("ETHOSOFT_MERCAN_TEST_MODEL")),
    reason="set ETHOSOFT_MERCAN_LIBRARY and ETHOSOFT_MERCAN_TEST_MODEL",
)
def test_real_native_stream_info_and_kv_reset():
    """Exercise the actual C++ model, not only mock tokenizer fragments."""
    import asyncio
    from ethosoftlib.mercan import Model
    with Model.from_pretrained(
        Path(os.environ["ETHOSOFT_MERCAN_TEST_MODEL"]),
        library_path=os.environ["ETHOSOFT_MERCAN_LIBRARY"],
        device="cpu",
        n_ctx=4096,
    ) as model:
        prompt = "Merhaba, nasılsın?"
        kwargs = dict(max_tokens=16, seed=42, temperature=0.7, save_history=False)
        expected = model.chat(prompt, **kwargs)
        received = "".join(model.stream_chat(prompt, **kwargs))
        assert received == expected
        assert received.strip()
        assert model.history == ()
        info = model.info()
        assert info["architecture"] == "nedolm"
        assert info["vocab_size"] == 32000

        async def check_async():
            assert await model.async_chat(prompt, **kwargs) == expected
            parts = []
            async for fragment in model.astream_chat(prompt, **kwargs):
                parts.append(fragment)
            assert "".join(parts) == expected
        asyncio.run(check_async())

        result = model.benchmark(prompt, runs=1, max_tokens=4)
        assert result["runs"] == 1
        assert result["average_seconds"] >= 0

        with model._model.context(n_ctx=4096) as context:
            tokens = model._model.tokenize("Merhaba", parse_special=True)
            context.decode(tokens)
            first = context.greedy_token()
            assert context.tokens_used == len(tokens)
            context.reset()
            assert context.tokens_used == 0
            context.decode(tokens)
            assert context.greedy_token() == first
