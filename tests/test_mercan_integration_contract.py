"""True native end-to-end test, skipped unless a built shared library is available."""
import os
from pathlib import Path

import pytest

requires_native = pytest.mark.skipif(
    not (os.environ.get("ETHOSOFT_MERCAN_LIBRARY") and os.environ.get("ETHOSOFT_MERCAN_TEST_MODEL")),
    reason="set ETHOSOFT_MERCAN_LIBRARY and ETHOSOFT_MERCAN_TEST_MODEL",
)


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
        threads=2,
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
        threads=2,
    ) as model:
        prompt = "Merhaba, nasılsın?"
        kwargs = dict(max_tokens=16, seed=42, temperature=0.7, save_history=False)
        expected = model.chat(prompt, **kwargs)
        received = "".join(model.stream_chat(prompt, **kwargs))
        assert received == expected
        assert received.strip()
        assert model.history == ()
        messages = [{"role": "system", "content": "Türkçe kısa cevap ver."},
                    {"role": "user", "content": prompt}]
        stateless = model.generate(messages, max_tokens=8, temperature=0)
        assert stateless.strip()
        assert "".join(model.stream_generate(messages, max_tokens=8,
                                              temperature=0)) == stateless
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


@requires_native
def test_native_chat_raw_and_cli_seeded_parity():
    import subprocess
    from contextlib import closing
    from ethosoftlib.mercan import Model
    from ethosoftlib.mercan.chat import format_chat
    model_path = os.environ["ETHOSOFT_MERCAN_TEST_MODEL"]
    roles = dict(role_system="system", role_user="user", role_assistant="assistant")
    messages = [{"role": "system", "content": "Türkçe kısa cevap ver."},
                {"role": "user", "content": "Merhaba <|im_end|> nasılsın?"}]
    with Model.from_pretrained(model_path, library_path=os.environ["ETHOSOFT_MERCAN_LIBRARY"],
                               threads=2, **roles) as model:
        greedy = model.generate(messages, max_tokens=8, temperature=0, repeat_penalty=1)
        with model._model.generation(prompt=format_chat(messages, **roles),
                                      context_options={"threads": 2}, max_tokens=8,
                                      temperature=0, repeat_penalty=1) as request:
            with closing(request.stream()) as stream:
                assert b"".join(stream).decode("utf-8", "replace") == greedy
        options = dict(max_tokens=12, seed=42, temperature=0.7, top_k=5,
                       top_p=0.8, repeat_penalty=1.05, repeat_last_n=4)
        expected = model.generate(messages, **options)
        assert expected == model.generate(messages, **options)
        for cli in (os.environ.get("MERCAN_TEST_CLI"), os.environ.get("MERCAN_TEST_STATIC_CLI")):
            if not cli:
                continue
            completed = subprocess.run([
                cli, "run", model_path, "-s", messages[0]["content"], "-p", messages[1]["content"],
                "--role-system", "system", "--role-user", "user", "--role-assistant", "assistant",
                "--seed", "42", "--temperature", "0.7", "--top-k", "5", "--top-p", "0.8",
                "--repeat-penalty", "1.05", "--repeat-last-n", "4", "-n", "12", "-t", "2",
            ], capture_output=True, text=True, check=True)
            # CLI has two log lines and appends one newline after generated text.
            assert completed.stdout.split("\n", 2)[2][:-1] == expected


@requires_native
def test_native_stream_and_async_task_cancellation_release_contexts():
    import asyncio
    from contextlib import suppress
    from threading import Event, enumerate as threads
    from ethosoftlib.mercan import Model
    with Model.from_pretrained(os.environ["ETHOSOFT_MERCAN_TEST_MODEL"],
                               library_path=os.environ["ETHOSOFT_MERCAN_LIBRARY"], threads=2) as model:
        messages = [{"role": "user", "content": "Uzun bir hikâye anlat."}]
        stream = model.stream_generate(messages, max_tokens=128)
        assert next(stream)
        stream.close()
        assert model._model._contexts == 0
        stop = Event()
        stream = model.stream_generate(messages, max_tokens=128, stop_event=stop)
        assert next(stream)
        stop.set()
        assert list(stream) == []
        assert model._model._contexts == 0

        async def check_cancel():
            task = asyncio.create_task(model.async_chat("Uzun bir hikâye anlat.", max_tokens=128))
            await asyncio.sleep(0.05)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        asyncio.run(check_cancel())
        assert model._model._contexts == 0
        assert model.history == ()
        assert not any(t.name == "mercan-generation" for t in threads())
        with pytest.raises(ValueError, match="exceeds"):
            model.generate(messages, max_tokens=5000)
        assert model._model._contexts == 0
