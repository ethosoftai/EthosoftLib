"""Download only the Mercan model artifact; native runtime downloads are never implicit."""
from __future__ import annotations

from pathlib import Path
import re

_HF_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve_model_file(
    model_id: str | Path, *, filename: str = "model.mercan",
    revision: str | None = None, cache_dir: str | Path | None = None,
    local_files_only: bool = False, token: str | bool | None = None,
) -> Path:
    """Resolve a local .mercan file or download one file into Hugging Face's cache.

    Remote ID forms: owner/repo or owner/repo:filename.mercan.
    The optional 'hub' extra is needed only for remote IDs.
    """
    path = Path(model_id).expanduser()
    if path.is_file():
        if path.suffix != ".mercan":
            raise ValueError("Local model file must end with .mercan")
        return path.resolve()

    value = str(model_id)
    if value.startswith((".", "/", "~")) or isinstance(model_id, Path):
        raise FileNotFoundError(f"Local Mercan model does not exist: {path}")
    if ":" in value:
        if filename != "model.mercan":
            raise ValueError("Use either :filename in model_id or filename=, not both")
        value, filename = value.split(":", 1)
    parts = value.split("/")
    if len(parts) != 2 or not all(_HF_COMPONENT.fullmatch(x) for x in parts):
        raise ValueError("Hugging Face model_id must have the form owner/repo")
    if not _HF_COMPONENT.fullmatch(filename) or not filename.endswith(".mercan"):
        raise ValueError("Only a single filename ending with .mercan is allowed")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Hugging Face support is optional. Install with: pip install 'ethosoftlib[hub]'"
        ) from exc
    downloaded = hf_hub_download(
        repo_id=value, filename=filename, repo_type="model",
        revision=revision, cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only, token=token,
    )
    result = Path(downloaded)
    if not result.is_file():
        raise FileNotFoundError(f"Hub returned a missing Mercan artifact: {result}")
    return result
