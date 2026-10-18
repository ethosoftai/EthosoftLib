"""Document text extraction; OCR models plug in here later.

    >>> from ethosoftlib import ocr
    >>> doc = ocr.read("rapor.pdf")
    >>> doc.text                       # every page that has a text layer
    >>> doc.needs_ocr                  # page numbers that are scanned images
    >>> png = ocr.render_page("rapor.pdf", 3)   # input for an OCR model

PDF text layers are read exactly with Poppler's ``pdftotext``; pages without
one are reported instead of being returned empty, and ``render_page`` turns
them into PNG images with ``pdftoppm``. No OCR model ships yet: images and
scanned pages raise :class:`OcrUnavailable` unless ``allow_missing=True``.
Poppler is the only requirement (``apt install poppler-utils``).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from ..core.errors import EthosoftError

__all__ = [
    "Document", "IMAGE_SUFFIXES", "OcrError", "OcrUnavailable", "Page",
    "page_count", "read", "render_page",
]

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"})
# A page whose text layer has fewer visible characters is treated as scanned.
MIN_PAGE_CHARS = 16
TIMEOUT = 120


class OcrError(EthosoftError):
    """Reading or rendering a document failed."""


class OcrUnavailable(OcrError):
    """The document needs OCR, but no OCR engine is available yet."""

    def __init__(self, message: str, pages: tuple[int, ...] = ()) -> None:
        super().__init__(message)
        self.pages = pages


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    source: str  # "text-layer", or "missing" when the page needs OCR

    def to_dict(self) -> dict[str, object]:
        return {"number": self.number, "text": self.text, "source": self.source}


@dataclass(frozen=True)
class Document:
    path: str
    pages: tuple[Page, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def needs_ocr(self) -> tuple[int, ...]:
        return tuple(page.number for page in self.pages if page.source == "missing")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "pages": [page.to_dict() for page in self.pages],
                "needs_ocr": list(self.needs_ocr)}


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise OcrError(f"{name} not found; install Poppler (apt install poppler-utils).")
    return path


def _run(args: list[str]) -> bytes:
    try:
        result = subprocess.run(args, capture_output=True, timeout=TIMEOUT, check=False)
    except subprocess.TimeoutExpired as exc:
        raise OcrError(f"{Path(args[0]).name} timed out after {TIMEOUT}s") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise OcrError(f"{Path(args[0]).name} failed: {detail[-1] if detail else result.returncode}")
    return result.stdout


def _pdf(path: str | os.PathLike[str]) -> Path:
    file = Path(path).expanduser()
    if not file.is_file():
        raise OcrError(f"Not a file: {file}")
    with file.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise OcrError(f"Not a PDF: {file}")
    return file


def page_count(path: str | os.PathLike[str]) -> int:
    file = _pdf(path)
    for line in _run([_tool("pdfinfo"), str(file)]).decode("utf-8", "replace").splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1])
    raise OcrError(f"pdfinfo reported no page count for {file}")


def _visible(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def read(path: str | os.PathLike[str], *, allow_missing: bool = False) -> Document:
    """Text of a PDF page by page. Scanned pages and images need OCR, which
    raises :class:`OcrUnavailable` unless ``allow_missing`` is set; then they
    are returned as ``source="missing"`` pages without text."""
    file = Path(path).expanduser()
    if file.suffix.lower() in IMAGE_SUFFIXES:
        if not file.is_file():
            raise OcrError(f"Not a file: {file}")
        if allow_missing:
            return Document(str(file), (Page(1, "", "missing"),))
        raise OcrUnavailable(f"{file.name} is an image; no OCR engine is available yet.", (1,))
    file = _pdf(file)
    count = page_count(file)
    # Pages are separated by form feeds, one after every page.
    raw = _run([_tool("pdftotext"), "-enc", "UTF-8", str(file), "-"]).decode("utf-8", "replace")
    texts = raw.split("\f")[:count]
    texts += [""] * (count - len(texts))
    pages = tuple(
        Page(number, text.strip(), "text-layer") if _visible(text) >= MIN_PAGE_CHARS
        else Page(number, "", "missing")
        for number, text in enumerate(texts, start=1)
    )
    document = Document(str(file), pages)
    if document.needs_ocr and not allow_missing:
        missing = document.needs_ocr
        shown = ", ".join(map(str, missing[:10])) + (" ..." if len(missing) > 10 else "")
        raise OcrUnavailable(
            f"{file.name}: page(s) {shown} have no text layer and need OCR, which is "
            "not available yet. Pass allow_missing=True to read the other pages.",
            missing,
        )
    return document


def render_page(path: str | os.PathLike[str], page: int, *, dpi: int = 300) -> bytes:
    """PNG image of one PDF page (1-based), as an OCR model's input."""
    file = _pdf(path)
    if not 1 <= page <= page_count(file):
        raise OcrError(f"{file.name} has no page {page}")
    if not 36 <= dpi <= 1200:
        raise ValueError("dpi must be in 36..1200")
    with tempfile.TemporaryDirectory(prefix="ethosoftlib-ocr-") as folder:
        # pdftoppm has no stdout mode for PNG; it appends ".png" to this root.
        root = Path(folder) / "page"
        _run([_tool("pdftoppm"), "-png", "-r", str(dpi), "-f", str(page), "-l", str(page),
              "-singlefile", str(file), str(root)])
        return root.with_suffix(".png").read_bytes()
