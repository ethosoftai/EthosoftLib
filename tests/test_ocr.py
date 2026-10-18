import shutil

import pytest

from ethosoftlib import ocr
from ethosoftlib.rag import Index, RagError

needs_poppler = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("pdfinfo", "pdftotext", "pdftoppm")),
    reason="Poppler (poppler-utils) is not installed",
)


def make_pdf(path, pages):
    """Minimal PDF: each item is a page's text, or None for an image-only page."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", None,
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        lines = (text or "").split("\n") if text else []
        ops = "".join(f"BT /F1 12 Tf 72 {720 - 16 * i} Td ({line}) Tj ET\n" for i, line in enumerate(lines))
        if text is None:  # a filled rectangle stands in for a scanned image
            ops = "0.5 g 72 500 300 200 re f\n"
        stream = ops.encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")
        content = len(objects)
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content)
        kids.append(len(objects))
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % kid for kid in kids), len(kids))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(bytes(out))
    return path


@needs_poppler
def test_text_layer_pages_are_read_in_order(tmp_path):
    pdf = make_pdf(tmp_path / "rapor.pdf", ["Birinci sayfa: Ankara raporu", "Ikinci sayfa\nKuantizasyon notlari"])
    doc = ocr.read(pdf)
    assert ocr.page_count(pdf) == 2
    assert [(p.number, p.source) for p in doc.pages] == [(1, "text-layer"), (2, "text-layer")]
    assert doc.pages[1].text == "Ikinci sayfa\nKuantizasyon notlari"
    assert doc.text == "Birinci sayfa: Ankara raporu\n\nIkinci sayfa\nKuantizasyon notlari"
    assert doc.needs_ocr == ()


@needs_poppler
def test_scanned_pages_are_reported_not_returned_empty(tmp_path):
    pdf = make_pdf(tmp_path / "tarama.pdf", ["Metin katmani olan sayfa burada", None, "x"])
    with pytest.raises(ocr.OcrUnavailable, match="page\\(s\\) 2, 3") as error:
        ocr.read(pdf)
    assert error.value.pages == (2, 3)
    doc = ocr.read(pdf, allow_missing=True)
    assert doc.needs_ocr == (2, 3)
    assert doc.text == "Metin katmani olan sayfa burada"
    assert doc.to_dict()["needs_ocr"] == [2, 3]


@needs_poppler
def test_render_page_returns_png(tmp_path):
    pdf = make_pdf(tmp_path / "tarama.pdf", ["Sayfa bir", None])
    png = ocr.render_page(pdf, 2, dpi=72)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ocr.OcrError, match="no page 3"):
        ocr.render_page(pdf, 3)
    with pytest.raises(ValueError):
        ocr.render_page(pdf, 1, dpi=5)


def test_images_and_non_pdfs(tmp_path):
    image = tmp_path / "fis.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    with pytest.raises(ocr.OcrUnavailable, match="image"):
        ocr.read(image)
    assert ocr.read(image, allow_missing=True).needs_ocr == (1,)
    text = tmp_path / "not.pdf"
    text.write_text("PDF degil")
    with pytest.raises(ocr.OcrError, match="Not a PDF"):
        ocr.read(text)
    with pytest.raises(ocr.OcrError, match="Not a file"):
        ocr.read(tmp_path / "yok.pdf")


@needs_poppler
def test_rag_indexes_pdf_pages_and_skips_unreadable_pdfs(tmp_path):
    make_pdf(tmp_path / "rapor.pdf", ["Giris sayfasi genel bilgiler", None, "Kuantizasyon ayarlari burada"])
    (tmp_path / "bozuk.pdf").write_bytes(b"%PDF-1.4 bozuk")
    (tmp_path / "not.md").write_text("Kuantizasyon notu", encoding="utf-8")
    index = Index()
    index.add_directory(tmp_path)
    hits = index.search("kuantizasyon", k=5)
    assert {(h.source, h.metadata.get("page")) for h in hits} == {("rapor.pdf", 3), ("not.md", None)}
    with pytest.raises(RagError, match="bozuk.pdf"):
        Index().add_file(tmp_path / "bozuk.pdf")


def test_ocr_provider_is_registered_lazily():
    from ethosoftlib import providers
    assert providers.describe("document", "ocr").factory == "ethosoftlib.ocr:read"
