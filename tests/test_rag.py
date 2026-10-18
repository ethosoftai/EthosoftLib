import pytest

from ethosoftlib import providers
from ethosoftlib.rag import Chunk, Index, RagError, tokenize
from ethosoftlib.rag import split_lines


def test_tokenize_handles_turkish_case_suffixes_and_stopwords():
    assert tokenize("IŞIK ve İSTANBUL'un Işıkları") == ["ışık", "istan", "ışıkl"]
    assert tokenize("Ankara'nın başkentidir 1923", prefix_len=None) == ["ankara", "başkentidir", "1923"]
    assert tokenize("başkenti") == tokenize("başkentidir") == ["başke"]


def test_inflected_turkish_query_finds_the_right_chunk():
    index = Index(chunk_words=5, overlap_words=0)
    index.add("İzmir Ege kıyısında bir liman kentidir.", "izmir.txt")
    index.add("Ankara, Türkiye'nin başkentidir.", "ankara.txt")
    index.add("Kapadokya peri bacalarıyla bilinir.", "kapadokya.txt")
    hits = index.search("Türkiye'nin başkenti neresi?", k=2)
    assert hits[0].source == "ankara.txt"
    assert all(hit.source != "kapadokya.txt" for hit in hits)


def test_rare_terms_outweigh_common_ones():
    index = Index(chunk_words=50)
    for i in range(5):
        index.add(f"model ayarları bölüm {i}", f"genel{i}.md")
    index.add("model kuantizasyon ayarları", "quant.md")
    assert index.search("kuantizasyon model", k=1)[0].source == "quant.md"


def test_split_lines_keeps_line_numbers_and_overlap():
    text = "\n".join(f"satir{i} a b c" for i in range(1, 11))  # 4 words per line
    chunks = split_lines(text, chunk_words=8, overlap_words=4)
    assert [first for first, _ in chunks] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert chunks[0][1] == "satir1 a b c\nsatir2 a b c"
    no_overlap = split_lines(text, chunk_words=8, overlap_words=0)
    assert [first for first, _ in no_overlap] == [1, 3, 5, 7, 9]
    assert split_lines("\n\n  \n", 8, 0) == []


def test_hit_points_to_the_line_where_the_chunk_starts():
    index = Index(chunk_words=3, overlap_words=0)
    index.add("giriş satırı burada\n\nGenerationConfig sıcaklığı ayarlar\nson satır", "cfg.py")
    hit = index.search("GenerationConfig", k=1)[0]
    assert (hit.source, hit.line) == ("cfg.py", 3)
    assert hit.text.startswith("GenerationConfig")


def test_add_directory_skips_tool_dirs_binaries_and_respects_pattern(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "kurulum.md").write_text("Mercan modelini indirme adımları", encoding="utf-8")
    (tmp_path / "notlar.txt").write_text("Mercan notları", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "x.md").write_text("Mercan venv", encoding="utf-8")
    (tmp_path / "model.md").write_bytes(b"Mercan\0binary")
    index = Index()
    assert index.add_directory(tmp_path, pattern="*.md") == 1
    assert [hit.source for hit in index.search("Mercan")] == ["docs/kurulum.md"]
    flat = Index()
    flat.add_directory(tmp_path, recursive=False)
    assert [hit.source for hit in flat.search("Mercan")] == ["notlar.txt"]
    with pytest.raises(RagError):
        index.add_directory(tmp_path / "yok")


def test_no_match_and_empty_index_return_nothing():
    assert Index().search("herhangi") == []
    index = Index()
    index.add("Ankara başkent", "a")
    assert index.search("kuantum fiziği") == []
    assert index.search("ve ile bir") == []


class _ConceptEmbedder:
    """Maps words to hand-made concept axes: a stand-in for a real model."""

    axes = {"başkent": 0, "hükümet": 0, "meclis": 0, "deniz": 1, "liman": 1, "kıyı": 1}

    def __init__(self):
        self.batches = []

    def embed(self, texts):
        self.batches.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.01, 0.01]
            for word, axis in self.axes.items():
                if word in text.lower():
                    vector[axis] += 1.0
            vectors.append(vector)
        return vectors


def test_hybrid_search_finds_semantic_matches_bm25_misses():
    embedder = _ConceptEmbedder()
    index = Index(embedder=embedder, chunk_words=10)
    index.add("Meclis ve hükümet bu şehirde toplanır.", "ankara.txt")
    index.add("Liman ve kıyı şeridi çok uzundur.", "izmir.txt")
    assert Index().search("başkent") == []  # BM25 alone has no shared term
    assert index.search("başkent", k=1)[0].source == "ankara.txt"
    assert embedder.batches[0] == ["Meclis ve hükümet bu şehirde toplanır."]


def test_embedder_vector_count_is_checked():
    class Broken:
        def embed(self, texts):
            return []

    with pytest.raises(RagError):
        Index(embedder=Broken()).add("metin", "a")


def test_save_and_load_round_trip(tmp_path):
    index = Index(chunk_words=4, overlap_words=0, prefix_len=None)
    index.add_chunks([Chunk("Ankara başkenttir", "a.txt", 7, {"dil": "tr"})])
    path = tmp_path / "index.json"
    index.save(path)
    loaded = Index.load(path)
    assert (loaded.prefix_len, len(loaded)) == (None, 1)
    hit = loaded.search("ankara")[0]
    assert (hit.source, hit.line, hit.metadata) == ("a.txt", 7, {"dil": "tr"})

    hybrid = Index(embedder=_ConceptEmbedder())
    hybrid.add("Meclis burada", "m.txt")
    hybrid.save(path)
    assert Index.load(path, embedder=_ConceptEmbedder()).search("başkent")[0].source == "m.txt"
    plain = Index()
    plain.add("x", "x")
    plain.save(path)
    with pytest.raises(RagError):
        Index.load(path, embedder=_ConceptEmbedder())


def test_rag_provider_is_registered_lazily():
    assert providers.describe("retrieval", "rag").factory == "ethosoftlib.rag:Index"
