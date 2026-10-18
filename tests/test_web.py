from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from ethosoftlib import providers
from ethosoftlib import web

DDG_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a" href="https://www.mgm.gov.tr/tahmin/il-ve-ilceler.aspx?il=ANKARA&amp;x=1">
    <b>Ankara</b> Hava Durumu</a>
  <a class="result__snippet" href="https://www.mgm.gov.tr/"><b>ANKARA</b> için detaylı &quot;hava&quot; durumu</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa%3Fb%3D1&amp;rut=x">Wrapped</a>
</div>
<div class="result result--ad">
  <a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=x">Sponsored</a>
</div>
"""


def test_duckduckgo_results_are_parsed_and_unwrapped():
    results = web.parse_duckduckgo_html(DDG_HTML)
    assert [r.url for r in results] == [
        "https://www.mgm.gov.tr/tahmin/il-ve-ilceler.aspx?il=ANKARA&x=1",
        "https://example.org/a?b=1",
    ]
    assert results[0].title == "Ankara Hava Durumu"
    assert results[0].snippet == 'ANKARA için detaylı "hava" durumu'


def test_html_to_text_drops_scripts_and_keeps_blocks():
    title, text = web.html_to_text(
        "<html><head><title> Başlık </title><style>p{}</style></head>"
        "<body><script>alert(1)</script><h1>Merhaba</h1><p>bir   iki</p><p>üç</p></body></html>"
    )
    assert title == "Başlık"
    assert text == "Merhaba\nbir iki\nüç"



def test_html_to_text_prefers_main_content_over_navigation():
    body = " ".join(f"kelime{i}" for i in range(30))
    title, text = web.html_to_text(
        f"<body><nav>Ana menü</nav><div>Reklam</div><main><p>{body}</p>"
        "<aside>Kenar</aside></main><footer>Telif</footer></body>"
    )
    assert text == body


def test_page_chrome_is_matched_by_whole_class_words():
    _, text = web.html_to_text(
        '<body class="vector-toc-available"><div class="lang-dropdown">Dil</div>'
        '<div role="navigation">Menü</div><div class="sharedMarkdownContent">Asıl metin</div>'
        '<div id="site-footer">Telif</div><div hidden>Gizli</div></body>'
    )
    assert text == "Asıl metin"

@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "http://127.0.0.1:9/"])
def test_unsafe_urls_are_refused(url):
    with pytest.raises(web.WebError):
        web.open_url(url)


class _FakeProvider:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results, *, timeout):
        self.calls.append((query, max_results, timeout))
        return [web.SearchResult("t", "https://example.org")]


def test_client_uses_given_provider_and_rejects_empty_query():
    provider = _FakeProvider()
    client = web.WebClient(provider, timeout=3)
    assert client.search("  ankara  ", 2)[0].url == "https://example.org"
    assert provider.calls == [("ankara", 2, 3)]
    with pytest.raises(web.WebError):
        client.search("   ")


def test_searxng_is_selected_from_environment(monkeypatch):
    monkeypatch.setenv("ETHOSOFT_SEARXNG_URL", "http://localhost:8888/")
    provider = web.default_provider()
    assert isinstance(provider, web.SearXNGSearch)
    assert provider.base_url == "http://localhost:8888"
    monkeypatch.delenv("ETHOSOFT_SEARXNG_URL")
    assert isinstance(web.default_provider(), web.DuckDuckGoSearch)


class _Handler(BaseHTTPRequestHandler):
    routes = {
        "/page": ("text/html; charset=utf-8", "<title>Sayfa</title><p>İçerik</p>".encode()),
        "/data.bin": ("application/octet-stream", b"\x00\x01" * 1000),
        "/search": ("application/json", b'{"results": [{"title": "S", "url": "https://s.org", "content": "c"}]}'),
    }

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/page")
            self.end_headers()
            return
        if path not in self.routes:
            self.send_error(404)
            return
        content_type, body = self.routes[path]
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@contextmanager
def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_open_url_follows_redirects_and_extracts_text():
    with _server() as base:
        page = web.open_url(base + "/redirect", allow_private=True)
    assert page.url.endswith("/page")
    assert (page.title, page.text, page.truncated) == ("Sayfa", "İçerik", False)


def test_open_url_rejects_binary_content():
    with _server() as base, pytest.raises(web.WebError, match="download"):
        web.open_url(base + "/data.bin", allow_private=True)


def test_download_to_directory_and_refuse_overwrite(tmp_path):
    with _server() as base:
        target = web.download(base + "/data.bin", tmp_path, allow_private=True)
        assert target == tmp_path / "data.bin"
        assert target.read_bytes() == b"\x00\x01" * 1000
        with pytest.raises(web.WebError, match="exists"):
            web.download(base + "/data.bin", tmp_path, allow_private=True)
        client = web.WebClient(allow_private=True)
        with pytest.raises(web.WebError, match="larger"):
            client.download(base + "/data.bin", tmp_path / "big.bin", max_bytes=10)
    assert not (tmp_path / "big.bin").exists()
    assert not (tmp_path / "big.bin.part").exists()


def test_searxng_json_results():
    with _server() as base:
        results = web.SearXNGSearch(base).search("q", 5, timeout=5)
    assert results == [web.SearchResult("S", "https://s.org", "c")]


def test_web_provider_is_registered_lazily():
    info = providers.describe("search", "web")
    assert info.factory == "ethosoftlib.web:WebClient"
