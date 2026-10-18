"""Web search, page reading and downloads without API keys or dependencies.

    >>> from ethosoftlib import web
    >>> for hit in web.search("Ankara hava durumu", max_results=3):
    ...     print(hit.title, hit.url)
    >>> page = web.open_url("https://example.com")
    >>> web.download("https://example.com/file.pdf", "indirilenler/")

Search uses DuckDuckGo's HTML endpoint by default. Set ``ETHOSOFT_SEARXNG_URL``
(or pass ``SearXNGSearch(base_url)``) to use a self-hosted SearXNG instance.
Only http(s) URLs are fetched, and hosts that resolve to loopback, private or
link-local addresses are refused unless ``allow_private=True``: these tools are
meant to be driven by models, which must not reach the local network.
"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..core.errors import EthosoftError

__all__ = [
    "DuckDuckGoSearch",
    "Page",
    "SearXNGSearch",
    "SearchProvider",
    "SearchResult",
    "WebClient",
    "WebError",
    "default_provider",
    "download",
    "open_url",
    "search",
]

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ethosoftlib-web"
DEFAULT_TIMEOUT = 15.0
MAX_PAGE_BYTES = 5 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
_CHUNK = 64 * 1024


class WebError(EthosoftError):
    """A search, page fetch or download failed."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {"url": self.url, "title": self.title, "text": self.text,
                "truncated": self.truncated}


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int, *, timeout: float) -> list[SearchResult]:
        ...


# --------------------------------------------------------------------------- HTTP

def _check_url(url: str, allow_private: bool) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise WebError(f"Only http(s) URLs are supported: {url!r}")
    if not allow_private:
        try:
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or None)
        except socket.gaierror as exc:
            raise WebError(f"Could not resolve {parsed.hostname!r}: {exc}") from exc
        for info in infos:
            address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
            if not address.is_global:
                raise WebError(
                    f"Refusing non-public address {address} for {parsed.hostname!r}; "
                    "pass allow_private=True to allow local network access."
                )
    return url


class _CheckedRedirects(HTTPRedirectHandler):
    def __init__(self, allow_private: bool) -> None:
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        _check_url(urljoin(req.full_url, newurl), self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url: str, *, data: bytes | None, timeout: float, allow_private: bool,
          headers: dict[str, str] | None = None):
    _check_url(url, allow_private)
    request = Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    opener = build_opener(_CheckedRedirects(allow_private))
    try:
        return opener.open(request, timeout=timeout)
    except HTTPError as exc:
        raise WebError(f"HTTP {exc.code} for {url}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise WebError(f"Request failed for {url}: {exc}") from exc


def _read_limited(response, limit: int) -> tuple[bytes, bool]:
    body = response.read(limit + 1)
    return body[:limit], len(body) > limit


def _charset(response) -> str:
    return response.headers.get_content_charset() or "utf-8"


# --------------------------------------------------------------------------- HTML

_SKIP = {"script", "style", "noscript", "template", "svg", "head",
         "nav", "aside", "footer", "form", "button", "menu", "dialog", "select"}
_SKIP_ROLES = {"navigation", "banner", "contentinfo", "complementary", "menu",
               "search", "dialog", "toolbar"}
# Page chrome is often a <div> whose class or id names what it is.
_SKIP_MARKERS = frozenset({
    "menu", "dropdown", "sidebar", "navbox", "navbar", "breadcrumb", "breadcrumbs",
    "cookie", "cookies", "toc", "footer", "share", "advert", "ads", "banner", "noprint",
})
_MAIN = {"main", "article"}
_BLOCK = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
          "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote"}
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
         "meta", "param", "source", "track", "wbr"}


def _is_chrome(tag: str, attrs: dict[str, str | None]) -> bool:
    if tag in _SKIP or "hidden" in attrs or attrs.get("aria-hidden") == "true":
        return True
    if (attrs.get("role") or "") in _SKIP_ROLES:
        return True
    # Page-wide state classes such as "vector-toc-available" sit on these.
    if tag in ("html", "body") or tag in _MAIN:
        return False
    names = f"{attrs.get('class') or ''} {attrs.get('id') or ''}".lower()
    # Whole words only: "share" must not hide "sharedMarkdownContent".
    return not _SKIP_MARKERS.isdisjoint(re.split(r"[^a-z0-9]+", names))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        # Text inside <main>/<article>; preferred over the whole page when present.
        self.main_parts: list[str] = []
        self.title_parts: list[str] = []
        # Open elements as (tag, skipped, main); unclosed tags are popped by
        # the nearest matching end tag, as browsers do for sloppy markup.
        self._stack: list[tuple[str, bool, bool]] = []
        self._in_title = False

    @property
    def _skipping(self) -> bool:
        return bool(self._stack) and self._stack[-1][1]

    def _emit(self, text: str) -> None:
        if self._skipping:
            return
        self.parts.append(text)
        if self._stack and self._stack[-1][2]:
            self.main_parts.append(text)

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
            return
        if tag in _BLOCK:
            self._emit("\n")
        if tag in _VOID:
            return
        skipped = self._skipping or _is_chrome(tag, dict(attrs))
        main = (bool(self._stack) and self._stack[-1][2]) or tag in _MAIN
        self._stack.append((tag, skipped, main))

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
            return
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                break
        if tag in _BLOCK or tag in _MAIN:
            self._emit("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        else:
            self._emit(data)


def html_to_text(html: str) -> tuple[str, str]:
    """Return ``(title, readable_text)`` for an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    raw = "".join(parser.main_parts)
    if len(raw.split()) < 20:
        raw = "".join(parser.parts)
    lines = (" ".join(line.split()) for line in raw.splitlines())
    text = "\n".join(line for line in lines if line)
    return " ".join("".join(parser.title_parts).split()), text


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._field: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if "result__a" in classes:
            self.results.append({"title": "", "url": attributes.get("href") or "", "snippet": ""})
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag == "a":
            self._field = None

    def handle_data(self, data):
        if self._field and self.results:
            self.results[-1][self._field] += data


def _unwrap_duckduckgo_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com") and parsed.path == "/l/":
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return url


def parse_duckduckgo_html(html: str) -> list[SearchResult]:
    parser = _DuckDuckGoParser()
    parser.feed(html)
    parser.close()
    results = []
    for item in parser.results:
        url = _unwrap_duckduckgo_url(item["url"])
        parsed = urlparse(url)
        # Sponsored results go through duckduckgo.com/y.js; they are not search hits.
        if parsed.scheme not in ("http", "https") or (parsed.hostname or "").endswith("duckduckgo.com"):
            continue
        results.append(SearchResult(
            title=" ".join(item["title"].split()),
            url=url,
            snippet=" ".join(item["snippet"].split()),
        ))
    return results


# ------------------------------------------------------------------- providers

class DuckDuckGoSearch:
    """Keyless search through DuckDuckGo's HTML endpoint."""

    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, *, region: str = "tr-tr") -> None:
        self.region = region

    def search(self, query: str, max_results: int, *, timeout: float) -> list[SearchResult]:
        data = urlencode({"q": query, "kl": self.region}).encode()
        with _open(self.endpoint, data=data, timeout=timeout, allow_private=False) as response:
            body, _ = _read_limited(response, MAX_PAGE_BYTES)
            html = body.decode(_charset(response), errors="replace")
        results = parse_duckduckgo_html(html)
        if not results and "anomaly" in html.lower():
            raise WebError("DuckDuckGo rate-limited this client; retry later or use SearXNG.")
        return results[:max_results]


class SearXNGSearch:
    """Search through a SearXNG instance with the JSON format enabled."""

    def __init__(self, base_url: str, *, language: str = "tr", allow_private: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.language = language
        # A self-hosted instance usually lives on the local network.
        self.allow_private = allow_private

    def search(self, query: str, max_results: int, *, timeout: float) -> list[SearchResult]:
        url = f"{self.base_url}/search?" + urlencode(
            {"q": query, "format": "json", "language": self.language}
        )
        with _open(url, data=None, timeout=timeout, allow_private=self.allow_private,
                   headers={"Accept": "application/json"}) as response:
            body, _ = _read_limited(response, MAX_PAGE_BYTES)
        try:
            payload = json.loads(body.decode(_charset(response), errors="replace"))
        except json.JSONDecodeError as exc:
            raise WebError("SearXNG did not return JSON; enable the json format.") from exc
        return [
            SearchResult(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                snippet=str(item.get("content") or ""),
            )
            for item in payload.get("results", [])
            if item.get("url")
        ][:max_results]


def default_provider() -> SearchProvider:
    searxng = os.environ.get("ETHOSOFT_SEARXNG_URL")
    return SearXNGSearch(searxng) if searxng else DuckDuckGoSearch()


# ----------------------------------------------------------------------- client

class WebClient:
    """Search, read and download with one provider and one network policy."""

    def __init__(self, provider: SearchProvider | None = None, *,
                 timeout: float = DEFAULT_TIMEOUT, allow_private: bool = False) -> None:
        self.provider = provider or default_provider()
        self.timeout = timeout
        self.allow_private = allow_private

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise WebError("Search query is empty.")
        return self.provider.search(query, max(1, int(max_results)), timeout=self.timeout)

    def open_url(self, url: str, max_chars: int = 20000) -> Page:
        with _open(url, data=None, timeout=self.timeout, allow_private=self.allow_private) as response:
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            if not (content_type.startswith("text/") or content_type in (
                "application/xhtml+xml", "application/json", "application/xml",
            )):
                raise WebError(f"{url} is {content_type}, not a text page; use download().")
            body, cut = _read_limited(response, MAX_PAGE_BYTES)
            raw = body.decode(_charset(response), errors="replace")
        if content_type in ("text/html", "application/xhtml+xml"):
            title, text = html_to_text(raw)
        else:
            title, text = "", raw.strip()
        truncated = cut or len(text) > max_chars
        return Page(url=final_url, title=title, text=text[:max_chars], truncated=truncated)

    def download(self, url: str, path: str | os.PathLike[str] = ".", *,
                 overwrite: bool = False, max_bytes: int = MAX_DOWNLOAD_BYTES) -> Path:
        target = Path(path).expanduser()
        with _open(url, data=None, timeout=self.timeout, allow_private=self.allow_private) as response:
            if target.is_dir() or str(path).endswith(("/", os.sep)):
                name = Path(urlparse(response.geturl()).path).name or "download"
                target = target / name
            if target.exists() and not overwrite:
                raise WebError(f"{target} already exists; pass overwrite=True to replace it.")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            written = 0
            try:
                with partial.open("wb") as handle:
                    while chunk := response.read(_CHUNK):
                        written += len(chunk)
                        if written > max_bytes:
                            raise WebError(f"{url} is larger than {max_bytes} bytes.")
                        handle.write(chunk)
                partial.replace(target)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        return target


def search(query: str, max_results: int = 5, **client_options) -> list[SearchResult]:
    return WebClient(**client_options).search(query, max_results)


def open_url(url: str, max_chars: int = 20000, **client_options) -> Page:
    return WebClient(**client_options).open_url(url, max_chars)


def download(url: str, path: str | os.PathLike[str] = ".", *, overwrite: bool = False,
             **client_options) -> Path:
    return WebClient(**client_options).download(url, path, overwrite=overwrite)
