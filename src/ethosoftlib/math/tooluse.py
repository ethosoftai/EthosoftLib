"""Duz metin -> tool-use bicimi.

    >>> from ethosoftlib.math import tool_use
    >>> tool_use("4 + 5")
    '<arac_cagrisi>topla(4, 5)</arac_cagrisi><arac_sonucu>9</arac_sonucu>'
    >>> tool_use("5 elma + 3 elma kac eder?")
    '<arac_cagrisi>topla(5, 3)</arac_cagrisi><arac_sonucu>8</arac_sonucu> elma kac eder?'
    >>> tool_use("Bir kitap 25 TL. 4 kitap icin 25 x 4 TL oderiz.")
    'Bir kitap 25 TL. 4 kitap icin <arac_cagrisi>carp(25, 4)</arac_cagrisi><arac_sonucu>100</arac_sonucu> TL oderiz.'

Mantik:
 1. Metindeki dogrulanabilir esitlikler/denklemler (3 + 4 = 7, x^2-4=0 ...) cevrilir.
 2. Sonucu yazilmamis, birimli veya Turkce sozcukle yazilmis islemler ("5 elma + 3 elma",
    "6 carpi 7", "12 - 5 kac eder") hesaplanip arac cagrisi + sonuc olarak yerine konur.
 Cevrilemeyen kisimlar oldugu gibi kalir.
"""
import re

from . import protocol as _P
from .calc import _normalize, _numeric_or_symbolic
from .converter import _CODE_HINT, convert_text, has_sym, is_call, render
from .protocol import ToolError, to_text

_W = r"[A-Za-zÇĞİÖŞÜçğıöşüâîû]{1,14}"
_OPWORDS = r"(?:arti|artı|eksi|carpi|çarpı|kere|kez|bolu|bölü|uzeri|üzeri|ussu|üssü|plus|minus|times)"
_OP = rf"(?:[+\-*/^×÷]|\b[xX]\b|\b{_OPWORDS}\b)"
_NUM = r"\(*-?\d[\d.,]*\)*"
# sayi (+ birim) (islem sayi (+ birim))+ ;  birim olarak islem sozcugu yenmesin
_UNIT = rf"(?:\s+(?!(?:[xX]\b|{_OPWORDS}\b)){_W})?"
_SPAN = re.compile(
    rf"(?<![\w.,/])(({_NUM}(?:\s*%)?){_UNIT}(?:\s*{_OP}\s*{_NUM}(?:\s*%)?{_UNIT})+)", re.U)
_TAG = re.compile(r"<(?:tool_call|arac_cagrisi)>.*?</(?:tool_call|arac_cagrisi)>"
                  r"<(?:tool_result|arac_sonucu)>.*?</(?:tool_result|arac_sonucu)>", re.S)
_TRAIL_WORD = re.compile(rf"\s+{_W}$", re.U)


def _convert_plain_segment(seg: str, counter: list) -> str:
    def repl(m):
        span = m.group(1)
        trailing = ""
        # sondaki birim sozcugu span disinda kalsin
        t = _TRAIL_WORD.search(span)
        if t and not re.search(rf"\b{_OPWORDS}\b$", span, re.I):
            trailing = span[t.start():]
            span = span[:t.start()]
        if not re.search(r"\d\s*(?:\)|%)?\s*(?:" + _OP + r")", span):
            return m.group(0)
        # tarih / aralik: 2020-2021, 5-10 (bosluksuz tire)
        if re.fullmatch(r"\d{1,4}\s*[-/]\s*\d{1,4}(\s*[-/]\s*\d{1,4})?", span) and " " not in span:
            return m.group(0)
        try:
            value, node = _numeric_or_symbolic(_normalize(span))
        except (ToolError, Exception):
            return m.group(0)
        if not is_call(node) or has_sym(node):
            return m.group(0)
        res = to_text(value)
        if len(res) > 40:
            return m.group(0)
        counter[0] += 1
        return f"{_P.CALL_OPEN}{render(node)}{_P.CALL_CLOSE}{_P.RES_OPEN}{res}{_P.RES_CLOSE}" + trailing

    return _SPAN.sub(repl, seg)


def tool_use(text: str, lang: str = None, return_stats: bool = False):
    """Duz metni tool-use bicimine cevirir (metnin geri kalani korunur)."""
    lang = lang or _P.LANG
    stats = {"equations": 0, "expressions": 0}
    with _P.use_language(lang):
        if _CODE_HINT.search(text):
            out = text
        else:
            out, st = convert_text(text, aggressive=True, lang=lang)
            stats["equations"] = st.get("converted", 0)
            counter = [0]
            parts, pos = [], 0
            for m in _TAG.finditer(out):
                parts.append(_P.localize(_convert_plain_segment(out[pos:m.start()], counter), lang))
                parts.append(m.group(0))
                pos = m.end()
            parts.append(_P.localize(_convert_plain_segment(out[pos:], counter), lang))
            out = "".join(parts)
            stats["expressions"] = counter[0]
    return (out, stats) if return_stats else out
