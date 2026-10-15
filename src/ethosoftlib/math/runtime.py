"""Arac calistirici: metindeki <arac_cagrisi>...</arac_cagrisi> bloklarini calistirir,
sonucu <arac_sonucu>...</arac_sonucu> arasina yazar.

    >>> from ethosoftlib.math import fill_results
    >>> fill_results("<arac_cagrisi>bol(48, 2)</arac_cagrisi>")
    '<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>'

Model uretimi sirasinda: model '</arac_cagrisi>' yazdiginda durdurulur, `fill_results` (veya
`result_block`) cagrilir, sonuc modele geri verilir, uretim devam eder.
"""
import re

from . import protocol as _P
from .protocol import ToolError, execute

_CALL_RE = re.compile(
    r"<(?P<tag>arac_cagrisi|tool_call)>(?P<call>.*?)</(?P=tag)>"
    r"(?:\s*<(?:arac_sonucu|tool_result)>.*?(?:</(?:arac_sonucu|tool_result)>|\Z))?",
    re.S,
)


def _lang_of(tag: str) -> str:
    return "tr" if tag == "arac_cagrisi" else "en"


def _run_one(call_text: str, env: dict, lang: str) -> str:
    with _P.use_language(lang):
        try:
            return execute(call_text.strip(), env)
        except ToolError as ex:
            return f"{_P.WORDS[lang]['error']}: {ex}"
        except Exception as ex:  # beklenmeyen: modele hata metni don, uretimi kirma
            return f"{_P.WORDS[lang]['error']}: {type(ex).__name__}"


def fill_results(text: str, env: dict = None, lang: str = None) -> str:
    """Metindeki her arac cagrisini calistirip sonuc blogunu yazar/gunceller.

    - Sonuc blogu yoksa eklenir, yarim kalmissa tamamlanir, yanlissa yeniden hesaplanir.
    - Ayni metindeki `tanimla(...)` tanimlari sonraki cagrilarda gecerlidir.
    - Hata olursa sonuc `HATA: neden` olur.
    - Turkce (<arac_cagrisi>) ve Ingilizce (<tool_call>) etiketlerin ikisi de calisir.
    """
    env = env if env is not None else {}

    def sub(m):
        tag = m.group("tag")
        L = lang or _lang_of(tag)
        res = _run_one(m.group("call"), env, L)
        co, cc, ro, rc = _P.TAGS[_lang_of(tag)]
        return f"{co}{m.group('call')}{cc}{ro}{res}{rc}"

    return _CALL_RE.sub(sub, text)


def result_block(call_or_text: str, env: dict = None, lang: str = None) -> str:
    """Yalnizca sonuc blogunu dondurur: '<arac_sonucu>24</arac_sonucu>'.

    Girdi ya duz cagri ('bol(48, 2)') ya da icinde <arac_cagrisi> olan metindir
    (birden fazla varsa SONUNCU cagri kullanilir).
    """
    env = env if env is not None else {}
    matches = list(_CALL_RE.finditer(call_or_text))
    if matches:
        m = matches[-1]
        # onceki cagrilarin tanimlarini (tanimla) ortama isle
        for pm in matches[:-1]:
            _run_one(pm.group("call"), env, lang or _lang_of(pm.group("tag")))
        tag, call = m.group("tag"), m.group("call")
        L = lang or _lang_of(tag)
    else:
        tag, call, L = "arac_cagrisi", call_or_text, lang or _P.LANG
    res = _run_one(call, env, L)
    _, _, ro, rc = _P.TAGS[_lang_of(tag)]
    return f"{ro}{res}{rc}"
