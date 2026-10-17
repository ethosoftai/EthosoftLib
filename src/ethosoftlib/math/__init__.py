"""ethosoftlib.math: matematik cozumlerini dogrulanmis tool-use izlerine ceviren modul.

    >>> import ethosoftlib.math as em
    >>> em.convert("Toplam 3 + 4 = 7 elma.")
    'Toplam <arac_cagrisi>topla(3, 4)</arac_cagrisi><arac_sonucu>7</arac_sonucu> elma.'
    >>> em.fill_results("<arac_cagrisi>bol(48, 2)</arac_cagrisi>")
    '<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>'
    >>> em.calculate("4 + 5")
    '9'

Gereksinim: yalnizca `sympy` (pip install 'ethosoftlib[math]'). Kok paket `ethosoftlib`
bu modulu import ETMEZ; sadece `import ethosoftlib.math` yapildiginda yuklenir.
"""
try:
    import sympy  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ethosoftlib.math sympy gerektirir: pip install 'ethosoftlib[math]'"
    ) from exc

from .protocol import (  # noqa: F401
    CALL_OPEN, CALL_CLOSE, RES_OPEN, RES_CLOSE, REG, ToolError, TR_NAMES, EN_NAMES,
    execute, parse_call, run, render_protocol_md, localize, delocalize, set_language,
)
from .calc import calculate, evaluate  # noqa: F401
from .tooluse import tool_use  # noqa: F401
from .runtime import fill_results, result_block  # noqa: F401
from . import mask  # noqa: F401
from .converter import convert_text, verify_converted  # noqa: F401

__version__ = "0.5.0"


def convert(text: str, aggressive: bool = False, lang: str = None) -> str:
    """Metindeki dogrulanabilir hesaplari arac cagrilarina cevirir; sadece metni dondurur.
    lang: 'tr' (varsayilan, <arac_cagrisi>/topla) veya 'en' (<tool_call>/add)."""
    return convert_text(text, aggressive=aggressive, lang=lang)[0]


def verify(text: str):
    """Metindeki her arac cagrisini yeniden calistirir; (toplam, hatali) dondurur."""
    return verify_converted(text)


def translator(**kw):
    """Ethosoft-Math-Translator (EN->TR, formul korumali) ceviricisini dondurur. Ek bagimlilik: ethosoftlib[translate]."""
    from .translate import MathTranslator
    return MathTranslator(**kw)


def functions():
    """Protokoldeki fonksiyonlar: {ad: {cat, sig, doc}}"""
    return {k: {"cat": v["cat"], "sig": v["sig"], "doc": v["doc"]} for k, v in REG.items()}


class MathToolkit:
    """Registry uyumlu cephe: providers.create('tooluse', 'math', lang='tr')."""

    def __init__(self, lang: str = "tr", aggressive: bool = False):
        self.lang = lang
        self.aggressive = aggressive

    def convert(self, text: str) -> str:
        return convert(text, aggressive=self.aggressive, lang=self.lang)

    def tool_use(self, text: str) -> str:
        return tool_use(text, lang=self.lang)

    def calculate(self, text: str, **kw) -> str:
        return calculate(text, lang=self.lang, **kw)

    def fill_results(self, text: str) -> str:
        return fill_results(text, lang=self.lang)

    def result_block(self, call_or_text: str) -> str:
        return result_block(call_or_text, lang=self.lang)

    def execute(self, call: str) -> str:
        return execute(call)

    def verify(self, text: str):
        return verify(text)

    def translate(self, text: str, **kw):
        """EN->TR matematik cevirisi (modeli ilk cagrida yukler)."""
        if getattr(self, "_translator", None) is None:
            self._translator = translator()
        return self._translator.translate(text, **kw)
