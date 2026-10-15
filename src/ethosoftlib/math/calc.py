"""Serbest yazilmis ifadeler icin hesap makinesi: "4 + 5" -> 9.

    >>> from ethosoftlib.math.calc import calculate
    >>> calculate("4 + 5")
    '9'
    >>> calculate("(3 + 4) x 2 ?")
    '14'
    >>> calculate("2 uzeri 10")
    '1024'
    >>> calculate("1.250,50 + 3,5")
    '1254'
    >>> calculate("50'nin %20'si")
    '10'
    >>> calculate("3 < 5")
    'dogru'
    >>> calculate("100 / 3", decimals=2)
    '33.33'
"""
import re
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

import sympy

from . import protocol as _P
from .protocol import EN_NAMES, REG, ToolError, run, to_text

_NUM = r"\d+(?:[.,]\d+)?"

# Turkce / Ingilizce soz -> sembol (sayilar arasinda)
_WORD_OPS = [
    (r"\b(?:arti|artı|plus)\b", " + "),
    (r"\b(?:eksi|minus)\b", " - "),
    (r"\b(?:carpi|çarpı|kere|kez|times|multiplied by)\b", " * "),
    (r"\b(?:bolu|bölü|divided by|over)\b", " / "),
    (r"\b(?:uzeri|üzeri|ussu|üssü|to the power of|power)\b", " ^ "),
    (r"\b(?:kare\s*k[oö]k[uü]?|karekok[uü]?)\s*", " sqrt "),
]
_FILLER = re.compile(
    r"\b(?:kac eder|kaç eder|kactir|kaçtır|kac|kaç|nedir|eder|bana|lutfen|lütfen|hesapla|hesaplar misin|"
    r"yi|yı|yu|yü|sonucu|sonuc|sonuç|what is|calculate|compute|equals?)\b[?]*", re.I)
_REL = re.compile(r"(<=|>=|==|!=|≠|≤|≥|<|>|=)")


def _fraction_text(fr: Fraction) -> str:
    return _P.fmt_frac(fr)


def _apply_mod(s: str) -> str:
    pat = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:\bmod\b|\bmodulo\b|%)\s*(-?\d+(?:\.\d+)?)", re.I)
    for _ in range(10):
        m = pat.search(s)
        if not m:
            break
        a, b = Fraction(m.group(1)), Fraction(m.group(2))
        if b == 0:
            raise ToolError("sifira bolme")
        r = a - b * (a / b).__floor__()
        s = s[:m.start()] + _fraction_text(r) + s[m.end():]
    return s


def _normalize(s: str, locale: str = "auto") -> str:
    s = s.strip()
    s = s.replace("−", "-").replace("–", "-").replace("×", "*").replace("÷", "/").replace("·", "*")
    s = s.replace("**", "^").replace("’", "'")
    s = _apply_mod(s)
    # yuzde kaliplari (soz operatorlerden once)
    s = re.sub(rf"({_NUM})\s*'?\s*n[iıuü]n\s*%\s*({_NUM})\s*'?\s*[sş][iıuü]", r"\2 % * \1", s, flags=re.I)
    s = re.sub(rf"\b(?:yuzde|yüzde)\s*({_NUM})\s+(?:of\s+)?({_NUM})", r"\1 % * \2", s, flags=re.I)
    s = re.sub(rf"%\s*({_NUM})", r"\1 %", s)
    s = re.sub(rf"\b(?:yuzde|yüzde)\s*({_NUM})", r"\1 %", s, flags=re.I)
    s = _FILLER.sub(" ", s)
    for pat, rep in _WORD_OPS:
        s = re.sub(pat, rep, s, flags=re.I)
    # bilimsel gosterim 1e3
    def _sci(m):
        e = int(m.group(2))
        if abs(e) > 30:
            return m.group(0)
        return _fraction_text(Fraction(m.group(1)) * (Fraction(10) ** e))
    s = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![\w.])", _sci, s)
    # bosluklu binlik: 10 000 / 1 000 000
    s = re.sub(r"(?<![\d.,])(\d{1,3}(?: \d{3})+)(?![\d])", lambda m: m.group(1).replace(" ", ""), s)
    # Turk yazimi 1.250,50 ; ve nokta=binlik (locale)
    s = re.sub(r"(?<![\d.,])\d{1,3}(?:\.\d{3})+,\d+(?![\d.,])",
               lambda m: m.group(0).replace(".", "").replace(",", "."), s)
    if locale == "tr":
        s = re.sub(r"(?<![\d.,])\d{1,3}(?:\.\d{3})+(?![\d.,])", lambda m: m.group(0).replace(".", ""), s)
    else:  # auto/en: 1.000.000 gibi birden cok nokta kesin binliktir
        s = re.sub(r"(?<![\d.,])\d{1,3}(?:\.\d{3}){2,}(?![\d.,])", lambda m: m.group(0).replace(".", ""), s)
    # 3,14 -> 3.14 (1,000 gibi uclu bin ayirac oldugu gibi kalir)
    s = re.sub(r"(?<=\d),(?=\d(?!\d\d(?!\d)))", ".", s)
    s = re.sub(r"(?<=\d),(?=\d{1,2}(?!\d))", ".", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[=?\s]+$", "", s)
    return s


_UNITS = set("km kg cm mm ml lt gr mg dk sn sa tl usd eur adet tane kez min sec hr mph kmh".split())


def _split_vars(s: str) -> str:
    """'2ab' -> '2*a*b' ; 'ab' -> 'a*b' (birim/fonksiyon adlarina dokunmaz)."""
    known = set(REG) | set(EN_NAMES) | set(_P.TR_NAMES.values()) | _UNITS | {"pi", "oo", "of"}

    def sub(m):
        w = m.group(0)
        if w in known:
            return w
        return "*".join(w)

    s = re.sub(r"(?<=\d)(?=[a-z])", "*", s)          # 2ab -> 2*ab
    return re.sub(r"(?<![A-Za-z_])[a-z]{2,3}(?![A-Za-z_(])", sub, s)


def _numeric_or_symbolic(s: str):
    from .converter import _make_scan, has_sym, parse_full, tokenize, to_sympy

    if re.search(r"(?<=\d)[a-z]{2,3}\b", s):
        probe = _split_vars(s)
        if probe != s and not re.search(r"\b(?:" + "|".join(_UNITS) + r")\b", s):
            s = probe
    toks = tokenize(_make_scan(s))
    node = parse_full(toks, allow_syms=False)
    if node is not None and not has_sym(node):
        return run(node), node
    node = parse_full(toks, allow_syms=True) if node is None else node
    if node is None:
        toks2 = tokenize(_make_scan(_split_vars(s)))
        node = parse_full(toks2, allow_syms=True)
    if node is not None:
        return sympy.simplify(to_sympy(node)), node
    raise ToolError(f"ifade anlasilamadi: {s!r}")


_CALL = re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)\s*\(.*\)$", re.S)


def _try_protocol_call(s: str):
    m = _CALL.match(s)
    if not m:
        return None
    name = m.group(1)
    if name in REG or name in EN_NAMES:
        try:
            return run(_P.parse_call(s))
        except ToolError:
            return None
    return None


def _extract(s: str):
    """Cumle icinden en uzun hesaplanabilir sayisal ifadeyi bul."""
    cands = re.findall(r"[-+(√]*\d[\d\s.,+\-*/^()!%xX×÷]*[\d)!%]|\d+", s)
    cands = sorted({c.strip() for c in cands}, key=len, reverse=True)
    for c in cands:
        if not re.search(r"[+\-*/^!xX×÷%]", c.strip("-+")):
            continue
        try:
            return _numeric_or_symbolic(_normalize(c))[0]
        except Exception:
            continue
    return None


def _eval_one(s: str):
    called = _try_protocol_call(s)
    if called is not None:
        return called
    try:
        return _numeric_or_symbolic(s)[0]
    except ToolError:
        got = _extract(s)
        if got is not None:
            return got
        raise


def evaluate(text: str, locale: str = "auto"):
    """Ifadeyi hesaplar; Fraction / sympy / bool dondurur. Cozulemezse ToolError."""
    s = _normalize(str(text), locale)
    if not s:
        raise ToolError("bos ifade")
    parts = _REL.split(s)
    if len(parts) >= 3:
        vals = [_eval_one(parts[i].strip()) for i in range(0, len(parts), 2)]
        ops = parts[1::2]
        ok = True
        for (a, b), op in zip(zip(vals, vals[1:]), ops):
            ok = ok and _compare(a, b, op)
        return ok
    return _eval_one(s)


def _compare(a, b, op):
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        d = a - b
    else:
        A = sympy.Rational(a.numerator, a.denominator) if isinstance(a, Fraction) else sympy.sympify(a)
        B = sympy.Rational(b.numerator, b.denominator) if isinstance(b, Fraction) else sympy.sympify(b)
        d = sympy.simplify(A - B)
        if d.is_number:
            d = Fraction(str(sympy.nsimplify(d))) if d.is_Rational else float(sympy.N(d, 30))
    if op in ("=", "=="):
        return d == 0
    if op in ("!=", "≠"):
        return d != 0
    if op in ("<=", "≤"):
        return d <= 0
    if op in (">=", "≥"):
        return d >= 0
    return d < 0 if op == "<" else d > 0


def _format(value, decimals):
    if decimals is None or isinstance(value, bool) or isinstance(value, str):
        return to_text(value)
    try:
        x = Decimal(str(sympy.N(sympy.sympify(value) if not isinstance(value, Fraction)
                                else sympy.Rational(value.numerator, value.denominator), decimals + 12)))
    except Exception:
        return to_text(value)
    q = Decimal(1).scaleb(-decimals)
    out = format(x.quantize(q, rounding=ROUND_HALF_UP), "f")
    if decimals > 0 and "." in out:
        out = out.rstrip("0").rstrip(".")
    return out


def calculate(text: str, trace: bool = False, lang: str = None, decimals: int = None,
              locale: str = "auto", on_error: str = "raise") -> str:
    """'4 + 5' -> '9'.

    trace=True   : '<arac_cagrisi>topla(4, 5)</arac_cagrisi><arac_sonucu>9</arac_sonucu>'
    decimals=N   : sonucu N ondalige yuvarlar (100/3 -> 33.33)
    locale       : 'auto' | 'tr' (tek nokta = binlik: 1.000 -> 1000) | 'en'
    on_error     : 'raise' (varsayilan) | 'text' ('HATA: ...' dondurur)
    """
    lang = lang or _P.LANG
    with _P.use_language(lang):
        try:
            value = evaluate(text, locale)
        except ToolError as ex:
            if on_error == "text":
                return f"{_P.WORDS[lang]['error']}: {ex}"
            raise
        res = _format(value, decimals)
        if not trace:
            return res
        from .converter import _make_scan, parse_full, render, tokenize
        s = _normalize(str(text), locale)
        node = parse_full(tokenize(_make_scan(s)), allow_syms=True)
        if _CALL.match(s):
            call = s
        elif node is not None and not isinstance(node, Fraction) and not _REL.search(s):
            call = render(node)
        else:
            call = f'hesapla({__import__("json").dumps(str(text), ensure_ascii=False)})'
        out = f"{_P.CALL_OPEN}{call}{_P.CALL_CLOSE}{_P.RES_OPEN}{res}{_P.RES_CLOSE}"
    return _P.localize(out, lang)
