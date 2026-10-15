"""Ceviri oncesi sembol/formul/arac-bloku koruma katmani.

Makine cevirisi (LLM veya Marian) LaTeX, sayilar ve arac cagrilarini bozabilir. Bu modul:

  1. korunacak parcalari yer tutucuya ([[1]], [[2]] ...) cevirir      -> protect()
  2. ceviri sonrasi yer tutucularin hepsinin dogru geldigini denetler -> check()
  3. yer tutucularin yerine orijinal parcalari geri koyar             -> restore()

    >>> from ethosoftlib.math.mask import protect, restore, check
    >>> m = protect("Cozum: $x^2 - 4 = 0$ ise <arac_cagrisi>coz(\\"x^2-4=0\\", \\"x\\")</arac_cagrisi>")
    >>> m.text
    'Cozum: [[1]] ise [[2]]'
    >>> restore(m.text.replace("Cozum", "Solution").replace(" ise ", " then "), m)
    'Solution: $x^2 - 4 = 0$ then <arac_cagrisi>coz("x^2-4=0", "x")</arac_cagrisi>'
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

_PH = "[[{}]]"
_PH_RE = re.compile(r"\[\[\s*(\d+)\s*\]\]")

_BLOCKS = [
    # arac cagrisi + sonuc blogu (her iki dil)
    r"<(?:arac_cagrisi|tool_call)>.*?</(?:arac_cagrisi|tool_call)>"
    r"(?:\s*<(?:arac_sonucu|tool_result)>.*?</(?:arac_sonucu|tool_result)>)?",
    r"```.*?```",
    r"\[asy\].*?\[/asy\]",
    r"\\begin\{(?P<env>[A-Za-z*]+)\}.*?\\end\{(?P=env)\}",
    r"\$\$.+?\$\$",
    r"\\\[.+?\\\]",
    r"\\\(.+?\\\)",
    r"`[^`\n]+`",
    r"https?://[^\s)>\]]+",
]
_MATH_DOLLAR = re.compile(r"\$[^$\n]{1,300}?\$")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?:%|°)?")
_COMBINED = re.compile("|".join(f"(?:{b})" for b in _BLOCKS), re.S)


def _is_math_dollar(inner: str) -> bool:
    """$...$ gercekten matematik mi, para birimi mi?"""
    if re.search(r"[\\^_{}]", inner):
        return True
    if re.fullmatch(r"\s*[A-Za-z]{1,3}\s*", inner):
        return True
    if re.search(r"[A-Za-z]\s*[=<>()]|[=<>]\s*[A-Za-z0-9]", inner):
        return True
    return False


@dataclass
class Masked:
    text: str
    spans: List[str] = field(default_factory=list)   # spans[i] -> [[i+1]] yerine konacak orijinal


def protect(text: str, mask_numbers: bool = False) -> Masked:
    """Korunacak parcalari [[n]] yer tutucularina cevirir."""
    spans: List[str] = []
    pieces = []
    pos = 0

    # once blok desenleri + $...$ (math ise); ikisini soldan saga birlestir
    cands = []
    for m in _COMBINED.finditer(text):
        cands.append((m.start(), m.end()))
    for m in _MATH_DOLLAR.finditer(text):
        if _is_math_dollar(m.group(0)[1:-1]):
            cands.append((m.start(), m.end()))
    cands.sort()
    merged = []
    for s, e in cands:
        if merged and s < merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1] = (merged[-1][0], e)
            continue
        merged.append((s, e))

    if mask_numbers:
        inside = merged
        extra = []
        for m in _NUMBER.finditer(text):
            if not any(a <= m.start() < b for a, b in inside):
                extra.append((m.start(), m.end()))
        merged = sorted(merged + extra)

    for s, e in merged:
        pieces.append(text[pos:s])
        spans.append(text[s:e])
        pieces.append(_PH.format(len(spans)))
        pos = e
    pieces.append(text[pos:])
    return Masked("".join(pieces), spans)


def check(translated: str, masked: Masked) -> List[str]:
    """Ceviri sonrasi denetim. Bos liste = sorun yok."""
    problems = []
    found = [int(x) for x in _PH_RE.findall(translated)]
    want = list(range(1, len(masked.spans) + 1))
    missing = sorted(set(want) - set(found))
    extra = sorted(set(found) - set(want))
    dup = sorted({i for i in found if found.count(i) > 1})
    if missing:
        problems.append(f"eksik yer tutucu: {missing}")
    if extra:
        problems.append(f"fazladan yer tutucu: {extra}")
    if dup:
        problems.append(f"tekrar eden yer tutucu: {dup}")
    return problems


def restore(translated: str, masked: Masked, strict: bool = True) -> Optional[str]:
    """Yer tutucularin yerine orijinal parcalari koyar.

    strict=True: yer tutucu eksik/fazla/tekrarsa None doner (cagiran yeniden denesin).
    strict=False: eldekini geri koyar, eksikleri sona ekler.
    """
    problems = check(translated, masked)
    if problems and strict:
        return None

    def sub(m):
        i = int(m.group(1))
        return masked.spans[i - 1] if 1 <= i <= len(masked.spans) else m.group(0)

    out = _PH_RE.sub(sub, translated)
    if problems and not strict:
        present = {int(x) for x in _PH_RE.findall(translated)}
        tail = [s for i, s in enumerate(masked.spans, 1) if i not in present]
        if tail:
            out = out.rstrip() + " " + " ".join(tail)
    return out


def numbers_of(text: str) -> List[str]:
    """Metindeki sayi dizgileri (noktali yazim normallestirilmis) - sayi korunumu kontrolu icin."""
    return sorted(n.replace(",", ".") for n in re.findall(r"\d+(?:[.,]\d+)?", text))


def numbers_preserved(source: str, translated: str) -> bool:
    """Cevirinin sayi coklugu kaynakla ayni mi? (yer tutucu rakamlari sayilmaz)"""
    strip = lambda s: _PH_RE.sub(" ", s)
    return numbers_of(strip(source)) == numbers_of(strip(translated))
