"""Ethosoft-Math-Translator ile matematik metni Ingilizce -> Turkce ceviri (formul/sembol korumali).

    >>> from ethosoft_math.translate import MathTranslator
    >>> t = MathTranslator()                      # Ethosoft/Ethosoft-Math-Translator (ilk kullanimda indirilir)
    >>> t.translate("If $x^2 = 9$, then x is ±3.")
    'Eğer $x^2 = 9$ ise, x = ±3 olur.'

Calisma sekli: LaTeX / kod / URL / arac blogu ve ozel semboller (±, ×, √, CJK, ...) [[n]] yer tutucusuna cevrilir,
metin satir satir cevrilir, yer tutucular geri konur. Yer tutucusu bozulan satir `strict=True` iken None doner.

Kurulum: pip install "ethosoftlib[translate]"   (torch + transformers + sentencepiece)
"""
import re
from typing import Callable, Dict, List, Optional

from . import mask as _mask

MODEL_ID = "Ethosoft/Ethosoft-Math-Translator"

_PH = re.compile(r"\[\[\s*(\d+)\s*\]\]")
_WORD = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2,}")
# matematik unicode + CJK + birim/isaret karakterleri (egitim verisi hazirlanirken kullanilan aralik)
_SYM = re.compile(
    "[Ͱ-Ͽ⁰-₟₠-⃏℀-⅏⅐-↏←-⇿∀-⋿■-◿"
    "⟀-⟯⦀-⧿⨀-⫿　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯"
    "±×÷°·²³¹¼½¾′″¬µ§…−]")


def protect_all(text: str, is_unk: Optional[Callable[[str], bool]] = None):
    """mask.protect + ozel semboller + (varsa) sozlukte olmayan isaretler. -> (maskeli metin, span listesi)"""
    m = _mask.protect(text)
    spans = list(m.spans)
    out = []
    for part in re.split(r"(\[\[\d+\]\])", m.text):
        if _PH.fullmatch(part):
            out.append(part)
            continue
        i, n = 0, len(part)

        def special(ch):
            return bool(_SYM.match(ch)) or (is_unk is not None and not ch.isalnum() and not ch.isspace() and is_unk(ch))

        while i < n:
            if special(part[i]):
                j = i
                while j < n and special(part[j]):
                    j += 1
                spans.append(part[i:j])
                out.append("[[%d]]" % len(spans))
                i = j
            else:
                out.append(part[i])
                i += 1
    return "".join(out), spans


def _renumber(line: str, spans: List[str]):
    ids = [int(x) for x in _PH.findall(line)]
    mp = {old: new for new, old in enumerate(ids, 1)}
    return _PH.sub(lambda m: "[[%d]]" % mp[int(m.group(1))], line), [spans[i - 1] for i in ids]


def _restore(out: str, sp: List[str], strict: bool) -> Optional[str]:
    ids = sorted(int(x) for x in _PH.findall(out))
    if ids != list(range(1, len(sp) + 1)):
        if strict:
            return None
        present = set(ids)
        res = _PH.sub(lambda m: sp[int(m.group(1)) - 1] if 1 <= int(m.group(1)) <= len(sp) else m.group(0), out)
        tail = [s for i, s in enumerate(sp, 1) if i not in present]
        return (res.rstrip() + " " + " ".join(tail)).strip() if tail else res
    return _PH.sub(lambda m: sp[int(m.group(1)) - 1], out)


class MathTranslator:
    """Ince ayarli MarianMT (en->tr) + maskeleme. Modeli ilk cagrida yukler."""

    def __init__(self, model: str = MODEL_ID, device: Optional[str] = None, num_beams: int = 4,
                 batch_size: int = 64, translate_lines: Optional[Callable[[List[str]], List[str]]] = None):
        self.model_id, self.device, self.num_beams, self.batch_size = model, device, num_beams, batch_size
        self._lines_fn = translate_lines        # test/ozel arka uc icin enjekte edilebilir
        self._tok = self._model = None
        self._unk_cache: Dict[str, bool] = {}

    # --- model -------------------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError('Ceviri icin: pip install "ethosoftlib[translate]"') from e
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = MarianTokenizer.from_pretrained(self.model_id)
        self._model = MarianMTModel.from_pretrained(self.model_id).to(self.device).eval()

    def _is_unk(self, ch: str) -> bool:
        if self._tok is None:
            return False
        if ch not in self._unk_cache:
            unk = self._tok.unk_token_id
            a = unk in self._tok(ch)["input_ids"]
            with self._tok.as_target_tokenizer():
                b = unk in self._tok(ch)["input_ids"]
            self._unk_cache[ch] = a or b
        return self._unk_cache[ch]

    def _translate_lines(self, lines: List[str]) -> List[str]:
        if self._lines_fn is not None:
            return self._lines_fn(lines)
        self._load()
        torch = self._torch
        order = sorted(range(len(lines)), key=lambda i: len(lines[i]))
        res = [""] * len(lines)
        with torch.no_grad():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = self._tok([lines[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                                max_length=256).to(self.device)
                out = self._model.generate(**enc, num_beams=self.num_beams, max_new_tokens=320)
                for i, h in zip(idx, self._tok.batch_decode(out, skip_special_tokens=True)):
                    res[i] = h
        return res

    # --- API ---------------------------------------------------------------------------------
    def translate_many(self, texts: List[str], strict: bool = True, tools: bool = False,
                       tool_lang: str = "tr") -> List[Optional[str]]:
        """Metin listesini cevirir. strict=True: yer tutucusu bozulan metin icin None; False: eksik parcalar sona eklenir.
        tools=True: ceviriden sonra dogrulanabilir hesaplar arac cagrisina cevrilir (`topla(3, 4)` -> 7); tool_lang 'tr'|'en'."""
        if self._lines_fn is None:
            self._load()
        plans = []
        for text in texts:
            masked, spans = protect_all(text, self._is_unk)
            items = []
            for ln in masked.split("\n"):
                ln = ln.strip()
                nl, sp = _renumber(ln, spans) if _PH.search(ln) else (ln, [])
                items.append((nl, sp, bool(_WORD.search(_PH.sub(" ", nl)))))
            plans.append(items)
        todo = sorted({nl for items in plans for nl, _, need in items if need})
        done = dict(zip(todo, self._translate_lines(todo))) if todo else {}
        out = []
        for items in plans:
            lines, bad = [], False
            for nl, sp, need in items:
                if not need:
                    lines.append(_PH.sub(lambda m: sp[int(m.group(1)) - 1], nl) if sp else nl)
                    continue
                r = _restore(done[nl], sp, strict)
                if r is None:
                    bad = True
                    break
                lines.append(r)
            out.append(None if bad else "\n".join(lines))
        if tools:
            out = [None if o is None else self.to_tools(o, tool_lang) for o in out]
        return out

    @staticmethod
    def to_tools(text: str, lang: str = "tr") -> str:
        """Turkce (ya da herhangi bir) cozum metnindeki hesaplari arac cagrisi bicimine cevirir (sympy gerekir)."""
        from .converter import convert_text
        return convert_text(text, lang=lang, locale="auto")[0]

    def translate(self, text: str, strict: bool = False, tools: bool = False) -> Optional[str]:
        return self.translate_many([text], strict=strict, tools=tools)[0]

    def translate_record(self, rec: dict, fields=("problem", "solution"), suffix: str = "_tr",
                         strict: bool = True, tools: bool = False, tool_fields=("solution",)) -> Optional[dict]:
        """{'problem':..,'solution':..} -> ayni kayit + problem_tr/solution_tr. Bir alan basarisizsa None.
        tools=True: tool_fields (varsayilan sadece 'solution') ceviri sonrasi arac cagrisi bicimine de cevrilir."""
        vals = self.translate_many([rec[f] for f in fields], strict=strict)
        if any(v is None for v in vals):
            return None
        new = dict(rec)
        for f, v in zip(fields, vals):
            new[f + suffix] = self.to_tools(v) if (tools and f in tool_fields) else v
        new["translator"] = {"provider": "ethosoft-math-translator", "model": self.model_id}
        return new
