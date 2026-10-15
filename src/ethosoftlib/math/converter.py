"""Cozum metinlerindeki hesaplari MathTool cagrilarina cevirir (v2).

Ilke: sadece DOGRULANABILEN hesaplar donusturulur; dogrulanamayan her sey oldugu gibi kalir.
Kapsam: sayisal eslitlikler, \\boxed cevaplar, yaklasik (≈) degerler, kelimeyle islemler,
birimler, yuzde, sembolik ozdeslik (expand/factor/simplify), denklem cozme (solve),
fonksiyon tanimlama (define) ve $...$ icinde tek basina duran sayisal ifadeler.
"""
import bisect
import json
import re
from decimal import Decimal
from fractions import Fraction

import sympy

from . import protocol as _P
from .protocol import (CALL_OPEN, CALL_CLOSE, RES_OPEN, RES_CLOSE, ToolError, fmt_frac,
                       to_text, run, execute, parse_call, _round_half_up, _sym)

# ------------------------------------------------------------------ tokenizer
_TOK = re.compile(r"""
  (?P<NUM>(?<![\d.])(?:\d{1,3}(?:(?:,\\!|\{,\}|,)\d{3})+(?!\d)|\d+)(?:\.\d+)?|(?<![\d.])\.\d+)
 |(?P<CMD>\\[A-Za-z]+)
 |(?P<ESC>\\[%$,;!:\s_])
 |(?P<WORD>[A-Za-z]+)
 |(?P<SP>\s+)
 |(?P<CH>.)
""", re.X | re.S)

_OPMAP = {"×": "*", "·": "*", "⋅": "*", "∗": "*", "÷": "/", "−": "-", "–": "-", "—": "-"}
_CMD_OP = {"\\times": "*", "\\cdot": "*", "\\div": "/", "\\ast": "*"}
_FUNC_WORDS = {"sqrt", "sin", "cos", "tan", "log", "ln", "exp", "abs",
               "asin", "acos", "atan", "gcd", "lcm"}
_CURRENCY = {"Rs", "INR", "USD", "EUR", "Re", "Rupees", "Dollars"}
_BINOM = ("\\binom", "\\dbinom", "\\tbinom")
_CMD_FUNCS = {"\\sin": "sin", "\\cos": "cos", "\\tan": "tan", "\\cot": "cot", "\\sec": "sec",
              "\\csc": "csc", "\\log": "log", "\\ln": "ln", "\\exp": "exp",
              "\\arcsin": "asin", "\\arccos": "acos", "\\arctan": "atan"}
_CMD_ALLOWED = set(_CMD_OP) | set(_CMD_FUNCS) | {"\\frac", "\\dfrac", "\\tfrac", "\\sqrt", "\\pi",
                                                 "\\circ", "\\boxed", "\\binom", "\\dbinom",
                                                 "\\tbinom", "\\gcd", "\\lcm", "\\sum", "\\prod"}
_STOP = set("""of and is are was were be been to the an by in on at for with so then thus that which
if as or we it has have had total each every per from into than this these those there here
not no yes can will would should could must may might do does did its their his her our your
what when where who how why also only just more less most least about over under between
after before again once both all any some such other same new old first second third""".split())
_NL_SYMS = set("xyznkpqwtr") | (set("ABCDEFGHJKLMNOPQRSTUVWXYZ"))
_UNIT_TEXT = {"\\text", "\\mbox", "\\mathrm", "\\textbf", "\\mathbf"}


def tokenize(s):
    out = []
    for m in _TOK.finditer(s):
        k, v = m.lastgroup, m.group()
        st, en = m.start(), m.end()
        if k == "NUM":
            out.append(("NUM", v, st, en))
        elif k == "CMD":
            if v in _CMD_OP:
                out.append(("OP", _CMD_OP[v], st, en))
            elif v in ("\\left", "\\right", "\\displaystyle", "\\quad", "\\qquad"):
                out.append(("SP", " ", st, en))
            elif v in ("\\cdots", "\\ldots", "\\dots", "\\cdot\\cdot\\cdot"):
                out.append(("OTHER", "⋯", st, en))
            elif v == "\\approx":
                out.append(("OTHER", "≈", st, en))
            elif v in ("\\le", "\\leq", "\\leqslant"):
                out.append(("OTHER", "≤", st, en))
            elif v in ("\\ge", "\\geq", "\\geqslant"):
                out.append(("OTHER", "≥", st, en))
            elif v == "\\lt":
                out.append(("OTHER", "<", st, en))
            elif v == "\\gt":
                out.append(("OTHER", ">", st, en))
            else:
                out.append(("CMD", v, st, en))
        elif k == "ESC":
            c = v[1]
            if c == "%":
                out.append(("OP", "%", st, en))
            elif c == "$":
                out.append(("DOLLAR", "$", st, en))
            elif c == "_":
                out.append(("OTHER", "_", st, en))
            else:
                out.append(("SP", " ", st, en))
        elif k == "WORD":
            out.append(("ID", v, st, en))
        elif k == "SP":
            out.append(("SP", "\n" if "\n" in v else " ", st, en))
        else:
            c = _OPMAP.get(v, v)
            if c == "$":
                out.append(("DOLLAR", c, st, en))
            elif c == "√":
                out.append(("SQRT", c, st, en))
            elif c == "°":
                out.append(("DEG", c, st, en))
            elif c in "²³":
                out.append(("UNIT", c, st, en))
            elif c in "+-*/^()!{}[]%":
                out.append(("OP", c, st, en))
            else:
                out.append(("OTHER", c, st, en))
    merged, q = [], 0
    while q < len(out):
        a = out[q]
        if a[0] == "OTHER" and a[1] in "<>" and q + 1 < len(out) and out[q + 1][:2] == ("OTHER", "=") \
                and out[q + 1][2] == a[3]:
            merged.append(("OTHER", "≤" if a[1] == "<" else "≥", a[2], out[q + 1][3]))
            q += 2
            continue
        merged.append(a)
        q += 1
    out = merged
    # \text{...} -> UNIT  (\mathrm{gcd} gibi fonksiyon adlari ID kalir)
    res, i, n = [], 0, len(out)
    while i < n:
        t = out[i]
        if t[0] == "CMD" and t[1] in _UNIT_TEXT and i + 1 < n and out[i + 1][:2] == ("OP", "{"):
            j = i + 2
            while j < n and out[j][:2] != ("OP", "}"):
                j += 1
            if j < n:
                inner = [x for x in out[i + 2:j] if x[0] != "SP"]
                if len(inner) == 1 and inner[0][0] == "ID" and inner[0][1] in _FUNC_WORDS:
                    res.append(("ID", inner[0][1], t[2], out[j][3]))
                else:
                    res.append(("UNIT", "", t[2], out[j][3]))
                i = j + 1
                continue
        res.append(t)
        i += 1
    return res


# --------------------------------------------------------------------- parser
class _Fail(Exception):
    pass


def _mk(name, *args):
    return ("call", name, list(args))


def is_call(n):
    return isinstance(n, tuple) and n[0] == "call"


def is_sym(n):
    return isinstance(n, tuple) and n[0] == "sym"


def has_sym(n):
    if is_sym(n):
        return True
    if is_call(n):
        return any(has_sym(a) for a in n[2])
    return False


def count_ops(n):
    if is_call(n):
        return 1 + sum(count_ops(a) for a in n[2])
    return 0


def literal_like(n):
    """Kendisi bir 'sayi yazimi' olan ifade: 3, -3, 3/4, sqrt(3), 2*sqrt(3), pi, 2*pi ..."""
    if isinstance(n, Fraction):
        return True
    if not is_call(n):
        return False
    name, args = n[1], n[2]
    if name == "pi":
        return True
    if name == "neg":
        return literal_like(args[0])
    if name == "div":
        a, b = args
        if isinstance(a, Fraction) and isinstance(b, Fraction):
            import math as _m
            return (a.denominator == 1 and b.denominator == 1 and b > 1
                    and _m.gcd(abs(a.numerator), b.numerator) == 1)
        return all(literal_like(x) for x in args)
    if name == "sqrt":
        return isinstance(args[0], Fraction)
    if name == "mul" and len(args) == 2:
        return isinstance(args[0], Fraction) and is_call(args[1]) and args[1][1] in ("sqrt", "pi") \
            and literal_like(args[1])
    return False


class _Parser:
    def __init__(self, toks, allow_syms=False, defined=()):
        self.t, self.nl = [], []
        pending = False
        for k, v, *_ in toks:
            if k == "SP":
                pending = pending or v == "\n"
                continue
            self.t.append((k, v))
            self.nl.append(pending)
            pending = False
        self.i = 0
        self.end_nonunit = 0
        self.allow_syms = allow_syms
        self.defined = set(defined)
        self.boxed = False

    def sym_ok(self, v):
        if len(v) == 1:
            return v.isalpha() and (self.allow_syms or v in _NL_SYMS)
        return 2 <= len(v) <= 3 and v.isalpha() and v.isupper() and v not in _CURRENCY

    def peek(self, off=0):
        j = self.i + off
        return self.t[j] if j < len(self.t) else (None, None)

    def eat(self, kind=None, val=None):
        k, v = self.peek()
        if k is None or (kind and k != kind) or (val is not None and v != val):
            raise _Fail
        self.i += 1
        return v

    def at(self, kind, val=None):
        k, v = self.peek()
        return k == kind and (val is None or v == val)

    def _is_unit(self):
        if self.i < len(self.nl) and self.nl[self.i]:
            return False
        k, v = self.peek()
        if k == "UNIT":
            return True
        if k == "ID" and v.islower() and v not in _STOP and v not in _FUNC_WORDS and v != "pi" \
                and len(v) <= 14 and not self.sym_ok(v):
            nk, nv = self.peek(1)
            return not (nk == "OP" and nv == "(")
        return False

    def _skip_units(self):
        c = 0
        while c < 3 and self._is_unit():
            self.i += 1
            c += 1
            if self.peek() == ("ID", "per") and self.peek(1)[0] == "ID":
                self.i += 2
            # km/h, slices/pack gibi bolme isaretli birimler
            while self.peek() == ("OP", "/") and self.peek(1)[0] in ("ID", "UNIT") and \
                    self.peek(1)[0] == "ID" and self.peek(1)[1].islower() and \
                    self.peek(1)[1] not in _STOP and self.peek(2) != ("OP", "("):
                self.i += 2
            while self.peek()[0] == "UNIT":
                self.i += 1

    def expr(self):
        acc = self.term()
        pure_add = False
        while self.at("OP", "+") or self.at("OP", "-"):
            if self.at("OP", "+") and self.peek(1) == ("OTHER", "⋯") and self.peek(2) == ("OP", "+"):
                terms = acc[2] if (is_call(acc) and acc[1] == "add" and pure_add) else None
                if terms is None or len(terms) < 2 or not all(isinstance(x, Fraction) for x in terms):
                    raise _Fail
                d = terms[1] - terms[0]
                if d == 0 or any(terms[i + 1] - terms[i] != d for i in range(len(terms) - 1)):
                    raise _Fail
                self.eat(); self.eat(); self.eat()
                last = self.term()
                if not isinstance(last, Fraction):
                    raise _Fail
                cnt = (last - terms[0]) / d + 1
                if cnt.denominator != 1 or cnt < len(terms) + 1 or cnt > 100000:
                    raise _Fail
                acc = _mk("arith_sum", terms[0], d, cnt)
                pure_add = False
                continue
            op = self.eat()
            rhs = self.term()
            if op == "+":
                if pure_add:
                    acc[2].append(rhs)
                else:
                    acc = _mk("add", acc, rhs)
                    pure_add = True
            else:
                acc = _mk("sub", acc, rhs)
                pure_add = False
        return acc

    def _starts_primary(self):
        if self.i < len(self.nl) and self.nl[self.i]:
            return False
        k, v = self.peek()
        if k == "OP" and v in ("(", "{"):
            return True
        if k == "CMD" and (v in ("\\frac", "\\dfrac", "\\tfrac", "\\sqrt", "\\pi") or v in _CMD_FUNCS):
            return True
        if k == "SQRT":
            return True
        if k == "ID" and (v == "pi" or v in _FUNC_WORDS):
            return True
        if k == "ID" and self.sym_ok(v):
            return True
        return False

    def term(self):
        acc = self.unary()
        pure_mul = False
        while True:
            if self.at("OP", "*"):
                self.eat()
                rhs = self.unary()
                if pure_mul:
                    acc[2].append(rhs)
                else:
                    acc = _mk("mul", acc, rhs)
                    pure_mul = True
            elif self.at("OP", "/"):
                self.eat()
                acc = _mk("div", acc, self.unary())
                pure_mul = False
            elif self._starts_primary():
                rhs = self.unary()
                if pure_mul:
                    acc[2].append(rhs)
                else:
                    acc = _mk("mul", acc, rhs)
                    pure_mul = True
            else:
                break
        return acc

    def unary(self):
        if self.at("OP", "-"):
            self.eat()
            v = self.unary()
            return -v if isinstance(v, Fraction) else _mk("neg", v)
        if self.at("OP", "+"):
            self.eat()
            return self.unary()
        return self.power()

    def power(self):
        base = self.postfix()
        if self.at("OP", "^"):
            self.eat()
            if self.at("CMD", "\\circ") or (self.at("OP", "{") and self.peek(1) == ("CMD", "\\circ")):
                if self.at("OP", "{"):
                    self.eat(); self.eat(); self.eat("OP", "}")
                else:
                    self.eat()
                return _mk("deg", base)
            return _mk("pow", base, self.exponent())
        return base

    def exponent(self):
        if self.at("OP", "{"):
            self.eat()
            e = self.expr()
            self.eat("OP", "}")
            return e
        if self.at("OP", "-"):
            self.eat()
            v = self.exponent()
            return -v if isinstance(v, Fraction) else _mk("neg", v)
        base = self.postfix(units=False)
        if self.at("OP", "^"):
            self.eat()
            return _mk("pow", base, self.exponent())
        return base

    def postfix(self, units=True):
        node = self.primary()
        while True:
            if self.at("OP", "!"):
                self.eat()
                node = _mk("fact", node)
            elif self.at("DEG"):
                self.eat()
                node = _mk("deg", node)
            elif self.at("OP", "%"):
                self.eat()
                if self.at("OP", "*"):
                    node = _mk("div", node, Fraction(100))
            else:
                break
        self.end_nonunit = self.i
        if units:
            self._skip_units()
        return node

    def _group(self, open_, close):
        self.eat("OP", open_)
        e = self.expr()
        self.eat("OP", close)
        return e

    def primary(self):
        k, v = self.peek()
        if k == "DOLLAR":
            self.eat()
            return self.primary()
        if k == "CMD" and v in ("\\sum", "\\prod"):
            self.eat()
            self.eat("OTHER", "_")
            self.eat("OP", "{")
            var = self.eat("ID")
            if len(var) != 1:
                raise _Fail
            self.eat("OTHER", "≔")
            lo = self.expr()
            self.eat("OP", "}")
            self.eat("OP", "^")
            hi = self._group("{", "}") if self.at("OP", "{") else self.primary()
            body = self.term()
            if not (isinstance(lo, Fraction) and isinstance(hi, Fraction)):
                raise _Fail
            return _mk("sum_range" if v == "\\sum" else "prod_range", infix(body), var, lo, hi)
        if k == "NUM" and v.isdigit() and self.peek(1) == ("OTHER", "_"):
            if self.peek(2)[0] == "NUM":
                bt, adv = self.peek(2)[1], 3
            elif self.peek(2) == ("OP", "{") and self.peek(3)[0] == "NUM" and self.peek(4) == ("OP", "}"):
                bt, adv = self.peek(3)[1], 5
            else:
                bt, adv = None, 0
            if bt is not None and bt.isdigit() and 2 <= int(bt) <= 16:
                try:
                    int(v, int(bt))
                except ValueError:
                    raise _Fail
                self.i += adv
                return _mk("from_base", v, Fraction(int(bt)))
        if k == "NUM":
            if (v.isdigit() and self.peek(1)[0] == "NUM" and self.peek(1)[1].isdigit()
                    and self.peek(2) == ("OP", "/") and self.peek(3)[0] == "NUM" and self.peek(3)[1].isdigit()
                    and not self.nl[self.i + 1] and int(self.peek(1)[1]) < int(self.peek(3)[1])):
                w = Fraction(int(v))
                a = Fraction(int(self.peek(1)[1]))
                b = Fraction(int(self.peek(3)[1]))
                self.i += 4
                return _mk("add", w, _mk("div", a, b))
            self.eat()
            return Fraction(Decimal(v.replace(",", "").replace("\\!", "").replace("{", "").replace("}", "")))
        if k == "OP" and v == "(":
            return self._group("(", ")")
        if k == "OP" and v == "{":
            return self._group("{", "}")
        if k == "OTHER" and v == "|":
            self.eat()
            e = self.expr()
            self.eat("OTHER", "|")
            return _mk("abs", e)
        if k == "CMD" and v == "\\boxed":
            self.eat()
            e = self._group("{", "}")
            self.boxed = True
            return e
        if k == "CMD" and v in ("\\frac", "\\dfrac", "\\tfrac"):
            self.eat()
            if self.at("NUM"):
                d = self.eat()
                if len(d) >= 2 and d.isdigit():
                    return _mk("div", Fraction(int(d[0])), Fraction(int(d[1:])))
                a = Fraction(Decimal(d))
                return _mk("div", a, self.primary())
            a = self._group("{", "}")
            b = self._group("{", "}")
            return _mk("div", a, b)
        if k == "CMD" and v in _BINOM:
            self.eat()
            a = self._group("{", "}")
            b = self._group("{", "}")
            return _mk("comb", a, b)
        if (k == "ID" and v in ("gcd", "lcm")) or (k == "CMD" and v in ("\\gcd", "\\lcm")):
            name = v.lstrip("\\")
            self.eat()
            self.eat("OP", "(")
            args = [self.expr()]
            while self.at("OTHER", ","):
                self.eat()
                args.append(self.expr())
            self.eat("OP", ")")
            if len(args) < 2:
                raise _Fail
            return _mk(name, *args)
        if k == "CMD" and v == "\\sqrt":
            self.eat()
            if self.at("OP", "["):
                self.eat()
                n = self.expr()
                self.eat("OP", "]")
                a = self._group("{", "}")
                return _mk("root", n, a)
            if self.at("OP", "{"):
                return _mk("sqrt", self._group("{", "}"))
            return _mk("sqrt", self.primary())
        if k == "SQRT":
            self.eat()
            if self.at("OP", "("):
                return _mk("sqrt", self._group("(", ")"))
            if self.at("NUM"):
                return _mk("sqrt", self.primary())
            raise _Fail
        if (k == "CMD" and v == "\\pi") or (k == "ID" and v == "pi"):
            self.eat()
            return _mk("pi")
        fname = None
        if k == "CMD" and v in _CMD_FUNCS:
            fname = _CMD_FUNCS[v]
        elif k == "ID" and v in _FUNC_WORDS:
            fname = v
        if fname == "log" and self.peek(1) == ("OTHER", "_"):
            self.eat(); self.eat()
            if self.at("OP", "{"):
                base = self._group("{", "}")
            else:
                base = self.primary()
            if self.at("OP", "("):
                arg = self._group("(", ")")
            elif self.at("OP", "{"):
                arg = self._group("{", "}")
            else:
                arg = self._arg_noexp()
            return _mk("log", arg, base)
        if fname:
            self.eat()
            if self.at("OP", "("):
                arg = self._group("(", ")")
            elif self.at("OP", "{"):
                arg = self._group("{", "}")
            else:
                arg = self._arg_noexp()
            return _mk(fname, arg)
        if k == "ID" and v in _CURRENCY:
            self.eat()
            if self.at("OTHER", "."):
                self.eat()
            return self.primary()
        if k == "ID" and (len(v) == 1 or self.sym_ok(v)) and v.isalpha():
            if v in self.defined and self.peek(1) == ("OP", "("):
                self.eat()
                a = self._group("(", ")")
                return _mk(v, a)
            if self.sym_ok(v):
                self.eat()
                return ("sym", v + self._subscript())
        raise _Fail

    def _subscript(self):
        if not self.at("OTHER", "_"):
            return ""
        self.eat()
        if self.at("OP", "{"):
            self.eat()
            parts = []
            while not self.at("OP", "}"):
                k, v = self.peek()
                if k is None or len(parts) > 6:
                    raise _Fail
                self.i += 1
                parts.append({"+": "p", "-": "m", "*": "t"}.get(v, v) if k == "OP" else v)
            self.eat("OP", "}")
            return "_" + re.sub(r"\W", "", "".join(parts))
        k, v = self.peek()
        if k in ("NUM", "ID"):
            self.i += 1
            return "_" + re.sub(r"\W", "", v)
        raise _Fail

    def _arg_noexp(self):
        node = self.primary()
        if self.at("DEG"):
            self.eat()
            return _mk("deg", node)
        if self.at("OP", "^") and (self.peek(1) == ("CMD", "\\circ") or
                                   (self.peek(1) == ("OP", "{") and self.peek(2) == ("CMD", "\\circ"))):
            self.eat()
            if self.at("OP", "{"):
                self.eat(); self.eat(); self.eat("OP", "}")
            else:
                self.eat()
            return _mk("deg", node)
        return node


def parse_full(toks, allow_syms=False, defined=()):
    p = _Parser(toks, allow_syms, defined)
    try:
        n = p.expr()
    except (_Fail, IndexError):
        return None
    return n if p.i == len(p.t) else None


def parse_prefix(toks, allow_syms=False, defined=()):
    """RHS: bir ifade ayristirir. (node, tuketilen_token_sayisi(birimsiz, SP dahil), boxed)"""
    nonsp = [i for i, t in enumerate(toks) if t[0] != "SP"]
    p = _Parser(toks, allow_syms, defined)
    try:
        n = p.expr()
    except (_Fail, IndexError):
        return None, 0, False
    if p.end_nonunit == 0:
        return None, 0, False
    return n, nonsp[p.end_nonunit - 1] + 1, p.boxed


# --------------------------------------------------------- cikti bicimleyiciler
def render(n):
    if isinstance(n, Fraction):
        return fmt_frac(n)
    if isinstance(n, str):
        return json.dumps(n)
    _, name, args = n
    return f"{name}({', '.join(render(a) for a in args)})"


_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2, "neg": 2.5, "pow": 3}


def infix(n, parent=0):
    """Sembolik arguman icin sympy'nin okuyabilecegi duz metin."""
    if isinstance(n, Fraction):
        if n.denominator == 1:
            return str(n.numerator) if n >= 0 or parent < 2 else f"({n.numerator})"
        return f"({n.numerator}/{n.denominator})"
    if is_sym(n):
        return n[1]
    _, name, a = n
    if name in ("add", "mul"):
        op = " + " if name == "add" else "*"
        s = op.join(infix(x, _PREC[name]) for x in a)
        return f"({s})" if parent > _PREC[name] else s
    if name == "sub":
        s = f"{infix(a[0], 1)} - {infix(a[1], 2)}"
        return f"({s})" if parent > 1 else s
    if name == "div":
        s = f"{infix(a[0], 2)}/{infix(a[1], 3)}"
        return f"({s})" if parent > 2 else s
    if name == "neg":
        s = f"-{infix(a[0], 3)}"
        return f"({s})" if parent > 2 else s
    if name == "pow":
        return f"{infix(a[0], 4)}^{infix(a[1], 4)}" if parent <= 3 else f"({infix(a[0], 4)}^{infix(a[1], 4)})"
    if name == "fact":
        return f"factorial({infix(a[0])})"
    if name == "pi":
        return "pi"
    if name == "deg":
        return f"({infix(a[0], 2)})*pi/180"
    return f"{name}({', '.join(infix(x) for x in a)})"


def to_sympy(n):
    if isinstance(n, Fraction):
        return sympy.Rational(n.numerator, n.denominator)
    if is_sym(n):
        return sympy.Symbol(n[1])
    _, name, a = n
    if name == "from_base":
        return sympy.Integer(int(a[0], int(a[1])))
    if name in ("sum_range", "prod_range"):
        return _back_sym(run(n))
    v = [to_sympy(x) for x in a]
    def _pw():
        if v[1].is_number and (abs(v[1]) > 64 or (v[1].is_Rational and v[1].q > 8)):
            raise ToolError("us cok buyuk")
        if v[1].free_symbols and (v[0].free_symbols or v[1].count_ops() > 6):
            raise ToolError("sembolik us")
        return v[0] ** v[1]

    m = {"add": lambda: sympy.Add(*v), "sub": lambda: v[0] - v[1], "mul": lambda: sympy.Mul(*v),
         "div": lambda: v[0] / v[1], "pow": _pw, "neg": lambda: -v[0],
         "fact": lambda: sympy.factorial(v[0]), "sqrt": lambda: sympy.sqrt(v[0]),
         "root": lambda: sympy.root(v[1], v[0]), "pi": lambda: sympy.pi,
         "deg": lambda: v[0] * sympy.pi / 180,
         "arith_sum": lambda: v[2] * (2 * v[0] + (v[2] - 1) * v[1]) / 2, "ln": lambda: sympy.log(v[0]),
         "log": lambda: sympy.log(v[0], v[1] if len(v) > 1 else 10), "exp": lambda: sympy.exp(v[0]), "abs": lambda: sympy.Abs(v[0])}
    for f in ("sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan"):
        m[f] = (lambda f=f: getattr(sympy, f)(v[0]))
    if name not in m:
        raise ToolError(f"to_sympy: {name}")
    return m[name]()


def _back_sym(x):
    return sympy.Rational(x.numerator, x.denominator) if isinstance(x, Fraction) else x


def _finite_ends(sol):
    out = []
    if isinstance(sol, sympy.Union):
        for a in sol.args:
            out += _finite_ends(a)
    elif isinstance(sol, sympy.Interval):
        for e in (sol.start, sol.end):
            if e.is_Rational:
                out.append(Fraction(int(e.p), int(e.q)))
    elif isinstance(sol, sympy.FiniteSet):
        for e in sol.args:
            if e.is_Rational:
                out.append(Fraction(int(e.p), int(e.q)))
    return out


def _same(a, b):
    try:
        if isinstance(a, Fraction):
            a = sympy.Rational(a.numerator, a.denominator)
        if isinstance(b, Fraction):
            b = sympy.Rational(b.numerator, b.denominator)
        d = a - b
        if d == 0:
            return True
        if not d.free_symbols:
            return abs(sympy.N(d, 40)) < sympy.Float("1e-30")
        if d.count_ops() > 40:
            return False
        d = sympy.expand(d)
        if d == 0:
            return True
        if d.has(sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log, sympy.factorial,
                 sympy.Abs, sympy.floor, sympy.ceiling):
            return False
        return sympy.cancel(d) == 0
    except Exception:
        return False


def _tag(call_str, result_str, prefix=""):
    return f"{prefix}{CALL_OPEN}{call_str}{CALL_CLOSE}{RES_OPEN}{result_str}{RES_CLOSE}"


# ----------------------------------------------------------- on isleme (uzunluk koruyan)
_GSM_RE = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
_LATEX_HINT = re.compile(r"\\(?:frac|dfrac|sqrt|times|cdot|boxed|text|left|right|div|pi|ldots|cdots|"
                         r"le|ge|neq|pm|sin|cos|tan|log|ln|binom|mathrm|begin|circ|approx|"
                         r"Rightarrow|implies)\b|\^\{|\\\(|\\\[")
_REGION = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
_ALLOWED_OPS = set("+-*/^(){}!%")
_X_MULT = re.compile(r"(?<=[\d)])(\s*|\s+(?:[A-Za-z]+\s+){1,2})[xX](\s*)(?=[\d($])")
_CODE_HINT = re.compile(r"^\s*(print\(|def |import |from \w+ import |for .+ in |return )", re.M)


def _pad(rep):
    return lambda m: rep + " " * (len(m.group(0)) - len(rep))


_WORD_SUBS = [
    (re.compile(r"(?<=[\d)}])\s+divided by\s+(?=[\d(\\$-])", re.I), _pad("/")),
    (re.compile(r"(?<=[\d)}])\s+(?:multiplied by|times)\s+(?=[\d(\\$-])", re.I), _pad("*")),
    (re.compile(r"(?<=[\d)}])\s+plus\s+(?=[\d(\\$-])", re.I), _pad("+")),
    (re.compile(r"(?<=[\d)}])\s+minus\s+(?=[\d(\\$-])", re.I), _pad("-")),
    (re.compile(r"(?<=[\d)}])[\s,]*(?:which equals|which is equal to|is equal to|equals|equal to|gives)\s+"
                r"(?=[\d(\\$-])", re.I), _pad("=")),
    (re.compile(r"(?<=%)\s+of\s+(?=[\d(\\$])", re.I), _pad("*")),
]


_ASY = re.compile(r"\[asy\].*?\[/asy\]", re.S)


_IS_EQ = re.compile(r"(\d[\d.,]*\s*[+\-*/]\s*\d[\d.,]*(?:\s*[+\-*/]\s*\d[\d.,]*)*)(\s+is\s+)(?=[\d$])")


def _is_to_eq(m):
    sp = m.group(2)
    return m.group(1) + " " + "=" + " " * (len(sp) - 2)


_SUM_HEAD = re.compile(r"(\\(?:sum|prod|lim)\s*_\s*\{[^{}=]*)=")


def _make_scan(text):
    text = _ASY.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    s = _X_MULT.sub(lambda m: m.group(1) + "*" + m.group(2), text)
    s = s.replace("&", " ")
    s = _SUM_HEAD.sub(lambda m: m.group(1) + "≔", s)
    s = s.replace("...", "⋯  ").replace("…", "⋯")
    for rx, rep in _WORD_SUBS:
        s = rx.sub(rep, s)
    s = _IS_EQ.sub(_is_to_eq, s)
    return s


def _dollar_math(text):
    for m in re.finditer(r"\$([^$\n]{1,200}?)\$", text):
        inner = re.sub(r"\\[A-Za-z]+", "", m.group(1))
        if re.search(r"[A-Za-z]{2,}", inner):
            continue
        if re.search(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", inner) and re.search(r"[=^_(]", inner):
            return True
    return False


_DBG = []


def _to_frac(v):
    r = sympy.Rational(str(sympy.N(v, 40)))
    return Fraction(int(r.p), int(r.q))


def _new_stats():
    return dict(found=0, converted=0, mismatch=0)


def _fix_gsm_annotations(text, st):
    out, pos = [], 0
    for m in _GSM_RE.finditer(text):
        expr, res = m.group(1).strip(), m.group(2).strip()
        pre = text[pos:m.start()]
        post_start = m.end()
        node = parse_full(tokenize(expr))
        ok, val = False, None
        if node is not None and is_call(node):
            rnode = parse_full(tokenize(res))
            try:
                val = run(node)
                ok = rnode is not None and _same(val, run(rnode))
            except Exception:
                ok = False
        st["found"] += 1
        if ok:
            pre2 = pre
            body = pre.rstrip()
            if body.endswith("="):
                before = body[:-1].rstrip()
                nrm = lambda z: re.sub(r"[\s$]", "", z)
                for k in range(max(1, len(expr) - 3), len(expr) + 8):
                    if k <= len(before) and nrm(before[-k:]) == nrm(expr):
                        pre2 = before[:-k]
                        break
            out.append(pre2)
            rest = text[post_start:]
            dup = re.match(re.escape(res) + r"(?![\d.])", rest)
            out.append(_tag(render(node), to_text(val)))
            st["converted"] += 1
            pos = post_start + (dup.end() if dup else 0)
        else:
            st["mismatch"] += 1
            out.append(pre)
            pos = post_start
    out.append(text[pos:])
    return "".join(out)


_ENV_REGION = re.compile(r"\\begin\{(align|equation|gather|eqnarray|aligned|array)(\*?)\}(.+?)"
                         r"\\end\{\1\2\}", re.S)


def _regions(text):
    regs = []
    for m in _ENV_REGION.finditer(text):
        regs.append((m.start(), m.end(), m.start(3), m.end(3), text[m.start():m.start(3)],
                     text[m.end(3):m.end()]))
    taken = [(r[0], r[1]) for r in regs]
    for m in _REGION.finditer(text):
        if any(a <= m.start() and m.end() <= b for a, b in taken):
            continue
        g = next(i for i in range(1, 5) if m.group(i) is not None)
        regs.append((m.start(), m.end(), m.start(g), m.end(g), text[m.start():m.start(g)],
                     text[m.end(g):m.end()]))
    regs.sort(key=lambda r: r[0])
    return regs


def _nums_after(text, pos):
    vals = set()
    for m in re.finditer(r"(?<![\d.])\d+(?:\.\d+)?", text[pos:]):
        try:
            vals.add(Fraction(Decimal(m.group(0))))
            vals.add(-Fraction(Decimal(m.group(0))))
        except Exception:
            pass
    for m in re.finditer(r"\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", text[pos:]):
        if int(m.group(2)):
            f = Fraction(int(m.group(1)), int(m.group(2)))
            vals.add(f)
            vals.add(-f)
    return vals


def convert_text(text: str, aggressive: bool = False, lang: str = None):
    """(yeni_metin, istatistik). lang: 'tr' (varsayilan) veya 'en' (kanonik etiket/adlar)."""
    lang = lang or _P.LANG
    with _P.use_language(lang):
        out, st = _convert_text_impl(text, aggressive)
    return _P.localize(out, lang), st


def _convert_text_impl(text: str, aggressive: bool = False):
    st = _new_stats()
    if _CODE_HINT.search(_ASY.sub("", text)):
        return text, st
    if "<<" in text:
        text = _fix_gsm_annotations(text, st)
    latex_mode = bool(_LATEX_HINT.search(text)) or _dollar_math(text)
    regions = _regions(text) if latex_mode else []
    starts = [r[0] for r in regions]

    def region_at(cs, ce):
        k = bisect.bisect_right(starts, cs) - 1
        if k >= 0 and regions[k][0] <= cs and ce <= regions[k][1]:
            return k
        return None

    scan = _make_scan(text)
    toks = tokenize(scan)
    n = len(toks)
    env = {}              # tanimli fonksiyonlar: ad -> (vars, sympy_body)
    edits = []            # (cs, ce, call_str, result_str, prefix, boxed_result)
    solve_cands = []      # (cs, ce, node_l, node_r, after_pos)
    last_end = 0

    def allowed_left(t, pos_in_region):
        k, v = t[0], t[1]
        if k in ("NUM", "SP", "SQRT", "DEG", "UNIT"):
            return True
        if k == "OP":
            return v in _ALLOWED_OPS
        if k == "OTHER":
            return v in (",", "|", "_", "⋯", "≔")
        if k == "DOLLAR":
            return not latex_mode
        if k == "CMD":
            return v in _CMD_ALLOWED
        if k == "ID":
            if v in _FUNC_WORDS or v == "pi" or v in _CURRENCY:
                return True
            if len(v) == 1 and v.isalpha() and ((latex_mode and pos_in_region) or v in _NL_SYMS):
                return True
            if 2 <= len(v) <= 3 and v.isupper() and v.isalpha():
                return True
            if v.islower() and v not in _STOP and len(v) <= 14:
                return True   # birim kelimesi olabilir
            return False
        return False

    for i, t in enumerate(toks):
        if t[0] != "OTHER" or t[1] not in ("=", "≈", "<", ">", "≤", "≥"):
            continue
        approx = t[1] == "≈"
        rel = t[1] in ("<", ">", "≤", "≥")
        if rel:
            prv = toks[i - 1] if i > 0 else None
            nx = toks[i + 1] if i + 1 < n else None
            if (prv and prv[0] in ("OP", "OTHER") and prv[1] in ("-", "=", "<", ">")) or \
               (nx and nx[0] == "OTHER" and nx[1] in ("<", ">")):
                continue
        if i + 1 < n and toks[i + 1][0] == "OTHER" and toks[i + 1][1] == "=":
            continue
        if i > 0 and toks[i - 1][0] == "OTHER" and toks[i - 1][1] in "=<>!≠":
            continue
        in_reg = region_at(t[2], t[3]) is not None
        allow_syms = latex_mode and in_reg
        j = i + 1
        while j < n and toks[j][0] == "SP":
            j += 1
        if j >= n:
            continue
        rhs, used, boxed = parse_prefix(toks[j:], allow_syms, env.keys())
        if rhs is None:
            continue
        r_end = j + used
        nxt = r_end
        while nxt < n and toks[nxt][0] == "SP":
            nxt += 1
        if nxt < n:
            k2, v2 = toks[nxt][0], toks[nxt][1]
            gap = scan[toks[r_end - 1][3]:toks[nxt][2]]
            if "\n" not in gap and ((k2 == "OP" and v2 in "^(!") or k2 in ("NUM", "DEG", "SQRT") or
               (k2 == "CMD" and v2 in _CMD_ALLOWED and v2 not in ("\\circ", "\\boxed"))):
                continue
        chain_next = nxt < n and toks[nxt][0] == "OTHER" and toks[nxt][1] == "="
        # sol taraf
        L = i
        while L - 1 >= 0 and allowed_left(toks[L - 1], region_at(toks[L - 1][2], toks[L - 1][3]) is not None):
            L -= 1
        while L < i and toks[L][0] == "SP":
            L += 1
        found = None
        for s in range(L, min(i, L + 60)):
            if toks[s][0] == "SP":
                continue
            lhs_toks = toks[s:i]
            # f(x) = ...  tanimi
            nonsp = [x for x in lhs_toks if x[0] != "SP"]
            if (allow_syms and len(nonsp) >= 4 and nonsp[0][0] == "ID" and len(nonsp[0][1]) == 1
                    and nonsp[1][:2] == ("OP", "(") and nonsp[-1][:2] == ("OP", ")")
                    and all(x[0] == "ID" and len(x[1]) == 1 or x[:2] == ("OTHER", ",") for x in nonsp[2:-1])
                    and not (s > 0 and toks[s - 1][0] == "ID")):
                found = (s, ("define", nonsp[0][1], [x[1] for x in nonsp[2:-1] if x[0] == "ID"]))
                break
            node = parse_full(lhs_toks, allow_syms, env.keys())
            if node is None or isinstance(node, Fraction) or is_sym(node):
                continue
            q = s - 1
            while q >= 0 and toks[q][0] == "SP":
                q -= 1
            if q >= 0:
                kq, vq = toks[q][0], toks[q][1]
                adjacent = (q == s - 1)
                fk, fv = toks[s][0], toks[s][1]
                if kq == "ID" and (fk == "OP" and fv in "+-*/^" or adjacent):
                    continue
                if kq == "OP" and vq in (")", "}", "^", "_") and fk == "OP":
                    continue
                if kq == "OTHER" and vq == "_":
                    continue
                if adjacent and kq in ("NUM", "ID"):
                    continue
                if kq == "NUM" and fk == "NUM":
                    continue
            found = (s, node)
            break
        if found is None and rel:
            continue
        if found is None:
            # "Sayi = 45 - 21" gibi: sol taraf etiket/degisken, sag taraf sayisal islem
            if approx or boxed or has_sym(rhs) or not is_call(rhs) or literal_like(rhs):
                continue
            if nxt < n and toks[nxt][0] == "OTHER" and toks[nxt][1] in ("=", "≈"):
                continue
            cs = toks[j][2]
            ce = toks[r_end - 1][3]
            if cs < last_end:
                continue
            try:
                v_ = run(rhs, env)
            except Exception:
                continue
            if isinstance(v_, str):
                continue
            st["found"] += 1
            edits.append((cs, ce, render(rhs), to_text(v_), "$" if toks[j][0] == "DOLLAR" and not latex_mode
                          else "", None))
            last_end = ce
            st["converted"] += 1
            continue
        s, lhs = found
        cs = toks[s][2]
        ce = toks[r_end - 1][3]
        if cs < last_end:
            continue
        st["found"] += 1

        # ---- fonksiyon tanimi
        if isinstance(lhs, tuple) and lhs and lhs[0] == "define":
            if not has_sym(rhs) or boxed:
                continue
            try:
                body = infix(rhs)
                vars_ = lhs[2]
                env[lhs[1]] = (vars_, _sym(body))
                edits.append((cs, ce, f'define("{lhs[1]}", "{", ".join(vars_)}", "{body}")',
                              f"{lhs[1]}({', '.join(vars_)}) = {body}", "", None))
                last_end = ce
                st["converted"] += 1
            except Exception:
                pass
            continue

        sym_l, sym_r = has_sym(lhs), has_sym(rhs)
        call_str = res_str = None
        if rel and not (sym_l or sym_r):
            continue
        try:
            if not sym_l and not sym_r:
                lv = run(lhs, env)
                rv = run(rhs, env)
                if approx:
                    if not (isinstance(rhs, Fraction) and isinstance(lv, (Fraction,)) or True):
                        continue
                    rtxt = text[toks[j][2]:toks[r_end - 1][3]]
                    mm = re.search(r"\.(\d+)", rtxt)
                    nd = len(mm.group(1)) if mm else 0
                    if not isinstance(rhs, Fraction):
                        continue
                    lvf = lv if isinstance(lv, Fraction) else Fraction(float(sympy.N(lv, 30)))
                    if _round_half_up(lvf, nd) != rhs:
                        st["mismatch"] += 1
                        continue
                    call_str = f"round({render(lhs)}, {nd})"
                    res_str = to_text(rhs)
                else:
                    if not _same(lv, rv):
                        rtxt = text[toks[j][2]:toks[r_end - 1][3]]
                        mm = re.search(r"\.(\d+)", rtxt)
                        if isinstance(rhs, Fraction) and mm and not boxed or (isinstance(rhs, Fraction) and mm):
                            nd = len(mm.group(1))
                            try:
                                lvf = lv if isinstance(lv, Fraction) else _to_frac(lv)
                                if _round_half_up(lvf, nd) == rhs and is_call(lhs):
                                    edits.append((cs, ce, f"round({render(lhs)}, {nd})", to_text(rhs), "",
                                                  to_text(rv) if boxed else None))
                                    last_end = ce
                                    st["converted"] += 1
                                    continue
                            except Exception:
                                pass
                        if isinstance(rhs, Fraction) and rhs.denominator == 1 and is_call(lhs):
                            try:
                                import math as _m
                                lvf = lv if isinstance(lv, Fraction) else _to_frac(lv)
                                if lvf.denominator != 1 and abs(lvf - rhs) < 1:
                                    if _round_half_up(lvf, 0) == rhs:
                                        call_i = f"round({render(lhs)}, 0)"
                                    elif Fraction(_m.floor(lvf)) == rhs:
                                        call_i = f"floor({render(lhs)})"
                                    else:
                                        call_i = None
                                    if call_i:
                                        edits.append((cs, ce, call_i, to_text(rhs), "",
                                                      to_text(rv) if boxed else None))
                                        last_end = ce
                                        st["converted"] += 1
                                        continue
                            except Exception:
                                pass
                        st["mismatch"] += 1
                        continue
                    if not literal_like(rhs) and not boxed:
                        # iki ifade de islem iceriyor: esitligi araca dogrulat
                        if not is_call(rhs) or count_ops(rhs) >= count_ops(lhs) + 3 or chain_next:
                            continue
                        call_str = f'equal("{infix(lhs)}", "{infix(rhs)}")'
                        res_str = to_text(True)
                    else:
                        call_str = render(lhs)
                        res_str = to_text(lv)
            else:
                if rel:
                    L_, R_ = to_sympy(lhs), to_sympy(rhs)
                    fs = (L_ - R_).free_symbols
                    if len(fs) != 1 or boxed:
                        continue
                    var = next(iter(fs))
                    diff_ = sympy.together(L_ - R_)
                    if diff_.count_ops() > 50:
                        continue
                    if sympy.Poly(sympy.fraction(diff_)[0], var).degree() > 4:
                        continue
                    relcls = {"<": sympy.Lt, ">": sympy.Gt, "≤": sympy.Le, "≥": sympy.Ge}[t[1]]
                    sol = sympy.solve_univariate_inequality(relcls(L_, R_), var, relational=False)
                    ends = _finite_ends(sol)
                    seen = _nums_after(text, ce)
                    if not ends or not any(e_ in seen for e_ in ends):
                        continue
                    opstr = {"<": "<", ">": ">", "≤": "<=", "≥": ">="}[t[1]]
                    call_ = f'solve("{infix(lhs)} {opstr} {infix(rhs)}", "{var}")'
                    res_ = execute(call_)
                    edits.append((cs, ce, call_, res_, "", None))
                    last_end = ce
                    st["converted"] += 1
                    continue
                if approx or isinstance(lhs, Fraction):
                    continue
                L_, R_ = to_sympy(lhs), to_sympy(rhs)
                if _same(L_, R_):
                    # ozdeslik: LHS'i en uygun fonksiyonla yeniden yaz, RHS ile ayni olmali
                    if sym_l and not sym_r:
                        continue
                    lhs_s = infix(lhs)
                    for fn, f in (("expand", sympy.expand), ("factor", sympy.factor),
                                  ("simplify", sympy.simplify)):
                        cand = f(L_)
                        if cand != L_ and _same(cand, R_):
                            call_str = f'{fn}("{lhs_s}")'
                            res_str = to_text(cand)
                            break
                    if call_str is None:
                        if (not is_call(rhs) and not has_sym(rhs)) or chain_next:
                            continue
                        call_str = f'equal("{lhs_s}", "{infix(rhs)}")'
                        res_str = to_text(True)
                else:
                    # denklem: solve adayi
                    fs = (L_ - R_).free_symbols
                    if len(fs) == 1 and (is_call(lhs) or is_call(rhs)) and not boxed:
                        solve_cands.append((cs, ce, lhs, rhs, ce, next(iter(fs))))
                    continue
        except Exception:
            continue
        edits.append((cs, ce, call_str, res_str, "", to_text(rv) if boxed and not sym_l and not sym_r
                      and not approx else None))
        # $5 * 3 = $15 bicimi (para birimi)
        if not latex_mode and toks[s][0] == "DOLLAR":
            edits[-1] = edits[-1][:4] + ("$",) + edits[-1][5:]
        last_end = ce
        st["converted"] += 1

    # ---- denklem cozme (solve): sonucu metinde sonradan gecen kokler
    taken = [(e[0], e[1]) for e in edits]
    for cs, ce, lhs, rhs, after, var in solve_cands:
        if any(a < ce and cs < b for a, b in taken):
            continue
        try:
            le, re_ = to_sympy(lhs), to_sympy(rhs)
            diff_ = sympy.together(le - re_)
            if diff_.count_ops() > 50:
                continue
            numer = sympy.fraction(diff_)[0]
            if sympy.Poly(numer, var).degree() > 4:
                continue
            sols = sympy.solve(sympy.Eq(le, re_), var)
        except Exception:
            continue
        if not sols or len(sols) > 4 or not all(getattr(s_, "is_real", False) for s_ in sols):
            continue
        seen = _nums_after(text, after)
        rat = [Fraction(int(s_.p), int(s_.q)) for s_ in sols if s_.is_Rational]
        if not rat or not any(r in seen for r in rat):
            continue
        sols = sorted(sols, key=lambda z: float(z))
        call = f'solve("{infix(lhs)} = {infix(rhs)}", "{var}")'
        try:
            res = execute(call)
        except Exception:
            continue
        edits.append((cs, ce, call, res, "", None))
        taken.append((cs, ce))
        st["converted"] += 1

    # ---- agresif mod: "=" olmayan, yalnizca sayilardan olusan islem ifadeleri
    if aggressive:
        occupied = [(e[0], e[1]) for e in edits]
        k = 0
        while k < n:
            tk = toks[k]
            is_start = tk[0] == "NUM" or (tk[0] == "OP" and tk[1] == "(")
            if not is_start or any(a < tk[3] and tk[2] < b for a, b in occupied):
                k += 1
                continue
            if k > 0 and toks[k - 1][0] in ("ID", "NUM") or (tk[2] > 0 and scan[tk[2] - 1] in "_."):
                k += 1
                continue
            j = k
            while j < n:
                tj = toks[j]
                if tj[0] in ("NUM", "DEG", "SQRT"):
                    j += 1
                elif tj[0] == "SP":
                    if "\n" in scan[tj[2]:tj[3]]:
                        break
                    j += 1
                elif tj[0] == "OP" and tj[1] in "+-*/^()!":
                    j += 1
                else:
                    break
            end, done = j, False
            while end - k >= 3:
                while end > k and (toks[end - 1][0] == "SP" or
                                   (toks[end - 1][0] == "OP" and toks[end - 1][1] in "+-*/^(")):
                    end -= 1
                if end - k < 3:
                    break
                node = parse_full(toks[k:end], False, env.keys())
                if node is not None and is_call(node) and not literal_like(node) and count_ops(node) >= 1:
                    cs_, ce_ = toks[k][2], toks[end - 1][3]
                    seg = text[cs_:ce_]
                    pk = k - 1
                    while pk >= 0 and toks[pk][0] == "SP" and "\n" not in scan[toks[pk][2]:toks[pk][3]]:
                        pk -= 1
                    nk = end
                    while nk < n and toks[nk][0] == "SP" and "\n" not in scan[toks[nk][2]:toks[nk][3]]:
                        nk += 1
                    prev_t = toks[pk] if pk >= 0 else None
                    next_t = toks[nk] if nk < n else None

                    def _ok_neighbor(t_):
                        if t_ is None or t_[0] in ("SP", "DOLLAR", "UNIT"):
                            return True
                        if t_[0] == "OTHER":
                            return t_[1] in ",;:.=≈?!"
                        if t_[0] == "ID":
                            return t_[1].islower() and len(t_[1]) > 1
                        if t_[0] == "OP":
                            return t_[1] in "%" and False
                        return False

                    bad = (
                        re.fullmatch(r"\d{1,4}\s*[-/]\s*\d{1,4}(\s*[-/]\s*\d{1,4})?", seg) is not None
                        and " " not in seg and not latex_mode
                    ) or any(a < ce_ and cs_ < b for a, b in occupied) \
                        or region_at(cs_, ce_) is not None \
                        or not _ok_neighbor(prev_t) or not _ok_neighbor(next_t)
                    if not bad:
                        try:
                            v_ = run(node, env)
                        except Exception:
                            v_ = None
                        if v_ is not None and not isinstance(v_, str) and len(to_text(v_)) <= 30:
                            edits.append((cs_, ce_, render(node), to_text(v_), "", None))
                            occupied.append((cs_, ce_))
                            st["converted"] += 1
                            done = True
                            break
                end -= 1
            k = end if done else k + 1

    # ---- $...$ icinde tek basina duran sayisal ifade
    for (a, b, ia, ib, op, cl) in regions:
        if any(x < b and a < y for x, y in taken):
            continue
        inner = text[ia:ib]
        if "=" in inner or not re.search(r"\d", inner):
            continue
        tk = tokenize(_make_scan(inner))
        node = parse_full(tk, False, env.keys())
        if node is None or not is_call(node) or literal_like(node) or count_ops(node) < 1:
            continue
        try:
            v = run(node, env)
        except Exception:
            continue
        if isinstance(v, str):
            continue
        edits.append((a, b, render(node), to_text(v), "", "REGION"))
        taken.append((a, b))
        st["converted"] += 1

    if not edits:
        return text, st
    # son guvenlik: ciktidaki her cagri yeniden calistirilir, tutmayanlar geri alinir
    for _ in range(4):
        out = _apply(text, edits, regions, region_at)
        env2, failed = {}, set()
        for m in _TAG_RE.finditer(out):
            try:
                if to_text(run(parse_call(m.group(1)), env2)) != m.group(2):
                    failed.add(m.group(1))
            except Exception:
                failed.add(m.group(1))
        if not failed:
            break
        edits = [e for e in edits if e[2] not in failed]
        st["converted"] = max(0, len(edits))
        if not edits:
            return text, st
    st["converted"] = len(edits)
    return out, st


def _wrap(content, op, cl):
    return f"{op}{content}{cl}" if content.strip() else ""


def _apply(text, edits, regions, region_at):
    edits = sorted(edits, key=lambda e: e[0])
    by_region, free = {}, []
    for e in edits:
        r = region_at(e[0], e[1])
        if r is not None and e[5] != "REGION":
            by_region.setdefault(r, []).append(e)
        else:
            free.append(e)
    # tam bolge degistirmeleri (REGION) ve serbest duzenlemeler
    out_edits = []
    for e in free:
        cs, ce, call, res, prefix, boxed = e
        if boxed == "REGION":
            out_edits.append((cs, ce, _tag(call, res)))
        else:
            tail = f"\\boxed{{{boxed}}}" if boxed else ""
            out_edits.append((cs, ce, _tag(call, res, prefix) + tail))
    for r, es in by_region.items():
        a, b, ia, ib, op, cl = regions[r]
        parts, pos = [], ia
        for cs, ce, call, res, prefix, boxed in es:
            parts.append(_wrap(text[pos:cs], op, cl))
            parts.append(_tag(call, res))
            if boxed:
                parts.append(f"{op}\\boxed{{{boxed}}}{cl}")
            pos = ce
        parts.append(_wrap(text[pos:ib], op, cl))
        out_edits.append((a, b, "".join(parts)))
    out_edits.sort(key=lambda e: e[0])
    out, pos = [], 0
    for cs, ce, new in out_edits:
        if cs < pos:
            continue
        out.append(text[pos:cs])
        out.append(new)
        pos = ce
    out.append(text[pos:])
    return "".join(out)


_TAG_RE = re.compile(re.escape(CALL_OPEN) + r"(.+?)" + re.escape(CALL_CLOSE) + re.escape(RES_OPEN) +
                     r"(.*?)" + re.escape(RES_CLOSE), re.S)
_TAG_RE_ANY = re.compile(r"<(?:tool_call|arac_cagrisi)>(.+?)</(?:tool_call|arac_cagrisi)>"
                         r"<(?:tool_result|arac_sonucu)>(.*?)</(?:tool_result|arac_sonucu)>", re.S)
_RES_CANON = {"dogru": "true", "yanlis": "false", "bos": "empty"}


def _canon_res(s):
    return _RES_CANON.get(s, s)


def verify_converted(text: str):
    """Her cagriyi sirayla yeniden calistirir (define ortami korunur); (toplam, hatali).
    Turkce ve Ingilizce etiket/adlarin ikisini de tanir."""
    total = bad = 0
    env = {}
    for m in _TAG_RE_ANY.finditer(text):
        total += 1
        try:
            got = _canon_res(to_text(run(parse_call(m.group(1)), env)))
            if got != _canon_res(m.group(2)):
                bad += 1
        except Exception:
            bad += 1
    return total, bad


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May. In total 48+24 = <<48+24=72>>72 clips. #### 72",
        "She earns $5 * 3 = $15 per day and 15 * 7 = $105 per week.",
        "We have $3 + 4 = 7$ and $\\frac{3}{4} \\times 8 = 6$. Also $\\sin 30^\\circ = \\frac{1}{2}$.",
        "Jungkook collected 6 divided by 3, which equals 2. Then 12 cm = 12 * 10 mm = 120 mm.",
        "So $x=\\frac{70}{8}=\\boxed{8.75}$. Also $1/3 \\approx 0.33$ and $100 \\cdot 2^{10}$ is large.",
        "Then $(x+1)^2 = x^2 + 2x + 1$. Solving $24 + 0.25s + \\frac{2}{3}s = s$ we get $s = \\boxed{288}$.",
        "Let $f(x) = 2x + 1$. Then $f(3) = 7$ and $20\\%$ of 50 is 10; 20% of 50 = 10.",
    ]
    for t in tests:
        o, st = convert_text(t)
        print("IN :", t)
        print("OUT:", o)
        print("ST :", st, "verify:", verify_converted(o))
        print()
