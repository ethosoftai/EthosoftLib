"""MathTool v1: fonksiyon kaydi (registry), cagri ayristirici ve calistirici.

Tek dogruluk kaynagi bu dosya. PROTOCOL.md buradan uretilir:
    python toolproto.py > PROTOCOL.md
"""
import ast
import json
import math
import re
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

import sympy

getcontext().prec = 120

# Ic bicim (kanonik, Ingilizce). Disariya varsayilan olarak Turkce yazilir (bkz. localize).
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
RES_OPEN, RES_CLOSE = "<tool_result>", "</tool_result>"

LANG = "tr"
TAGS = {
    "en": ("<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>"),
    "tr": ("<arac_cagrisi>", "</arac_cagrisi>", "<arac_sonucu>", "</arac_sonucu>"),
}
WORDS = {
    "en": {"true": "true", "false": "false", "empty": "empty", "error": "ERROR"},
    "tr": {"true": "dogru", "false": "yanlis", "empty": "bos", "error": "HATA"},
}
# kanonik ad -> Turkce ad (ASCII; ayristirici yalnizca ASCII tanir)
TR_NAMES = {
    "add": "topla", "sub": "cikar", "mul": "carp", "div": "bol", "idiv": "tam_bol", "mod": "mod",
    "pow": "us", "neg": "negatif", "abs": "mutlak", "sqrt": "karekok", "root": "kok",
    "floor": "taban", "ceil": "tavan", "round": "yuvarla", "min": "en_kucuk", "max": "en_buyuk",
    "avg": "ortalama", "calc": "hesapla",
    "pct_of": "yuzde_de", "pct": "yuzde_kac", "pct_change": "yuzde_degisim",
    "fact": "faktoriyel", "comb": "kombinasyon", "perm": "permutasyon", "gcd": "obeb", "lcm": "okek",
    "is_prime": "asal_mi", "factorint": "asal_carpanlar", "n_divisors": "bolen_sayisi",
    "digit_sum": "rakam_toplami", "to_base": "tabana_cevir", "from_base": "tabandan_cevir",
    "exp": "ustel", "ln": "dogal_log", "log": "log", "pi": "pi", "e": "euler",
    "deg": "dereceden_radyana", "rad2deg": "radyandan_dereceye",
    "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot", "sec": "sec", "csc": "csc",
    "asin": "arcsin", "acos": "arccos", "atan": "arctan", "atan2": "arctan2",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "hypot": "hipotenus", "dist": "uzaklik", "area_circle": "daire_alani",
    "circumference": "cember_uzunlugu", "area_triangle": "ucgen_alani", "area_heron": "heron_alani",
    "area_trapezoid": "yamuk_alani", "vol_sphere": "kure_hacmi", "area_sphere": "kure_yuzeyi",
    "vol_cylinder": "silindir_hacmi", "vol_cone": "koni_hacmi",
    "polygon_angle_sum": "cokgen_aci_toplami",
    "median": "medyan", "mode": "tepe_deger", "variance": "varyans", "stdev": "standart_sapma",
    "binom_pmf": "binom_olasilik",
    "arith_nth": "aritmetik_terim", "arith_sum": "aritmetik_toplam", "geom_nth": "geometrik_terim",
    "geom_sum": "geometrik_toplam", "sum_range": "toplam_aralik", "prod_range": "carpim_aralik",
    "simplify": "sadelestir", "expand": "ac", "factor": "carpanlara_ayir", "solve": "coz",
    "solve_system": "denklem_sistemi_coz", "subs": "yerine_koy", "evalf": "ondalik", "equal": "esit_mi",
    "diff": "turev", "integrate": "integral", "limit": "limit",
    "det": "determinant", "inv": "ters_matris", "matmul": "matris_carp", "transpose": "devrik",
    "dot": "ic_carpim", "cross": "vektorel_carpim", "norm": "norm",
    "define": "tanimla",
}
EN_NAMES = {v: k for k, v in TR_NAMES.items() if v != k}


def _w(key):
    return WORDS[LANG][key]


class use_language:
    """with use_language("en"): ...  -> sonuc kelimeleri (dogru/true) o dilde uretilir."""

    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        global LANG
        self.prev, LANG = LANG, self.lang

    def __exit__(self, *a):
        global LANG
        LANG = self.prev


def set_language(lang: str):
    global LANG
    if lang not in TAGS:
        raise ValueError("lang 'tr' veya 'en' olmali")
    LANG = lang


class ToolError(Exception):
    pass


# ---------------------------------------------------------------- sayi bicimi
def fmt_frac(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    d = x.denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    if d == 1:
        s = format(Decimal(x.numerator) / Decimal(x.denominator), "f")
        return s.rstrip("0").rstrip(".") if "." in s else s
    return f"{x.numerator}/{x.denominator}"


def to_text(v) -> str:
    if isinstance(v, Fraction):
        return fmt_frac(v)
    if isinstance(v, bool):
        return _w("true") if v else _w("false")
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(to_text(i) for i in v) + "]"
    if isinstance(v, sympy.Basic):
        if v.is_Rational:
            return fmt_frac(Fraction(int(v.p), int(v.q)))
        return sympy.sstr(v).replace("**", "^")
    return str(v)


def _sy(x):
    if isinstance(x, Fraction):
        return sympy.Rational(x.numerator, x.denominator)
    return x


def _back(e):
    if isinstance(e, sympy.Basic) and e.is_Rational:
        return Fraction(int(e.p), int(e.q))
    return e


def _is_f(*xs):
    return all(isinstance(x, Fraction) for x in xs)


def _int(x, name="arg"):
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x)
    raise ToolError(f"{name}: tam sayi bekleniyor")


# ------------------------------------------------------------------- kayit
REG = {}
CATEGORY_ORDER = ["arithmetic", "percent", "number_theory", "exp_log", "trig",
                  "geometry", "stats_prob", "sequences", "algebra", "calculus",
                  "linear_algebra", "functions"]


def reg(name, cat, sig, doc):
    def deco(fn):
        REG[name] = dict(fn=fn, cat=cat, sig=sig, doc=doc)
        return fn
    return deco


# --- aritmetik
@reg("calc", "arithmetic", "calc(\"ifade\")", "serbest yazilmis ifadeyi hesaplar: calc(\"4 + 5\") -> 9")
def _calc(expr):
    from .calc import evaluate
    return evaluate(str(expr))


@reg("add", "arithmetic", "add(a, b, ...)", "toplama (+)")
def _add(*a):
    if len(a) < 2:
        raise ToolError("add en az 2 arguman")
    if _is_f(*a):
        return sum(a, Fraction(0))
    return _back(sum((_sy(x) for x in a), sympy.Integer(0)))


@reg("sub", "arithmetic", "sub(a, b)", "cikarma (-)")
def _sub(a, b):
    return a - b if _is_f(a, b) else _back(_sy(a) - _sy(b))


@reg("mul", "arithmetic", "mul(a, b, ...)", "carpma (*, x, ×, ·)")
def _mul(*a):
    if len(a) < 2:
        raise ToolError("mul en az 2 arguman")
    if _is_f(*a):
        r = Fraction(1)
        for x in a:
            r *= x
        return r
    r = sympy.Integer(1)
    for x in a:
        r *= _sy(x)
    return _back(r)


@reg("div", "arithmetic", "div(a, b)", "bolme, kesin oran (/, ÷)")
def _div(a, b):
    if b == 0:
        raise ToolError("sifira bolme")
    return a / b if _is_f(a, b) else _back(_sy(a) / _sy(b))


@reg("idiv", "arithmetic", "idiv(a, b)", "tam bolme, tabana yuvarlar (//)")
def _idiv(a, b):
    if b == 0:
        raise ToolError("sifira bolme")
    return Fraction(math.floor(a / b))


@reg("mod", "arithmetic", "mod(a, b)", "mod, kalan (%)")
def _mod(a, b):
    if b == 0:
        raise ToolError("sifira bolme")
    return a - b * math.floor(a / b)


@reg("pow", "arithmetic", "pow(a, b)", "us alma (^)")
def _pow(a, b):
    if _is_f(a, b) and b.denominator == 1:
        if abs(b) > 2000 or (abs(a.numerator) > 10**6 and abs(b) > 200):
            raise ToolError("us cok buyuk")
        if a == 0 and b < 0:
            raise ToolError("sifirin negatif kuvveti")
        return a ** int(b)
    return _back(sympy.Pow(_sy(a), _sy(b)))


@reg("neg", "arithmetic", "neg(a)", "isaret degistirme (-a)")
def _neg(a):
    return -a if _is_f(a) else _back(-_sy(a))


@reg("abs", "arithmetic", "abs(a)", "mutlak deger |a|")
def _abs(a):
    return abs(a) if _is_f(a) else _back(sympy.Abs(_sy(a)))


@reg("sqrt", "arithmetic", "sqrt(a)", "karekok (√)")
def _sqrt(a):
    if _is_f(a):
        if a < 0:
            raise ToolError("negatif sayinin karekoku")
        n, d = a.numerator, a.denominator
        rn, rd = math.isqrt(n), math.isqrt(d)
        if rn * rn == n and rd * rd == d:
            return Fraction(rn, rd)
    return _back(sympy.sqrt(_sy(a)))


@reg("root", "arithmetic", "root(n, a)", "n. dereceden kok")
def _root(n, a):
    return _back(sympy.root(_sy(a), _sy(n)))


def _round_half_up(x: Fraction, nd: int = 0) -> Fraction:
    q = x * (Fraction(10) ** nd)
    r = math.floor(q + Fraction(1, 2)) if q >= 0 else -math.floor(-q + Fraction(1, 2))
    return Fraction(r) / (Fraction(10) ** nd)


@reg("floor", "arithmetic", "floor(a)", "tabana yuvarlama")
def _floor(a):
    return Fraction(math.floor(a))


@reg("ceil", "arithmetic", "ceil(a)", "tavana yuvarlama")
def _ceil(a):
    return Fraction(math.ceil(a))


@reg("round", "arithmetic", "round(a, nd=0)", "nd basamaga yuvarla (yarim yukari)")
def _round(a, nd=Fraction(0)):
    return _round_half_up(a, int(nd))


@reg("min", "arithmetic", "min(a, b, ...)", "en kucuk")
def _min(*a):
    return min(a)


@reg("max", "arithmetic", "max(a, b, ...)", "en buyuk")
def _max(*a):
    return max(a)


@reg("avg", "arithmetic", "avg(a, b, ...)", "aritmetik ortalama")
def _avg(*a):
    return sum(a, Fraction(0)) / len(a)


# --- yuzde
@reg("pct_of", "percent", "pct_of(p, x)", "x'in yuzde p'si = p*x/100")
def _pct_of(p, x):
    return p * x / 100


@reg("pct", "percent", "pct(a, b)", "a, b'nin yuzde kaci = a/b*100")
def _pct(a, b):
    if b == 0:
        raise ToolError("sifira bolme")
    return a / b * 100


@reg("pct_change", "percent", "pct_change(old, new)", "yuzde degisim = (new-old)/old*100")
def _pct_change(old, new):
    if old == 0:
        raise ToolError("sifira bolme")
    return (new - old) / old * 100


# --- sayi teorisi
@reg("fact", "number_theory", "fact(n)", "faktoriyel (n!), n <= 500")
def _fact(n):
    n = _int(n)
    if not 0 <= n <= 500:
        raise ToolError("fact aralik disi")
    return Fraction(math.factorial(n))


@reg("comb", "number_theory", "comb(n, k)", "kombinasyon C(n,k)")
def _comb(n, k):
    n, k = _int(n), _int(k)
    if n < 0 or k < 0 or n > 5000:
        raise ToolError("comb aralik disi")
    return Fraction(math.comb(n, k))


@reg("perm", "number_theory", "perm(n, k)", "permutasyon P(n,k)")
def _perm(n, k):
    n, k = _int(n), _int(k)
    if n < 0 or k < 0 or n > 5000:
        raise ToolError("perm aralik disi")
    return Fraction(math.perm(n, k))


@reg("gcd", "number_theory", "gcd(a, b, ...)", "OBEB")
def _gcd(*a):
    return Fraction(math.gcd(*[_int(x) for x in a]))


@reg("lcm", "number_theory", "lcm(a, b, ...)", "OKEK")
def _lcm(*a):
    return Fraction(math.lcm(*[_int(x) for x in a]))


@reg("is_prime", "number_theory", "is_prime(n)", "asal mi? true/false")
def _is_prime(n):
    return bool(sympy.isprime(_int(n)))


@reg("factorint", "number_theory", "factorint(n)", "asal carpanlara ayirma, orn. 2^3 * 3")
def _factorint(n):
    f = sympy.factorint(_int(n))
    return " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items()))


@reg("n_divisors", "number_theory", "n_divisors(n)", "bolen sayisi")
def _n_div(n):
    return Fraction(int(sympy.divisor_count(_int(n))))


@reg("digit_sum", "number_theory", "digit_sum(n)", "rakamlar toplami")
def _digit_sum(n):
    return Fraction(sum(int(c) for c in str(abs(_int(n)))))


@reg("to_base", "number_theory", "to_base(n, b)", "n'yi b tabaninda yaz (2..36)")
def _to_base(n, b):
    n, b = _int(n), _int(b)
    if not 2 <= b <= 36:
        raise ToolError("taban 2..36")
    digs = "0123456789abcdefghijklmnopqrstuvwxyz"
    s, m = "", abs(n)
    while True:
        s = digs[m % b] + s
        m //= b
        if m == 0:
            break
    return ("-" if n < 0 else "") + s


@reg("from_base", "number_theory", "from_base(\"s\", b)", "b tabanindaki s'yi onluga cevir")
def _from_base(s, b):
    return Fraction(int(str(s), _int(b)))


# --- us / log / sabitler
@reg("exp", "exp_log", "exp(x)", "e^x")
def _exp(x):
    return _back(sympy.exp(_sy(x)))


@reg("ln", "exp_log", "ln(x)", "dogal logaritma")
def _ln(x):
    if x <= 0:
        raise ToolError("ln alan disi")
    return _back(sympy.log(_sy(x)))


@reg("log", "exp_log", "log(x, base=10)", "logaritma")
def _log(x, base=Fraction(10)):
    if x <= 0 or base <= 0 or base == 1:
        raise ToolError("log alan disi")
    return _back(sympy.log(_sy(x), _sy(base)))


@reg("pi", "exp_log", "pi()", "π sabiti")
def _pi():
    return sympy.pi


@reg("e", "exp_log", "e()", "Euler sabiti")
def _e():
    return sympy.E


# --- trigonometri (radyan; derece icin deg())
@reg("deg", "trig", "deg(x)", "dereceyi radyana cevir: x*pi/180")
def _deg(x):
    return _back(_sy(x) * sympy.pi / 180)


@reg("rad2deg", "trig", "rad2deg(x)", "radyani dereceye cevir")
def _rad2deg(x):
    return _back(sympy.simplify(_sy(x) * 180 / sympy.pi))


def _trig(name, fn, doc):
    @reg(name, "trig", f"{name}(x)", doc)
    def f(x):
        return _back(sympy.simplify(fn(_sy(x))))
    return f


for _n, _f, _d in [("sin", sympy.sin, "sinus (radyan)"), ("cos", sympy.cos, "kosinus (radyan)"),
                   ("tan", sympy.tan, "tanjant (radyan)"), ("cot", sympy.cot, "kotanjant"),
                   ("sec", sympy.sec, "sekant"), ("csc", sympy.csc, "kosekant"),
                   ("asin", sympy.asin, "arcsin"), ("acos", sympy.acos, "arccos"),
                   ("atan", sympy.atan, "arctan"), ("sinh", sympy.sinh, "sinh"),
                   ("cosh", sympy.cosh, "cosh"), ("tanh", sympy.tanh, "tanh")]:
    _trig(_n, _f, _d)


@reg("atan2", "trig", "atan2(y, x)", "iki arguman arctan")
def _atan2(y, x):
    return _back(sympy.atan2(_sy(y), _sy(x)))


# --- geometri
@reg("hypot", "geometry", "hypot(a, b)", "hipotenus = sqrt(a^2+b^2)")
def _hypot(a, b):
    return _sqrt(a * a + b * b)


@reg("dist", "geometry", "dist(x1, y1, x2, y2)", "iki nokta arasi uzaklik")
def _dist(x1, y1, x2, y2):
    return _sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


@reg("area_circle", "geometry", "area_circle(r)", "daire alani pi*r^2")
def _area_circle(r):
    return _back(sympy.pi * _sy(r) ** 2)


@reg("circumference", "geometry", "circumference(r)", "cember uzunlugu 2*pi*r")
def _circumference(r):
    return _back(2 * sympy.pi * _sy(r))


@reg("area_triangle", "geometry", "area_triangle(b, h)", "ucgen alani b*h/2")
def _area_tri(b, h):
    return b * h / 2


@reg("area_heron", "geometry", "area_heron(a, b, c)", "Heron formulu")
def _area_heron(a, b, c):
    s = (a + b + c) / 2
    return _sqrt(s * (s - a) * (s - b) * (s - c))


@reg("area_trapezoid", "geometry", "area_trapezoid(a, b, h)", "yamuk alani (a+b)*h/2")
def _area_trap(a, b, h):
    return (a + b) * h / 2


@reg("vol_sphere", "geometry", "vol_sphere(r)", "kure hacmi 4/3*pi*r^3")
def _vol_sphere(r):
    return _back(sympy.Rational(4, 3) * sympy.pi * _sy(r) ** 3)


@reg("area_sphere", "geometry", "area_sphere(r)", "kure yuzeyi 4*pi*r^2")
def _area_sphere(r):
    return _back(4 * sympy.pi * _sy(r) ** 2)


@reg("vol_cylinder", "geometry", "vol_cylinder(r, h)", "silindir hacmi pi*r^2*h")
def _vol_cyl(r, h):
    return _back(sympy.pi * _sy(r) ** 2 * _sy(h))


@reg("vol_cone", "geometry", "vol_cone(r, h)", "koni hacmi pi*r^2*h/3")
def _vol_cone(r, h):
    return _back(sympy.pi * _sy(r) ** 2 * _sy(h) / 3)


@reg("polygon_angle_sum", "geometry", "polygon_angle_sum(n)", "n-genin ic acilar toplami (n-2)*180")
def _poly_angle(n):
    return (n - 2) * 180


# --- istatistik / olasilik
@reg("median", "stats_prob", "median(a, b, ...)", "medyan")
def _median(*a):
    s = sorted(a)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


@reg("mode", "stats_prob", "mode(a, b, ...)", "tepe deger (esitlikte en kucuk)")
def _mode(*a):
    best = max(a.count(x) for x in a)
    return min(x for x in a if a.count(x) == best)


@reg("variance", "stats_prob", "variance(a, b, ...)", "populasyon varyansi")
def _variance(*a):
    m = sum(a, Fraction(0)) / len(a)
    return sum(((x - m) ** 2 for x in a), Fraction(0)) / len(a)


@reg("stdev", "stats_prob", "stdev(a, b, ...)", "populasyon standart sapmasi")
def _stdev(*a):
    return _sqrt(_variance(*a))


@reg("binom_pmf", "stats_prob", "binom_pmf(n, k, p)", "binom olasilik C(n,k) p^k (1-p)^(n-k)")
def _binom(n, k, p):
    n, k = _int(n), _int(k)
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


# --- diziler
@reg("arith_nth", "sequences", "arith_nth(a1, d, n)", "aritmetik dizi n. terim")
def _arith_nth(a1, d, n):
    return a1 + (n - 1) * d


@reg("arith_sum", "sequences", "arith_sum(a1, d, n)", "aritmetik dizi ilk n terim toplami")
def _arith_sum(a1, d, n):
    return n * (2 * a1 + (n - 1) * d) / 2


@reg("geom_nth", "sequences", "geom_nth(a1, r, n)", "geometrik dizi n. terim")
def _geom_nth(a1, r, n):
    return a1 * _pow(r, n - 1)


@reg("sum_range", "sequences", "sum_range(\"expr\", \"i\", a, b)", "toplam sum_{i=a}^{b} expr (b-a <= 100000)")
def _sum_range(e, var, a, b):
    a, b = _int(a), _int(b)
    if b - a > 100000:
        raise ToolError("aralik cok buyuk")
    return _back(sympy.summation(_sym(e), (sympy.Symbol(str(var)), a, b)))


@reg("prod_range", "sequences", "prod_range(\"expr\", \"i\", a, b)", "carpim prod_{i=a}^{b} expr (b-a <= 2000)")
def _prod_range(e, var, a, b):
    a, b = _int(a), _int(b)
    if b - a > 2000:
        raise ToolError("aralik cok buyuk")
    return _back(sympy.product(_sym(e), (sympy.Symbol(str(var)), a, b)))


@reg("geom_sum", "sequences", "geom_sum(a1, r, n)", "geometrik dizi ilk n terim toplami")
def _geom_sum(a1, r, n):
    if r == 1:
        return a1 * n
    return a1 * (1 - _pow(r, n)) / (1 - r)


# --- sembolik (string argumanli)
_SYM_BAD = re.compile(r"__|[A-Za-z_]\s*\.|\[|\]|lambda|import|exec|eval|open\(|;|\\")
_SYM_NAMES = {n: getattr(sympy, n) for n in [
    "sqrt", "sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "sinh", "cosh",
    "tanh", "exp", "log", "ln", "Abs", "pi", "E", "I", "oo", "factorial", "binomial", "Rational",
    "Integer", "Float", "Symbol", "floor", "ceiling", "gcd", "lcm", "Eq", "Function"]}
_SYM_NAMES["e"] = sympy.E
_SYM_NAMES["comb"] = sympy.binomial
_SYM_NAMES["abs"] = sympy.Abs
_SYM_NAMES["ln"] = sympy.log


def _sym(s):
    s = str(s)
    if _SYM_BAD.search(s) or len(s) > 2000:
        raise ToolError("gecersiz sembolik ifade")
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            implicit_multiplication_application, convert_xor)
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        return parse_expr(s, local_dict=dict(_SYM_NAMES), global_dict={"__builtins__": {}},
                          transformations=tr, evaluate=True)
    except Exception as ex:
        raise ToolError(f"ayristirilamadi: {ex}")


def _sym_eq(s):
    s = str(s)
    if "=" in s:
        l, r = s.split("=", 1)
        return sympy.Eq(_sym(l), _sym(r))
    return _sym(s)


@reg("simplify", "algebra", "simplify(\"expr\")", "sadelestir")
def _simplify(e):
    return sympy.simplify(_sym(e))


@reg("expand", "algebra", "expand(\"expr\")", "parantez ac")
def _expand(e):
    return sympy.expand(_sym(e))


@reg("factor", "algebra", "factor(\"expr\")", "carpanlara ayir")
def _factor(e):
    return sympy.factor(_sym(e))


def _set_text(s):
    if s == sympy.S.EmptySet:
        return _w("empty")
    if isinstance(s, sympy.Union):
        return " U ".join(_set_text(a) for a in sorted(
            s.args, key=lambda x: float(x.start) if getattr(x, "start", None) is not None
            and x.start.is_finite else -1e18))
    if isinstance(s, sympy.Interval):
        l = "(" if s.left_open else "["
        r = ")" if s.right_open else "]"
        return f"{l}{to_text(s.start)}, {to_text(s.end)}{r}"
    if isinstance(s, sympy.FiniteSet):
        return "{" + ", ".join(to_text(x) for x in sorted(s.args, key=float)) + "}"
    return str(s)


@reg("solve", "algebra", "solve(\"eq\", \"x\")", "denklem veya esitsizlik coz (esitlik yoksa =0 varsayilir)")
def _solve(eq, var):
    v = sympy.Symbol(str(var))
    if re.search(r"[<>]", str(eq)):
        rel = _sym(str(eq))
        return _set_text(sympy.solve_univariate_inequality(rel, v, relational=False))
    sols = sympy.solve(_sym_eq(eq), v)
    return sorted(sols, key=lambda z: (not z.is_real, float(sympy.re(z)) if z.is_number else 0)) \
        if all(getattr(z, "is_number", False) for z in sols) else sols


@reg("solve_system", "algebra", "solve_system(\"eq1; eq2\", \"x, y\")", "denklem sistemi coz")
def _solve_system(eqs, vars_):
    eq_list = [_sym_eq(e) for e in str(eqs).split(";")]
    vs = [sympy.Symbol(v.strip()) for v in str(vars_).split(",")]
    r = sympy.solve(eq_list, vs, dict=True)
    return "; ".join(", ".join(f"{k}={to_text(v)}" for k, v in d.items()) for d in r)


@reg("equal", "algebra", "equal(a, b)", "iki ifade ozdes/esit mi? true/false (sayi veya \"ifade\")")
def _equal_tool(a, b):
    def conv(x):
        return _sy(x) if isinstance(x, Fraction) else _sym(x)
    d = conv(a) - conv(b)
    if d == 0:
        return True
    if not d.free_symbols:
        return bool(abs(sympy.N(d, 40)) < sympy.Float("1e-30"))
    d = sympy.expand(d)
    return bool(d == 0 or sympy.cancel(d) == 0)


@reg("subs", "algebra", "subs(\"expr\", \"x\", value)", "degisken yerine deger koy")
def _subs(e, var, val):
    return _back(_sym(e).subs(sympy.Symbol(str(var)), _sy(val) if not isinstance(val, str) else _sym(val)))


@reg("evalf", "algebra", "evalf(\"expr\", digits=10)", "ondalik yaklasik deger")
def _evalf(e, digits=Fraction(10)):
    v = _sym(e) if isinstance(e, str) else _sy(e)
    return str(sympy.N(v, int(digits)))


@reg("diff", "calculus", "diff(\"expr\", \"x\")", "turev")
def _diff(e, var):
    return sympy.diff(_sym(e), sympy.Symbol(str(var)))


@reg("integrate", "calculus", "integrate(\"expr\", \"x\", a?, b?)", "integral (belirsiz veya belirli)")
def _integrate(e, var, a=None, b=None):
    v = sympy.Symbol(str(var))
    if a is None:
        return sympy.integrate(_sym(e), v)
    return _back(sympy.integrate(_sym(e), (v, _sy(a), _sy(b))))


@reg("limit", "calculus", "limit(\"expr\", \"x\", point)", "limit (point: sayi veya \"oo\")")
def _limit(e, var, point):
    p = _sym(point) if isinstance(point, str) else _sy(point)
    return _back(sympy.limit(_sym(e), sympy.Symbol(str(var)), p))


# --- lineer cebir (matris/vektor: "[[1,2],[3,4]]")
def _mat(s):
    try:
        return sympy.Matrix(ast.literal_eval(str(s)))
    except Exception:
        raise ToolError("matris ayristirilamadi")


def _mat_text(m):
    return str(m.tolist()).replace(" ", "")


@reg("det", "linear_algebra", "det(\"[[1,2],[3,4]]\")", "determinant")
def _det(m):
    return _back(_mat(m).det())


@reg("inv", "linear_algebra", "inv(\"[[1,2],[3,4]]\")", "ters matris")
def _inv(m):
    return _mat_text(_mat(m).inv())


@reg("matmul", "linear_algebra", "matmul(A, B)", "matris carpimi")
def _matmul(a, b):
    return _mat_text(_mat(a) * _mat(b))


@reg("transpose", "linear_algebra", "transpose(A)", "devrik matris")
def _transpose(m):
    return _mat_text(_mat(m).T)


@reg("dot", "linear_algebra", "dot(u, v)", "ic carpim")
def _dot(u, v):
    return _back(_mat(u).dot(_mat(v)))


@reg("cross", "linear_algebra", "cross(u, v)", "vektorel carpim (3B)")
def _cross(u, v):
    return _mat_text(sympy.Matrix(_mat(u)).cross(sympy.Matrix(_mat(v))))


@reg("norm", "linear_algebra", "norm(u)", "vektor uzunlugu")
def _norm(u):
    return _back(_mat(u).norm())


# --- fonksiyon tanimlama
@reg("define", "functions", "define(\"f\", \"x, y\", \"x^2 + y\")",
     "f(x,y) fonksiyonu tanimla; sonra f(2, 3) diye cagrilir")
def _define(name, vars_, body):
    return f"{name}({vars_}) = {body}"  # kayit run() icinde


# ---------------------------------------------------------- cagri ayristirma
_TOK = re.compile(r'\s*(?:(?P<num>-?\d+(?:\.\d+)?)|(?P<id>[A-Za-z_][A-Za-z_0-9]*)|'
                  r'(?P<str>"(?:[^"\\]|\\.)*")|(?P<p>[(),]))')


def _tokenize(s):
    pos, out = 0, []
    s = s.strip()
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m:
            raise ToolError(f"gecersiz karakter: {s[pos:pos+10]!r}")
        pos = m.end()
        for k in ("num", "id", "str", "p"):
            if m.group(k) is not None:
                out.append((k, m.group(k)))
                break
    return out


def parse_call(text: str):
    toks = _tokenize(text)
    pos = 0

    def parse_val():
        nonlocal pos
        if pos >= len(toks):
            raise ToolError("beklenmeyen son")
        k, v = toks[pos]
        if k == "num":
            pos += 1
            return Fraction(Decimal(v))
        if k == "str":
            pos += 1
            return ast.literal_eval(v)
        if k == "id":
            pos += 1
            if pos < len(toks) and toks[pos] == ("p", "("):
                pos += 1
                args = []
                if toks[pos] == ("p", ")"):
                    pos += 1
                    return ("call", v, args)
                while True:
                    args.append(parse_val())
                    if toks[pos] == ("p", ","):
                        pos += 1
                        continue
                    if toks[pos] == ("p", ")"):
                        pos += 1
                        break
                    raise ToolError("',' veya ')' bekleniyordu")
                return ("call", v, args)
            return v  # cplak isim -> string
        raise ToolError(f"beklenmeyen: {v}")

    node = parse_val()
    if pos != len(toks):
        raise ToolError("fazla token")
    return node


def run(node, env=None):
    env = env if env is not None else {}
    if isinstance(node, (Fraction, str)):
        return node
    _, name, args = node
    if name not in env:
        name = EN_NAMES.get(name, name)
    if name in env:
        vars_, body = env[name]
        vals = [run(a, env) for a in args]
        if len(vals) != len(vars_):
            raise ToolError(f"{name}: {len(vars_)} arguman bekleniyor")
        return _back(body.subs({sympy.Symbol(v): _sy(x) for v, x in zip(vars_, vals)}))
    if name not in REG:
        raise ToolError(f"bilinmeyen fonksiyon: {name}")
    if name == "define":
        n, vs, body = [run(a, env) for a in args]
        env[str(n)] = ([v.strip() for v in str(vs).split(",")], _sym(body))
        return f"{n}({vs}) = {body}"
    vals = [run(a, env) for a in args]
    try:
        return REG[name]["fn"](*vals)
    except ToolError:
        raise
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, ArithmeticError) as ex:
        raise ToolError(f"{name}: {ex}")


def execute(call_text: str, env=None) -> str:
    return to_text(run(parse_call(call_text), env))


# --------------------------------------------------------------- yerellestirme
def render_node(n) -> str:
    if isinstance(n, Fraction):
        return fmt_frac(n)
    if isinstance(n, str):
        return json.dumps(n, ensure_ascii=False)
    _, name, args = n
    return f"{name}({', '.join(render_node(a) for a in args)})"


def _rename(n):
    if isinstance(n, tuple):
        return ("call", TR_NAMES.get(n[1], n[1]), [_rename(a) for a in n[2]])
    return n


_EN_TAG = re.compile(r"<tool_call>(.+?)</tool_call><tool_result>(.*?)</tool_result>", re.S)
_TR_TAG = re.compile(r"<arac_cagrisi>(.+?)</arac_cagrisi><arac_sonucu>(.*?)</arac_sonucu>", re.S)


def localize(text: str, lang: str = "tr") -> str:
    """Kanonik (Ingilizce) etiket/ad bicimini istenen dile cevirir. lang='en' dokunmaz."""
    if lang != "tr":
        return text
    co, cc, ro, rc = TAGS["tr"]

    def sub(m):
        try:
            call = render_node(_rename(parse_call(m.group(1))))
        except ToolError:
            return m.group(0)
        return f"{co}{call}{cc}{ro}{m.group(2)}{rc}"

    return _EN_TAG.sub(sub, text)


def delocalize(text: str) -> str:
    """Turkce etiket/adlari kanonik Ingilizce bicime geri cevirir."""
    def sub(m):
        try:
            node = parse_call(m.group(1))
        except ToolError:
            return m.group(0)

        def ren(n):
            if isinstance(n, tuple):
                return ("call", EN_NAMES.get(n[1], n[1]), [ren(a) for a in n[2]])
            return n
        return f"{CALL_OPEN}{render_node(ren(node))}{CALL_CLOSE}{RES_OPEN}{m.group(2)}{RES_CLOSE}"

    return _TR_TAG.sub(sub, text)


# ------------------------------------------------------------ dokuman uretimi
_HEADER = r"""# MathTool v1 - Matematik Tool-Use Protokolu

Amac: Matematik cozumlerindeki **her hesabi** bir arac cagrisina cevirmek. Model sayiyi
"kafadan" yazmaz; hesabi araca yaptirir, sonucu aractan alir.

## Bicim

```
<arac_cagrisi>FONKSIYON(arg1, arg2, ...)</arac_cagrisi><arac_sonucu>SONUC</arac_sonucu>
```

- `<arac_cagrisi>` blogunu model uretir, `<arac_sonucu>` blogunu **calistirici** yazar (egitimde loss maskelenir).
- Fonksiyon adlari Turkcedir (ASCII): `topla`, `cikar`, `carp`, `bol`, `us`, `karekok` ...
  Ingilizce adlar da calisir (`add`, `sub`, `mul` ...); `lang="en"` ile Ingilizce etiket/ad uretilir.
- Her cagri tek bir fonksiyon agaci: ic ice cagri serbest. `(3+4)*2` -> `carp(topla(3, 4), 2)`.
- Argumanlar: sayi (`3`, `-2`, `0.5`), ic cagri, ya da `"..."` sembolik ifade.
- Sayilar her zaman **nokta ondalikli**, binlik ayirac yok (`1000`, `0.25`).
- Kesin sonuc: tam sayi `7`, tam ondalik `0.25`, aksi halde kesir `1/3`. Irrasyonel sonuc kesin yazilir: `sqrt(2)`, `pi`.
- Yaklasik deger icin `ondalik("ifade", 6)` kullanilir (sonuc metin).
- Mantiksal sonuclar: `dogru` / `yanlis`; bos kume: `bos`; hata: `HATA: neden`.
- Serbest yazilmis ifade icin `hesapla("4 + 5")` -> `9` (Turkce sozcukler de olur: `"4 arti 5"`).

## Operator -> fonksiyon tablosu

| Yazi | Fonksiyon | Yazi | Fonksiyon |
|---|---|---|---|
| `a + b` | `topla(a, b)` | `a ^ b`, `a**b` | `us(a, b)` |
| `a - b` | `cikar(a, b)` | `-a` | `negatif(a)` |
| `a * b`, `a x b`, `a × b`, `a · b` | `carp(a, b)` | `n!` | `faktoriyel(n)` |
| `a / b`, `a ÷ b`, `\frac{a}{b}` | `bol(a, b)` | `√a`, `\sqrt{a}` | `karekok(a)` |
| `a % b` (kalan) | `mod(a, b)` | `p% of x` | `yuzde_de(p, x)` |
| `a // b` | `tam_bol(a, b)` | `\binom{n}{k}`, `C(n,k)` | `kombinasyon(n, k)` |

Toplama/carpma ayni seviyede ardisiksa tek cagri: `1+2+3` -> `topla(1, 2, 3)`.
Aciyi dereceyle verirsen `dereceden_radyana()` ile radyana cevir: `sin(30°)` -> `sin(dereceden_radyana(30))` -> `0.5`.

## Fonksiyon tanimlama ve kullanma

```
<arac_cagrisi>tanimla("f", "x", "x^2 + 1")</arac_cagrisi><arac_sonucu>f(x) = x^2 + 1</arac_sonucu>
<arac_cagrisi>f(3)</arac_cagrisi><arac_sonucu>10</arac_sonucu>
```

Tanim, ayni cozum (konusma) icinde gecerlidir.

## Ornekler

| Metin | Tool-use hali |
|---|---|
| `3 + 4 = 7` | `<arac_cagrisi>topla(3, 4)</arac_cagrisi><arac_sonucu>7</arac_sonucu>` |
| `48 / 2 = 24` | `<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>` |
| `sin 30° = 1/2` | `<arac_cagrisi>sin(dereceden_radyana(30))</arac_cagrisi><arac_sonucu>0.5</arac_sonucu>` |
| `C(10, 3) = 120` | `<arac_cagrisi>kombinasyon(10, 3)</arac_cagrisi><arac_sonucu>120</arac_sonucu>` |
| `x^2 - 4 = 0` | `<arac_cagrisi>coz("x^2 - 4 = 0", "x")</arac_cagrisi><arac_sonucu>[-2, 2]</arac_sonucu>` |
| `d/dx x^3` | `<arac_cagrisi>turev("x^3", "x")</arac_cagrisi><arac_sonucu>3*x^2</arac_sonucu>` |
| `4 + 5` (serbest) | `<arac_cagrisi>hesapla("4 + 5")</arac_cagrisi><arac_sonucu>9</arac_sonucu>` |

## Fonksiyon listesi
"""

_CAT_TR = {"arithmetic": "Aritmetik", "percent": "Yuzde", "number_theory": "Sayi teorisi / sayma",
           "exp_log": "Us, logaritma, sabitler", "trig": "Trigonometri", "geometry": "Geometri",
           "stats_prob": "Istatistik / olasilik", "sequences": "Diziler", "algebra": "Cebir (sembolik)",
           "calculus": "Analiz", "linear_algebra": "Lineer cebir", "functions": "Fonksiyon tanimlama"}


def render_protocol_md() -> str:
    out = [_HEADER]
    for cat in CATEGORY_ORDER:
        items = [(n, r) for n, r in REG.items() if r["cat"] == cat]
        if not items:
            continue
        out.append(f"\n### {_CAT_TR[cat]}\n\n| Cagri | Ingilizce ad | Aciklama |\n|---|---|---|")
        for n, r in items:
            tr = TR_NAMES.get(n, n)
            sig = r["sig"].replace(n, tr, 1) if r["sig"].startswith(n) else r["sig"]
            out.append(f"| `{sig}` | `{n}` | {r['doc']} |")
    out.append(f"\nToplam: **{len(REG)} fonksiyon**.\n")
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(render_protocol_md())
