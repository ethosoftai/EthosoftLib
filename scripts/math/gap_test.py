import sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import ethosoftlib.math as em

QUERIES = [
    "4 + 5", "4+5", "4 +5", "4 + 5 = ?", "4 artı 5", "10 - 3", "6 * 7", "6 x 7", "6 çarpı 7", "6 kere 7",
    "20 / 4", "20 bölü 4", "20 ÷ 4", "2^10", "2 üzeri 10", "2**5", "3,5 + 2,5", "1.000 + 500", "1.000,75 * 2",
    "(2 + 3) * 4", "2 + 3 * 4", "-5 + 3", "5!", "sqrt(16)", "√16", "√(2)*√(2)", "kare kök 81",
    "17 mod 5", "17 % 5", "20% of 50", "yüzde 20 50", "50'nin %20'si", "%20 of 150", "0.1 + 0.2",
    "1/3 + 1/6", "3/4 * 8", "2 1/2 + 1 1/2", "sin(90°)", "cos(0)", "log(100)", "ln(1)", "pi * 2",
    "10 km / 2", "5 elma + 3 elma", "3 < 5", "7 = 7", "2 + 2 = 5", "min(3, 7)", "max(3, 7, 9)", "abs(-4)",
    "100 / 3", "1e3 + 1", "0,5 * 4", "kaç eder 4 + 5", "4 + 5 kaç eder", "bana 12 * 12 yi hesapla",
    "x + x", "3x + 2x", "(a+b)^2", "2 + ", "abc", "", "5 / 0", "10 000 + 5", "1 000 000 * 2", "7 5 +",
]
ok = 0
for q in QUERIES:
    try:
        r = em.calculate(q)
        tag = "ok"
    except Exception as e:
        r = f"HATA: {type(e).__name__}: {e}"
        tag = "ERR"
    print(f"{q!r:34} -> {r}")
