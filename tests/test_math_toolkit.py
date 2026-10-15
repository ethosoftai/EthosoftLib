import pytest
pytest.importorskip("sympy")
import ethosoftlib.math as em


def test_execute_turkish_names():
    assert em.execute("topla(3, 4)") == "7"
    assert em.execute("bol(1, 3)") == "1/3"
    assert em.execute("bol(1, 4)") == "0.25"
    assert em.execute("carp(topla(3, 4), 2)") == "14"
    assert em.execute("faktoriyel(5)") == "120"
    assert em.execute("kombinasyon(10, 3)") == "120"


def test_english_aliases_still_work():
    assert em.execute("add(3, 4)") == "7"
    assert em.execute("mul(add(3, 4), 2)") == "14"
    assert em.execute("fact(5)") == "120"


def test_trig_and_symbolic():
    assert em.execute("sin(dereceden_radyana(30))") == "0.5"
    assert em.execute('coz("x^2 - 4 = 0", "x")') == "[-2, 2]"
    assert em.execute('ac("(x+1)^2")') == "x^2 + 2*x + 1"
    assert em.execute('coz("x^2 - 5*x + 6 > 0", "x")') == "(-oo, 2) U (3, oo)"
    assert em.execute('esit_mi("(x+1)^2", "x^2+2x+1")') == "dogru"


def test_convert_simple_equation_turkish_default():
    out = em.convert("Toplam 3 + 4 = 7 elma.")
    assert out == "Toplam <arac_cagrisi>topla(3, 4)</arac_cagrisi><arac_sonucu>7</arac_sonucu> elma."


def test_convert_english_mode():
    out = em.convert("Toplam 3 + 4 = 7 elma.", lang="en")
    assert out == "Toplam <tool_call>add(3, 4)</tool_call><tool_result>7</tool_result> elma."
    assert em.delocalize(em.convert("2 + 2 = 4")) == "<tool_call>add(2, 2)</tool_call><tool_result>4</tool_result>"


def test_wrong_equation_is_left_alone():
    text = "Wrong: 2 + 2 = 5."
    assert em.convert(text) == text


def test_gsm8k_annotation():
    text = "Natalia sold 48/2 = <<48/2=24>>24 clips in May."
    out = em.convert(text)
    assert "<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>" in out
    assert "<<" not in out


def test_latex_boxed_and_rounding():
    out = em.convert("So $x=\\frac{70}{8}=\\boxed{8.75}$ and $80/3 = 26.67$.")
    assert "bol(70, 8)" in out and "\\boxed{8.75}" in out
    assert "yuvarla(bol(80, 3), 2)" in out


def test_every_converted_call_verifies_both_languages():
    text = ("We have $\\binom{5}{2} = 10$, $5! = 120$, $\\sqrt{50} = 5\\sqrt{2}$, "
            "$\\sum_{k=1}^{4} k^2 = 30$ and $1 + 2 + \\cdots + 100 = 5050$.")
    for lang in ("tr", "en"):
        total, bad = em.verify(em.convert(text, lang=lang))
        assert total >= 4 and bad == 0


def test_code_is_not_touched():
    code = "x = 3 + 4\nprint(x)"
    assert em.convert(code) == code


def test_calculator():
    assert em.calculate("4 + 5") == "9"
    assert em.calculate("4+5=") == "9"
    assert em.calculate("(3 + 4) x 2 ?") == "14"
    assert em.calculate("2 uzeri 10") == "1024"
    assert em.calculate("12 bolu 4") == "3"
    assert em.calculate("1.250,50 + 3,5") == "1254"
    assert em.calculate("kare kok 144") == "12"
    assert em.calculate("1/3 + 1/6") == "0.5"
    assert em.calculate("(x+1)^2 - x^2") == "2*x + 1"


def test_calculator_trace_and_tool():
    assert em.calculate("4 + 5", trace=True) == "<arac_cagrisi>topla(4, 5)</arac_cagrisi><arac_sonucu>9</arac_sonucu>"
    assert em.execute('hesapla("7 * 6")') == "42"
    assert em.execute('calc("2 uzeri 5")') == "32"


def test_protocol_doc_is_turkish_and_lists_functions():
    md = em.render_protocol_md()
    assert "topla(a, b, ...)" in md and "coz(" in md and "hesapla(" in md
    assert "<arac_cagrisi>" in md
    assert len(em.functions()) > 80


def test_calculator_extended():
    assert em.calculate("17 mod 5") == "2"
    assert em.calculate("17 % 5") == "2"
    assert em.calculate("50'nin %20'si") == "10"
    assert em.calculate("%20 of 150") == "30"
    assert em.calculate("10 000 + 5") == "10005"
    assert em.calculate("1.000.000 + 1") == "1000001"
    assert em.calculate("1.000 + 500", locale="tr") == "1500"
    assert em.calculate("1e3 + 1") == "1001"
    assert em.calculate("3 < 5") == "dogru"
    assert em.calculate("2 + 2 = 5") == "yanlis"
    assert em.calculate("1 < 2 < 3") == "dogru"
    assert em.calculate("max(3, 7, 9)") == "9"
    assert em.calculate("kombinasyon(10, 3)") == "120"
    assert em.calculate("bana 12 * 12 yi hesapla") == "144"
    assert em.calculate("kaç eder 4 + 5") == "9"
    assert em.calculate("100 / 3", decimals=2) == "33.33"
    assert em.calculate("5 / 0", on_error="text").startswith("HATA")


def test_calculator_symbolic_implicit_variables():
    assert em.calculate("3x + 2x") == "5*x"
    assert em.calculate("2ab + 3ab") == "5*a*b"
    assert em.calculate("(a+b)^2 = a^2+2ab+b^2") == "dogru"
    assert em.calculate("10 km / 2") == "5"


def test_fill_results_basic():
    assert em.fill_results("<arac_cagrisi>bol(48, 2)</arac_cagrisi>") == \
        "<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>"


def test_fill_results_partial_wrong_and_text():
    assert em.fill_results("<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>") == \
        "<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>"
    assert em.fill_results("<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>999</arac_sonucu> ok") == \
        "<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu> ok"
    assert em.fill_results("x <arac_cagrisi>topla(1, 2)</arac_cagrisi> y").startswith("x <arac_cagrisi>")


def test_fill_results_define_error_and_english():
    out = em.fill_results('<arac_cagrisi>tanimla("f", "x", "x^2 + 1")</arac_cagrisi> '
                          '<arac_cagrisi>f(3)</arac_cagrisi>')
    assert "<arac_sonucu>10</arac_sonucu>" in out
    assert "HATA: sifira bolme" in em.fill_results("<arac_cagrisi>bol(1, 0)</arac_cagrisi>")
    assert em.fill_results("<tool_call>add(3, 4)</tool_call>") == "<tool_call>add(3, 4)</tool_call><tool_result>7</tool_result>"


def test_result_block():
    assert em.result_block("bol(48, 2)") == "<arac_sonucu>24</arac_sonucu>"
    assert em.result_block("on <arac_cagrisi>carp(topla(3, 4), 2)</arac_cagrisi>") == "<arac_sonucu>14</arac_sonucu>"


def test_root_import_does_not_load_math_or_sympy():
    import subprocess
    import sys
    code = (
        "import sys, ethosoftlib; "
        "assert 'ethosoftlib.math' not in sys.modules; "
        "assert 'sympy' not in sys.modules; "
        "assert ethosoftlib.providers.describe('tooluse', 'math').name == 'math'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_registry_creates_math_toolkit():
    from ethosoftlib import providers
    kit = providers.create("tooluse", "math", lang="tr")
    assert kit.fill_results("<arac_cagrisi>bol(48, 2)</arac_cagrisi>") == \
        "<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>"
    assert kit.calculate("4 + 5") == "9"
