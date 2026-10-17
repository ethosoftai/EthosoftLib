from ethosoftlib.math.translate import MathTranslator, protect_all


def _fake(lines):
    """Sahte arka uc: kelimeleri 'TR:' ile isaretler, yer tutuculari aynen kopyalar."""
    return ["TR: " + ln for ln in lines]


def test_protect_all_masks_latex_and_symbols():
    m, sp = protect_all("If $x^2=9$ then x = ±3 and 你好.")
    assert "$x^2=9$" in sp and "±" in sp and "你好" in sp
    assert "[[1]]" in m and "$" not in m


def test_roundtrip_with_fake_backend_restores_everything():
    t = MathTranslator(translate_lines=_fake)
    src = "Solve $x^2-4=0$ for x.\n\n$$x=\\pm 2$$\nThe answer is ±2 meters."
    out = t.translate(src, strict=True)
    assert out is not None
    assert "$x^2-4=0$" in out and "$$x=\\pm 2$$" in out and "±" in out
    assert out.count("\n") == src.count("\n")
    assert out.startswith("TR: Solve")


def test_broken_placeholder_is_rejected_when_strict():
    def drop(lines):
        return [ln.replace("[[1]]", "") for ln in lines]
    t = MathTranslator(translate_lines=drop)
    assert t.translate("Compute $x^2$ now please.", strict=True) is None
    nonstrict = t.translate("Compute $x^2$ now please.", strict=False)
    assert "$x^2$" in nonstrict            # eksik parca sona eklenir


def test_translate_record_adds_fields():
    t = MathTranslator(translate_lines=_fake)
    r = t.translate_record({"problem": "What is $1+1$?", "solution": "It is $2$."})
    assert r["problem_tr"].startswith("TR:") and "$1+1$" in r["problem_tr"]
    assert r["translator"]["provider"] == "ethosoft-math-translator"


def _tr_backend(lines):
    """Sahte 'Turkce' arka uc: sayilari Turkce bicime cevirir, yer tutuculari korur."""
    table = {"Peter has 3 apples and buys 4 more, so 3 + 4 = 7 apples.": "Peter'ın 3 elması var ve 4 tane daha alır, yani 3 + 4 = 7 elma.",
             "Each costs 4.50 dollars, so 2 * 4.50 = 9.00 dollars.": "Her biri 4,50 dolardır, yani 2 * 4,50 = 9,00 dolar."}
    return [table.get(ln, ln) for ln in lines]


def test_translate_with_tools_converts_calculations_after_translation():
    t = MathTranslator(translate_lines=_tr_backend)
    out = t.translate("Peter has 3 apples and buys 4 more, so 3 + 4 = 7 apples.", tools=True)
    assert "<arac_cagrisi>topla(3, 4)</arac_cagrisi><arac_sonucu>7</arac_sonucu>" in out
    assert "elma" in out


def test_translate_with_tools_handles_turkish_decimal_comma():
    t = MathTranslator(translate_lines=_tr_backend)
    out = t.translate("Each costs 4.50 dollars, so 2 * 4.50 = 9.00 dollars.", tools=True)
    assert "carp(2, 4.5)" in out


def test_translate_record_tools_only_touches_solution():
    t = MathTranslator(translate_lines=_tr_backend)
    r = t.translate_record({"problem": "Each costs 4.50 dollars, so 2 * 4.50 = 9.00 dollars.",
                            "solution": "Peter has 3 apples and buys 4 more, so 3 + 4 = 7 apples."}, tools=True)
    assert "<arac_cagrisi>" in r["solution_tr"] and "<arac_cagrisi>" not in r["problem_tr"]
