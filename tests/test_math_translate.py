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
