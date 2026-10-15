import pytest
pytest.importorskip("sympy")
from ethosoftlib.math.mask import protect, restore, check, numbers_preserved


def test_protects_latex_tool_blocks_and_code():
    src = ('Cozum: $x^2 - 4 = 0$ ise <arac_cagrisi>coz("x^2-4=0", "x")</arac_cagrisi><arac_sonucu>[-2, 2]</arac_sonucu> '
           've \[ \frac{a}{b} \] ile `print(1)`.')
    m = protect(src)
    assert "$" not in m.text and "arac_cagrisi" not in m.text and "frac" not in m.text and "print" not in m.text
    assert len(m.spans) == 4
    assert restore(m.text, m) == src


def test_currency_is_not_math():
    m = protect("Kitap $5 ve defter $10 eder.")
    assert m.spans == []


def test_restore_after_fake_translation():
    m = protect("Mayıs'ta <arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu> kart sattım, $x$ bilinmiyor.")
    tr = m.text.replace("Mayıs'ta", "In May").replace("kart sattım", "cards sold").replace("bilinmiyor", "unknown")
    out = restore(tr, m)
    assert out is not None and "<arac_cagrisi>bol(48, 2)</arac_cagrisi>" in out and "$x$" in out


def test_strict_restore_rejects_broken_placeholders():
    m = protect("A $x^2$ B $y^2$ C")
    assert restore("A [[1]] B C", m) is None
    assert check("A [[1]] B [[2]] [[2]]", m)
    assert "$y^2$" in restore("A [[1]] B C", m, strict=False)


def test_mask_numbers_and_number_preservation():
    m = protect("3 elma ve 4.5 kg", mask_numbers=True)
    assert m.spans == ["3", "4.5"]
    assert numbers_preserved("3 elma ve 4.5 kg", "3 apples and 4.5 kg")
    assert not numbers_preserved("3 elma ve 4.5 kg", "2 apples and 4.5 kg")
