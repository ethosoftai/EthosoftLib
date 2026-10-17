import ethosoftlib.math as em


def _conv(s, **kw):
    out, _ = em.convert_text(s, **kw)
    return out


def test_turkish_units_with_special_letters():
    out = _conv("20 erkek öğrenci * 42 kg/öğrenci = 840 kg")
    assert "<arac_cagrisi>carp(20, 42)</arac_cagrisi><arac_sonucu>840</arac_sonucu>" in out


def test_turkish_decimal_comma_and_currency_suffix():
    out = _conv("Her biri 4,50 $ olan 2 hamur işi alır, yani bu 2*4,50 = 9,00 $'dır.")
    assert "carp(2, 4.5)" in out and "$'dır" in out
    assert "4,50 $ olan" in out            # donusmeyen yerdeki Turkce sayi bicimi korunur
    assert em.verify_converted(out) == (1, 0)


def test_turkish_thousands_dot():
    out = _conv("Fabrika ayda 50.000 bar üretir; 50.000 - 36.000 = 14.000 bar kalır.")
    assert "cikar(50000, 36000)" in out
    assert "ayda 50.000 bar" in out


def test_turkish_word_operators():
    out = _conv("12 çarpı 5 eşittir 60 ve 60 bölü 4 eşittir 15.")
    assert "carp(12, 5)" in out and "bol(60, 4)" in out


def test_x_as_multiplication_between_turkish_words():
    out = _conv("toplam 10 sıra x 20 gül = 200 gül vardır.")
    assert "carp(10, 20)" in out


def test_unicode_times_and_divide_signs_are_operators_not_letters():
    # regresyon: U+00D7 / U+00F7 harf araligina girmemeli
    assert "carp(6, 7)" in _conv("Toplam 6 × 7 = 42 elma.")
    assert "bol(48, 2)" in _conv("48 ÷ 2 = 24")
    assert "carp(6, 7)" in _conv("The area is 6 × 7 = 42 square units.")


def test_english_unaffected_by_locale_auto():
    assert _conv("The total is 1,500 + 2,500 = 4,000 dollars.") == _conv("The total is 1,500 + 2,500 = 4,000 dollars.", locale="en")
    assert "topla(1500, 2500)" in _conv("The total is 1,500 + 2,500 = 4,000 dollars.")


def test_explicit_locale_tr_forces_comma_decimal():
    out = _conv("3,5 + 1,5 = 5", locale="tr")
    assert "topla(3.5, 1.5)" in out
