# MathTool v1 - Matematik Tool-Use Protokolu

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


### Aritmetik

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `hesapla("ifade")` | `calc` | serbest yazilmis ifadeyi hesaplar: calc("4 + 5") -> 9 |
| `topla(a, b, ...)` | `add` | toplama (+) |
| `cikar(a, b)` | `sub` | cikarma (-) |
| `carp(a, b, ...)` | `mul` | carpma (*, x, ×, ·) |
| `bol(a, b)` | `div` | bolme, kesin oran (/, ÷) |
| `tam_bol(a, b)` | `idiv` | tam bolme, tabana yuvarlar (//) |
| `mod(a, b)` | `mod` | mod, kalan (%) |
| `us(a, b)` | `pow` | us alma (^) |
| `negatif(a)` | `neg` | isaret degistirme (-a) |
| `mutlak(a)` | `abs` | mutlak deger |a| |
| `karekok(a)` | `sqrt` | karekok (√) |
| `kok(n, a)` | `root` | n. dereceden kok |
| `taban(a)` | `floor` | tabana yuvarlama |
| `tavan(a)` | `ceil` | tavana yuvarlama |
| `yuvarla(a, nd=0)` | `round` | nd basamaga yuvarla (yarim yukari) |
| `en_kucuk(a, b, ...)` | `min` | en kucuk |
| `en_buyuk(a, b, ...)` | `max` | en buyuk |
| `ortalama(a, b, ...)` | `avg` | aritmetik ortalama |

### Yuzde

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `yuzde_de(p, x)` | `pct_of` | x'in yuzde p'si = p*x/100 |
| `yuzde_kac(a, b)` | `pct` | a, b'nin yuzde kaci = a/b*100 |
| `yuzde_degisim(old, new)` | `pct_change` | yuzde degisim = (new-old)/old*100 |

### Sayi teorisi / sayma

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `faktoriyel(n)` | `fact` | faktoriyel (n!), n <= 500 |
| `kombinasyon(n, k)` | `comb` | kombinasyon C(n,k) |
| `permutasyon(n, k)` | `perm` | permutasyon P(n,k) |
| `obeb(a, b, ...)` | `gcd` | OBEB |
| `okek(a, b, ...)` | `lcm` | OKEK |
| `asal_mi(n)` | `is_prime` | asal mi? true/false |
| `asal_carpanlar(n)` | `factorint` | asal carpanlara ayirma, orn. 2^3 * 3 |
| `bolen_sayisi(n)` | `n_divisors` | bolen sayisi |
| `rakam_toplami(n)` | `digit_sum` | rakamlar toplami |
| `tabana_cevir(n, b)` | `to_base` | n'yi b tabaninda yaz (2..36) |
| `tabandan_cevir("s", b)` | `from_base` | b tabanindaki s'yi onluga cevir |

### Us, logaritma, sabitler

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `ustel(x)` | `exp` | e^x |
| `dogal_log(x)` | `ln` | dogal logaritma |
| `log(x, base=10)` | `log` | logaritma |
| `pi()` | `pi` | π sabiti |
| `euler()` | `e` | Euler sabiti |

### Trigonometri

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `dereceden_radyana(x)` | `deg` | dereceyi radyana cevir: x*pi/180 |
| `radyandan_dereceye(x)` | `rad2deg` | radyani dereceye cevir |
| `sin(x)` | `sin` | sinus (radyan) |
| `cos(x)` | `cos` | kosinus (radyan) |
| `tan(x)` | `tan` | tanjant (radyan) |
| `cot(x)` | `cot` | kotanjant |
| `sec(x)` | `sec` | sekant |
| `csc(x)` | `csc` | kosekant |
| `arcsin(x)` | `asin` | arcsin |
| `arccos(x)` | `acos` | arccos |
| `arctan(x)` | `atan` | arctan |
| `sinh(x)` | `sinh` | sinh |
| `cosh(x)` | `cosh` | cosh |
| `tanh(x)` | `tanh` | tanh |
| `arctan2(y, x)` | `atan2` | iki arguman arctan |

### Geometri

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `hipotenus(a, b)` | `hypot` | hipotenus = sqrt(a^2+b^2) |
| `uzaklik(x1, y1, x2, y2)` | `dist` | iki nokta arasi uzaklik |
| `daire_alani(r)` | `area_circle` | daire alani pi*r^2 |
| `cember_uzunlugu(r)` | `circumference` | cember uzunlugu 2*pi*r |
| `ucgen_alani(b, h)` | `area_triangle` | ucgen alani b*h/2 |
| `heron_alani(a, b, c)` | `area_heron` | Heron formulu |
| `yamuk_alani(a, b, h)` | `area_trapezoid` | yamuk alani (a+b)*h/2 |
| `kure_hacmi(r)` | `vol_sphere` | kure hacmi 4/3*pi*r^3 |
| `kure_yuzeyi(r)` | `area_sphere` | kure yuzeyi 4*pi*r^2 |
| `silindir_hacmi(r, h)` | `vol_cylinder` | silindir hacmi pi*r^2*h |
| `koni_hacmi(r, h)` | `vol_cone` | koni hacmi pi*r^2*h/3 |
| `cokgen_aci_toplami(n)` | `polygon_angle_sum` | n-genin ic acilar toplami (n-2)*180 |

### Istatistik / olasilik

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `medyan(a, b, ...)` | `median` | medyan |
| `tepe_deger(a, b, ...)` | `mode` | tepe deger (esitlikte en kucuk) |
| `varyans(a, b, ...)` | `variance` | populasyon varyansi |
| `standart_sapma(a, b, ...)` | `stdev` | populasyon standart sapmasi |
| `binom_olasilik(n, k, p)` | `binom_pmf` | binom olasilik C(n,k) p^k (1-p)^(n-k) |

### Diziler

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `aritmetik_terim(a1, d, n)` | `arith_nth` | aritmetik dizi n. terim |
| `aritmetik_toplam(a1, d, n)` | `arith_sum` | aritmetik dizi ilk n terim toplami |
| `geometrik_terim(a1, r, n)` | `geom_nth` | geometrik dizi n. terim |
| `toplam_aralik("expr", "i", a, b)` | `sum_range` | toplam sum_{i=a}^{b} expr (b-a <= 100000) |
| `carpim_aralik("expr", "i", a, b)` | `prod_range` | carpim prod_{i=a}^{b} expr (b-a <= 2000) |
| `geometrik_toplam(a1, r, n)` | `geom_sum` | geometrik dizi ilk n terim toplami |

### Cebir (sembolik)

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `sadelestir("expr")` | `simplify` | sadelestir |
| `ac("expr")` | `expand` | parantez ac |
| `carpanlara_ayir("expr")` | `factor` | carpanlara ayir |
| `coz("eq", "x")` | `solve` | denklem veya esitsizlik coz (esitlik yoksa =0 varsayilir) |
| `denklem_sistemi_coz("eq1; eq2", "x, y")` | `solve_system` | denklem sistemi coz |
| `esit_mi(a, b)` | `equal` | iki ifade ozdes/esit mi? true/false (sayi veya "ifade") |
| `yerine_koy("expr", "x", value)` | `subs` | degisken yerine deger koy |
| `ondalik("expr", digits=10)` | `evalf` | ondalik yaklasik deger |

### Analiz

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `turev("expr", "x")` | `diff` | turev |
| `integral("expr", "x", a?, b?)` | `integrate` | integral (belirsiz veya belirli) |
| `limit("expr", "x", point)` | `limit` | limit (point: sayi veya "oo") |

### Lineer cebir

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `determinant("[[1,2],[3,4]]")` | `det` | determinant |
| `ters_matris("[[1,2],[3,4]]")` | `inv` | ters matris |
| `matris_carp(A, B)` | `matmul` | matris carpimi |
| `devrik(A)` | `transpose` | devrik matris |
| `ic_carpim(u, v)` | `dot` | ic carpim |
| `vektorel_carpim(u, v)` | `cross` | vektorel carpim (3B) |
| `norm(u)` | `norm` | vektor uzunlugu |

### Fonksiyon tanimlama

| Cagri | Ingilizce ad | Aciklama |
|---|---|---|
| `tanimla("f", "x, y", "x^2 + y")` | `define` | f(x,y) fonksiyonu tanimla; sonra f(2, 3) diye cagrilir |

Toplam: **94 fonksiyon**.

