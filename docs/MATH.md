# ethosoftlib.math

Matematik çözümlerini **doğrulanmış tool-use izlerine** çeviren, modelin araç çağrılarını
çalıştıran ve serbest ifadeleri hesaplayan modül. Tamamen Python; tek gereksinim `sympy`.

```bash
pip install 'ethosoftlib[math]'
```

Kök paket `ethosoftlib` bu modülü ve `sympy`'yi **yüklemez**; yalnızca `import ethosoftlib.math` yapınca yüklenir.

## Biçim

```
<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>
```

Model `<arac_cagrisi>` bloğunu yazar, `<arac_sonucu>` bloğunu araç doldurur.
Fonksiyon adları Türkçedir (`topla`, `cikar`, `carp`, `bol`, `us`, `karekok`, `coz`, `turev`...);
İngilizce adlar (`add`, `sub`, `mul`...) da çalışır. `lang="en"` ile `<tool_call>` biçimi üretilir.
Tüm fonksiyonlar: [MATH_PROTOCOL.md](MATH_PROTOCOL.md).

## Kullanım

```python
import ethosoftlib.math as em

# 1) Araç çalıştırıcı: modelin çağrısını çalıştırıp sonucu yaz
em.fill_results("<arac_cagrisi>bol(48, 2)</arac_cagrisi>")
# '<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>'
em.result_block("bol(48, 2)")            # '<arac_sonucu>24</arac_sonucu>'

# 2) Düz metin -> tool-use
em.tool_use("Mayıs'ta 48 / 2 = 24 kart sattım.")

# 3) Çözüm metnindeki doğrulanabilir hesapları çevir (yanlış eşitlikler olduğu gibi kalır)
em.convert("Toplam 3 + 4 = 7 elma.")
em.verify(em.convert(text))              # (çağrı_sayısı, hatalı_sayısı)

# 4) Hesap makinesi
em.calculate("4 + 5")                    # '9'
em.calculate("50'nin %20'si")            # '10'
em.calculate("100 / 3", decimals=2)      # '33.33'
```

Registry üzerinden:

```python
from ethosoftlib import providers

kit = providers.create("tooluse", "math", lang="tr")
kit.fill_results("<arac_cagrisi>carp(topla(3, 4), 2)</arac_cagrisi>")
```

Komut satırı: `python -m ethosoftlib.math fill "<arac_cagrisi>bol(48, 2)</arac_cagrisi>"`
veya `ethosoftlib-math calc "4 + 5"`.

## Çalışma ilkeleri

- Her dönüştürülen çağrı, çıktıya girmeden **yeniden çalıştırılıp doğrulanır**; tutmayan atılır.
- Python/PoT kodu içeren metne dokunulmaz.
- Çeviri modelinden önce çalıştırılmalıdır; araç blokları çeviriden korunmalıdır.
- `scripts/math/` altında kapsam ölçüm araçları bulunur (paketin parçası değildir).

## Sınırlar

Ölçülen kapsam (ayarlanmamış örneklerde): hesabı olan çözüm satırlarının ~%85'inde en az bir
doğrulanmış çağrı üretilir; geri kalanı demet, sözel sayı, türev cümlesi gibi kural dışı yazımlardır.

## Ceviri (EN -> TR): Ethosoft-Math-Translator

`ethosoftlib.math.translate`, ince ayarli [`Ethosoft/Ethosoft-Math-Translator`](https://huggingface.co/Ethosoft/Ethosoft-Math-Translator)
modeliyle (NedoTranslator tabanli) matematik metnini Turkceye cevirir. LaTeX, kod, URL, arac bloklari ve matematik
sembolleri `[[n]]` ile maskelenir, sadece soz kismi cevrilir, formuller birebir geri konur. Formulu garanti edilemeyen
satir bozulmaz, reddedilir (`None`).

```bash
pip install "ethosoftlib[translate]"
```

```python
import ethosoftlib.math as em

t = em.translator()                                  # Ethosoft/Ethosoft-Math-Translator
t.translate("If $x^2 = 9$, then x = ±3.")
t.translate_record({"problem": "What is $1+1$?", "solution": "It is $2$."})   # None: formul risk altinda

em.MathToolkit().translate("Compute $x^2$.")         # registry cephesi
```

```bash
ethosoftlib-math translate "Compute $x^2$ for x = 3."
ethosoftlib-math translate --input data.jsonl --output data_tr.jsonl
```

Model karti ve sonuclar: <https://huggingface.co/Ethosoft/Ethosoft-Math-Translator>.
