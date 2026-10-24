# Faz 1 — Veri denetimi (2026-10-07)

Script: `scripts/rev1/data_audit.py` → `data_audit.json`, `tab_descriptive.{tex,csv}`, `fig_variable_distributions.{pdf,png}`
Baz tex: `rev1/pinam_latex_submitted/main.tex` (satır numaraları bu dosyaya göre)

## 1. Kayıt sayısı — tex DOĞRU
- xlsx: 731 kayıt (454 uçuş / 277 uçuşsuz), 2023-01-01 – 2024-12-31.
- Model: **728 kayıt (451 / 277)**, 2023-01-04 – 2024-12-31.
- Atılan 3 kayıt: 2023-01-01, 01-02 ve 01-03; üçü de flight=1.
- Atılma sebebi: `build_features` içindeki eski pipeline'dan kalan gecikmeli (lag) ve rolling sütunların ilk günlerde NaN olması ve bu satırların `dropna` ile düşmesi. PI-NAM bu sütunları kullanmıyor, ama aynı 728 satırla çalışıyor. Tex'teki "insufficient temporal history" açıklaması özünde doğru.
- Uygulama notu: PI-NAM'in 14 günlük penceresi, ilk günlerde (2023-01-04'ten itibaren) eksik geçmişi ilk satırı tekrarlayarak dolduruyor (padding). Etkisi ihmal edilebilir; Faz 9 dokümantasyonunda belirtilecek.

## 2. Değişken sayısı — 16
xlsx'te `datetime` ve `flight` dışında 16 meteorolojik sütun var. Model de 16 kanal kullanıyor. "17", büyük olasılıkla hedef sütununun da sayılmasından kaynaklanıyor.

## 3. Visual Crossing tanımları (resmî doküman, metrik birim grubu)
Kaynaklar: visualcrossing.com, *Timeline Weather API* ve *Unit groups and measurement units* sayfaları.

| Değişken | Birim | Günlük değerin tanımı | Table 1'deki ifade |
|---|---|---|---|
| tempmax / tempmin / temp | °C | max / min / ortalama | ✓ |
| feelslike(max/min) | °C | max / min / ortalama | ✓ |
| humidity | % | ortalama | ✓ |
| precip | mm | günlük toplam | ✓ |
| **snow** | **cm** | günlük kar yağışı | ✗ "mm" yazıyor |
| snowdepth | cm | yerdeki kar kalınlığı | ✓ |
| windgust | km/h | **günün en yüksek saatlik değeri** | ✓ ("maximum") |
| **windspeed** | km/h | **günün en yüksek saatlik sürekli rüzgâr hızı** | ✗ "Mean wind speed" yazıyor |
| winddir | ° | ortalama | ✓ |
| sealevelpressure | hPa | ortalama | ✓ |
| cloudcover | % | ortalama | ✓ |
| visibility | km | ortalama | ✓ |

## 4. ⚠ Tex'te yanlış olan bir iddia (satır 204)
Tex: *"No instances of meteorological-label contradictions (e.g., a flight day associated with >50 km/h gusts) were observed."*

Veri: **27 uçuş gününde günlük max hamle 50 km/h'yi geçiyor (en yüksek 77.8 km/h).**

Bu fiziksel bir çelişki değil. `windgust`, gün içindeki en yüksek saatlik değer ve büyük olasılıkla öğleden sonra konvektif ısınma sırasında oluşuyor. Uçuşlar ise sabah erken, sakin saatlerde yapılıyor. Ama cümle bu hâliyle yanlış ve hakem veriyi indirip kontrol ederse hemen fark eder.

Ek fayda: bu bulgu, günlük max değerlerin şafak koşullarını birebir temsil etmediğini gösteriyor. Karar anı tartışmasında (R2-2, R3-1) ve sınırlılıklar bölümünde dürüst bir argüman olarak kullanılabilir.

## 5. Tex düzeltme listesi (kullanıcı uygulayacak)
| Satır | Mevcut | Önerilen |
|---|---|---|
| 171 | "The 17 raw meteorological features" | "The 16 meteorological variables" (+ "together with the binary flight label") |
| 188 | `snow & mm & Snowfall equivalent` | `snow & cm & Daily snowfall` |
| 191 | `windspeed & km/h & Mean wind speed` | `windspeed & km/h & Maximum hourly sustained wind speed` |
| 190 (windgust) | "Maximum recorded wind gust" | "Maximum hourly wind gust" (tutarlılık için) |
| 204 | "728 × 17 raw data cells" | "728 × 16 meteorological cells" |
| 204 | ">50 km/h gust" çelişki cümlesi | Kaldırılmalı veya düzeltilmeli (bkz. madde 4) |
| 204 | "mean 12.9 ± 8.6 °C" | **13.0 ± 8.5 °C** (hesaplanan: 12.99 ± 8.55) |
| 204 | "dual-source cross-referencing" | Neyin neyle karşılaştırıldığı açıkça yazılmalı (hakem sorabilir) |
| Table 1 caption | — | "Daily aggregates as defined by Visual Crossing (metric units)" eklenebilir |

Doğru olanlar: wind gust aralığı 12.6–87.1 km/h ✓; günlük ortalama sıcaklık aralığı −8.2 / 31.5 °C ✓; eksik veya sonsuz hücre yok ✓; N = 728 ve 451/277 ✓.

## 6. R3-10 için yeni materyal
- `fig_variable_distributions.pdf`: 16 değişkenin sınıfa göre violin grafikleri, 11 pt font.
- `tab_descriptive.tex`: genel ve sınıf bazında ortalama ± SD, aralık (`\label{tab:descriptive}`).
- Göze çarpanlar: no-flight günlerinde bulut kapalılığı belirgin şekilde yüksek (65.8'e karşı 34.3 %), hamle daha güçlü (39.5'e karşı 31.0 km/h), nem daha yüksek, sıcaklık daha düşük. Görüş mesafesinin ortalaması iki sınıfta neredeyse aynı (13.7 km), ama dağılımları farklı.
