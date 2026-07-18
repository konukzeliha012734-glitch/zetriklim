# Zetriklim

Zetriklim; kullanıcı tanımlı havza, il, ilçe veya bölge sınırlarında
doğrulanabilir iklim verisi, kuraklık, uzaktan algılama ve topoğrafya analizi üreten açık kaynaklı
bir uygulamadır. Sabit yüzölçümü sınırı yoktur.

## Temel özellikler

- Tek dosya GeoPackage, GeoJSON ve SHP yükleme
- Tam Shapefile paketi veya ZIP desteği
- Kullanıcıya ait Google Earth Engine projesiyle çalışma
- Project ID yardım ekranı ve oturuma özel Google Earth Engine yetkilendirmesi
- Kullanıcıya ait Climate Engine API anahtarını doğrulama ve poligon zaman serisi
- CHIRPS yağışı ve ERA5-Land sıcaklığı
- SPI-1/3/6/12 ve kullanıcı seçimli diğer ölçekler
- Sentinel-2 tabanlı NDVI, NDWI, NDMI, NDBI, EVI ve SAVI
- Landsat 8/9 Collection 2 Level-2 tabanlı LST
- SRTM tabanlı DEM, eğim ve bakı; MERIT Hydro + SRTM tabanlı TWI
- Bir çalışmada birden fazla analiz modülünü birlikte seçebilme
- Excel, CSV, GeoPackage, GeoJSON, PNG ve GeoTIFF çıktıları
- Seçilen dönem için CHIRPS toplam yağış ve ERA5-Land ortalama sıcaklık haritaları
- Havza sınırına kırpılmış SPI, yağış, sıcaklık, indis ve topoğrafya rasterleri
- Kaynak, yöntem ve işlem metadata kaydı
- Her raster için formül, ürün, dönem, sahne sayısı ve dışa aktarma çözünürlüğü

## Akademik Araştırma modu

- SPI ve SPEI için 1/3/6/9/12/18/24 aylık ölçekler ve kullanıcı tanımlı referans dönemi
- Gamma, Pearson Tip III ve log-logistic dağılım seçenekleri; KS uyum testi ve AIC
- Kuraklık başlangıcı, bitişi, süresi, şiddeti ve yoğunluğunu içeren olay kataloğu
- Mann–Kendall, mevsimsel Mann–Kendall, trend-free prewhitening, Sen eğimi ve Pettitt testi
- Benjamini–Hochberg çoklu karşılaştırma / FDR düzeltmesi
- SPI/SPEI ile NDVI, EVI ve LST arasında 0–12 aylık gecikmeli ilişki analizi
- ESA WorldCover sınıflarına göre orman, mera ve tarım alanı zonal karşılaştırmaları
- CHIRPS, ERA5-Land ve isteğe bağlı istasyon verisiyle Bias, MAE, RMSE, korelasyon ve KGE
- Kaynaklar arası ensemble yayılımı ve belirsizlik serisi
- Eksik ay, yinelenen tarih, fiziksel aralık ve uydu geçerli piksel kalite kontrolleri
- Excel/CSV tabloları, CBS katmanları, izlenebilir metadata ve bağımsız HTML bilimsel rapor

## Çalıştırma

Windows'ta önce `Kurulum-Windows.bat`, ardından `Zetriklim-Baslat.bat`
dosyasına çift tıklayabilirsiniz.

PowerShell ile:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

Kurulum tamamlandıktan sonra `Zetriklim-Baslat.bat` dosyasına çift tıklayarak da
uygulamayı başlatabilirsiniz.

## Climate Engine bağlantısı

Climate Engine anahtarını kaynak koduna yazmayın. Oturumdan önce ortam değişkeni
olarak tanımlayın:

```powershell
$env:CLIMATE_ENGINE_API_KEY="anahtarınız"
streamlit run streamlit_app.py
```

Climate Engine kataloğu kaynak seçeneği olarak korunur; anahtar bulunmadığında veri
uydurulmaz. Çalışır Earth Engine analizleri, ilgili üretici koleksiyonlarından canlı
olarak indirilir ve kaynak bilgisi çıktı paketine kaydedilir.

## Google Earth Engine bağlantısı

Her kullanıcı kendi Google Cloud / Earth Engine Project ID değerini kullanır.
Proje Earth Engine için kaydedildikten sonra bir defa:

```powershell
.\.venv\Scripts\python.exe gee_auth.py --project "YOUR_PROJECT_ID"
```

komutunu çalıştırın. Tarayıcıdaki Google izin ekranı tamamlandığında kimlik
bilgileri kullanıcının Earth Engine ayarlarında saklanır. `client_secret` dosyası
uygulamaya kopyalanmaz ve kaynak kontrolüne eklenmez.

SPI analizinde birincil yağış ürünü GEE üzerindeki CHIRPS Daily'dir. GEE bağlı
değilse uygulama ERA5 yağışını kullanır ve bu durumu metadata içinde açıkça
belirtir. Kaynaklar sessizce veya kayıtsız biçimde birleştirilmez.

## Coğrafi dosyalar

- `.gpkg` ve `.geojson`: Tek dosya, geometri ve öznitelikleri birlikte taşır.
- `.shp`: Tek başına geometri okunabilir. `.prj` olmadığı için CRS uygulamada
  kullanıcı tarafından belirtilir; öznitelikler mevcut olmaz.
- Tam Shapefile: `.shp`, `.shx`, `.dbf`, `.prj` birlikte seçilebilir veya ZIP
  olarak yüklenebilir.

## Lisans

Zetriklim [MIT Lisansı](LICENSE) ile yayımlanır.
