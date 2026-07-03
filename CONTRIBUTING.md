# Zetriklim'e katkı

Zetriklim açık kaynaklı ve yeniden üretilebilir havza iklim analizi için
geliştirilir.

## Geliştirme ilkeleri

- Kaynağı, sürümü ve birimi doğrulanamayan veri eklenmez.
- Sessiz veri kaynağı değişimi veya eksik değeri sıfırla doldurma yapılmaz.
- Her yeni veri bağlayıcısı kaynak URL'si ve atıf bilgisini taşımalıdır.
- SPI hesap değişiklikleri sentetik ve gerçek yağış serileriyle test edilmelidir.
- Kimlik bilgileri, OAuth dosyaları ve özel anahtarlar repoya eklenmez.

## Yerel geliştirme

1. Sanal ortam oluşturun.
2. `pip install -r requirements.txt` çalıştırın.
3. `streamlit run app.py` ile uygulamayı açın.
4. GEE kullanacaksanız kendi Project ID'nizle `gee_auth.py` çalıştırın.

Değişikliklerle birlikte küçük, yeniden üretilebilir bir test senaryosu ekleyin.
