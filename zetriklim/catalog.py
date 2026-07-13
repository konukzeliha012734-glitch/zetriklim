"""Açık veri kaynakları, değişkenler ve ayrıntılı analiz kataloğu."""

from datetime import date

SOURCES = {
    "Otomatik en uygun açık kaynak": {
        "status": "Hazır",
        "description": "Değişken, dönem, konum ve çözünürlüğe göre uygun ürün önerilir.",
        "products": ["Otomatik ürün eşleştirme"],
    },
    "Climate Engine": {
        "status": "API anahtarı gerekli",
        "description": "CHIRPS, ERA5, MODIS ve diğer ürünlere ortak analiz arayüzü.",
        "products": ["CHIRPS Daily", "ERA5", "ERA5-Land", "TerraClimate", "MODIS", "Landsat", "Sentinel-2"],
    },
    "Google Earth Engine": {
        "status": "OAuth kullanıcı girişi gerekli",
        "description": "CHIRPS, ERA5-Land, Sentinel, Landsat ve MODIS raster koleksiyonları.",
        "products": ["CHIRPS Daily", "ERA5-Land Daily", "Sentinel-2", "Landsat Collection 2", "MODIS"],
    },
    "Copernicus Climate Data Store": {
        "status": "Ücretsiz hesap gerekli",
        "description": "ERA5/ERA5-Land ve iklim projeksiyonlarının üretici dağıtım noktası.",
        "products": ["ERA5 Hourly", "ERA5-Land Hourly", "ERA5 Monthly", "CMIP6"],
    },
    "CHIRPS / UCSB Climate Hazards Center": {
        "status": "Açık erişim",
        "description": "1981'den günümüze istasyon destekli küresel yağış.",
        "products": ["CHIRPS Daily", "CHIRPS Pentad", "CHIRPS Monthly"],
    },
    "NASA Earthdata": {
        "status": "Ücretsiz hesap gerekli",
        "description": "Uydu, hidroloji, arazi yüzeyi ve atmosfer ürünleri.",
        "products": ["GPM IMERG", "MODIS", "SMAP", "GLDAS", "MERRA-2", "GRACE"],
    },
    "NASA POWER": {
        "status": "Açık erişim",
        "description": "Noktasal ve bölgesel agroklimatolojik meteoroloji serileri.",
        "products": ["POWER Daily", "POWER Hourly", "POWER Monthly"],
    },
    "NOAA": {
        "status": "Açık erişim",
        "description": "Gözlem, yeniden analiz, deniz yüzeyi ve iklim ürünleri.",
        "products": ["NCEP/NCAR", "CPC", "CMORPH", "OISST", "GHCN"],
    },
    "Open-Meteo Historical": {
        "status": "Açık erişim",
        "description": "ERA5 tabanlı geçmiş hava serilerine hızlı API erişimi.",
        "products": ["Historical Weather API", "Climate API"],
    },
    "ESA Copernicus Data Space": {
        "status": "Ücretsiz hesap gerekli",
        "description": "Sentinel uydu verileri ve Copernicus servisleri.",
        "products": ["Sentinel-1", "Sentinel-2", "Sentinel-3", "Copernicus DEM"],
    },
    "USGS EarthExplorer": {
        "status": "Ücretsiz hesap gerekli",
        "description": "Landsat, SRTM ve uzun dönemli yeryüzü gözlemleri.",
        "products": ["Landsat Collection 2", "SRTM", "ASTER GDEM"],
    },
    "Yerel dosya": {
        "status": "Hazır",
        "description": "CSV, Excel, NetCDF veya GeoTIFF verisini analiz eder.",
        "products": ["CSV", "Excel", "NetCDF", "GeoTIFF"],
    },
}

VARIABLES = {
    "Yağış": ["CHIRPS Daily", "GPM IMERG", "ERA5-Land Hourly", "CMORPH", "TerraClimate"],
    "Hava sıcaklığı": ["ERA5 Hourly", "ERA5-Land Hourly", "POWER Daily", "TerraClimate"],
    "Yüzey sıcaklığı (LST)": ["MODIS", "Sentinel-3", "Landsat Collection 2"],
    "Bağıl nem": ["ERA5 Hourly", "MERRA-2", "POWER Hourly"],
    "Çiy noktası": ["ERA5 Hourly", "ERA5-Land Hourly", "MERRA-2"],
    "Rüzgâr hızı ve yönü": ["ERA5 Hourly", "ERA5-Land Hourly", "MERRA-2"],
    "Buharlaşma / gerçek ET": ["ERA5-Land Hourly", "MODIS", "GLDAS"],
    "Potansiyel evapotranspirasyon": ["ERA5-Land Hourly", "TerraClimate", "POWER Daily"],
    "Yüzey / deniz seviyesi basıncı": ["ERA5 Hourly", "MERRA-2", "POWER Hourly"],
    "Toprak nemi": ["SMAP", "ERA5-Land Hourly", "GLDAS"],
    "Kar örtüsü / kar su eşdeğeri": ["MODIS", "ERA5-Land Hourly", "GLDAS"],
    "Güneş radyasyonu": ["ERA5-Land Hourly", "POWER Daily", "MERRA-2"],
    "Bulutluluk": ["ERA5 Hourly", "MODIS", "Sentinel-3"],
    "NDVI / EVI": ["Sentinel-2", "MODIS", "Landsat Collection 2"],
    "NDWI / yüzey suyu": ["Sentinel-2", "Landsat Collection 2", "MODIS"],
    "Arazi örtüsü": ["ESA WorldCover", "MODIS", "Copernicus Global Land Cover"],
    "Yükselti / eğim / bakı": ["Copernicus DEM", "SRTM", "ASTER GDEM"],
    "Yeraltı suyu depolama anomalisi": ["GRACE"],
}

ANALYSES = {
    "Kuraklık indisleri": ["SPI", "SPEI", "PDSI", "scPDSI", "EDDI", "RDI", "Z-indeksi"],
    "Kuraklık olay karakteri": ["Süre", "Şiddet", "Yoğunluk", "Sıklık", "Başlangıç-bitiş", "Alan yayılımı"],
    "Eğilim ve homojenlik": ["Mann–Kendall", "Mevsimsel Mann–Kendall", "Sen eğimi", "Pettitt değişim noktası", "SNHT", "Buishand"],
    "İklim uçları (ETCCDI)": ["CDD", "CWD", "R10mm", "R20mm", "Rx1day", "Rx5day", "TX90p", "TN10p", "WSDI", "CSDI", "Don günü"],
    "Hidroklimatoloji": ["Su dengesi", "Akış katsayısı", "Yağış anomalisi", "ET anomalisi", "Toprak nemi anomalisi"],
    "Uzaktan algılama indisleri": ["NDVI", "NDWI", "NDMI", "NDBI", "EVI", "SAVI", "LST"],
    "Topoğrafik ve hidrolojik türevler": ["DEM", "SLOPE", "ASPECT", "TWI"],
    "Bitki ve yüzey": ["VHI", "VCI", "TCI", "Yağış-bitki gecikmeli korelasyonu"],
    "Mekânsal istatistik": ["Zonal istatistik", "Moran's I", "Getis-Ord Gi*", "IDW", "Kriging", "Yükseklik kuşakları"],
    "Föhn ve topoğrafya": ["Rüzgâr üstü-altı farkı", "Sıcaklık-nem farkı", "Rüzgârın sırta dik bileşeni", "Föhn olay sınıflaması"],
}

ANALYSIS_METHODS = {
    "SPI": {
        "title": "Standartlaştırılmış Yağış İndisi",
        "source": "CHIRPS Daily",
        "resolution": "~5,5 km",
        "purpose": "Meteorolojik kuraklığın farklı zaman ölçeklerinde izlenmesi",
    },
    "SPEI": {
        "title": "Standartlaştırılmış Yağış-Evapotranspirasyon İndisi",
        "source": "CHIRPS + ERA5-Land",
        "resolution": "~5,5–11 km",
        "purpose": "Yağış ve atmosferik su talebini birlikte kullanarak iklimsel kuraklığın izlenmesi",
    },
    "NDVI": {
        "title": "Normalize Edilmiş Fark Bitki Örtüsü İndisi",
        "source": "Sentinel-2 L2A",
        "resolution": "10 m",
        "purpose": "Bitki örtüsü canlılığı ve yoğunluğu",
    },
    "NDWI": {
        "title": "Normalize Edilmiş Fark Su İndisi",
        "source": "Sentinel-2 L2A",
        "resolution": "10 m",
        "purpose": "Açık su yüzeylerinin belirlenmesi",
    },
    "NDMI": {
        "title": "Normalize Edilmiş Fark Nem İndisi",
        "source": "Sentinel-2 L2A",
        "resolution": "20 m",
        "purpose": "Vejetasyon ve yüzey nemi",
    },
    "NDBI": {
        "title": "Normalize Edilmiş Fark Yapılaşma İndisi",
        "source": "Sentinel-2 L2A",
        "resolution": "20 m",
        "purpose": "Yapılaşmış alanların belirlenmesi",
    },
    "EVI": {
        "title": "Geliştirilmiş Bitki Örtüsü İndisi",
        "source": "Sentinel-2 L2A",
        "resolution": "10 m",
        "purpose": "Yoğun bitki örtüsünde geliştirilmiş duyarlılık",
    },
    "SAVI": {
        "title": "Toprak Ayarlı Bitki Örtüsü İndisi",
        "source": "Sentinel-2 L2A",
        "resolution": "10 m",
        "purpose": "Seyrek bitki örtüsünde toprak etkisinin azaltılması",
    },
    "LST": {
        "title": "Arazi Yüzey Sıcaklığı",
        "source": "Landsat 8/9 Collection 2 L2",
        "resolution": "30 m",
        "purpose": "Termal yüzey örüntüsü ve sıcaklık anomalileri",
    },
    "DEM": {
        "title": "Sayısal Yükseklik Modeli",
        "source": "SRTM V3",
        "resolution": "30 m",
        "purpose": "Yükselti dağılımı",
    },
    "SLOPE": {
        "title": "Eğim",
        "source": "SRTM V3",
        "resolution": "30 m",
        "purpose": "Topoğrafik eğim derecesi",
    },
    "ASPECT": {
        "title": "Bakı",
        "source": "SRTM V3",
        "resolution": "30 m",
        "purpose": "Yamaç yönelimi",
    },
    "TWI": {
        "title": "Topoğrafik Nem İndisi",
        "source": "MERIT Hydro + SRTM",
        "resolution": "~90 m",
        "purpose": "Topoğrafik su birikme potansiyeli",
    },
}


def academic_data_package(components: list[str] | tuple[str, ...]) -> dict[str, object]:
    """Seçilen akademik bileşenler için gerekli GEE koleksiyonlarını eşleştirir."""
    allowed = {"SPI", "SPEI", "NDVI", "EVI", "LST"}
    selected = list(dict.fromkeys(component for component in components if component in allowed))
    if not selected:
        selected = ["SPEI"]

    collections: list[str] = []
    variables: list[str] = []

    def add_unique(target: list[str], *values: str) -> None:
        for value in values:
            if value not in target:
                target.append(value)

    if any(component in selected for component in ("SPI", "SPEI")):
        add_unique(collections, "CHIRPS")
        add_unique(variables, "Yağış")
    if "SPEI" in selected:
        add_unique(collections, "ERA5-Land")
        add_unique(variables, "Hava sıcaklığı", "Potansiyel evapotranspirasyon")
    if any(component in selected for component in ("NDVI", "EVI")):
        add_unique(collections, "Sentinel-2 SR Harmonized")
    if "NDVI" in selected:
        add_unique(variables, "NDVI")
    if "EVI" in selected:
        add_unique(variables, "EVI")
    if "LST" in selected:
        add_unique(collections, "Landsat 8/9 Collection 2 L2")
        add_unique(variables, "Yüzey sıcaklığı (LST)")
    if any(component in selected for component in ("NDVI", "EVI", "LST")):
        add_unique(collections, "ESA WorldCover")
        add_unique(variables, "Arazi örtüsü")

    if any(component in selected for component in ("SPI", "SPEI")):
        start_date = date(1981, 1, 1)
    elif any(component in selected for component in ("NDVI", "EVI")):
        start_date = date(2017, 3, 28)
    else:
        start_date = date(2013, 4, 11)

    return {
        "components": selected,
        "collections": collections,
        "variables": variables,
        "label": " + ".join(collections),
        "start_date": start_date,
    }
