"""Harita katmanlarının statik ve etkileşimli çıktılarda ortak görsel tanımları."""

MAP_VISUAL_STYLES: dict[str, dict[str, object]] = {
    "SPI": {
        "minimum": -3.0,
        "maximum": 3.0,
        "colors": ["#8b1a1a", "#d6604d", "#f4a582", "#f7f7f7", "#92c5de", "#4393c3", "#2166ac"],
        "index": [-3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0],
        "caption": "SPI: aşırı kurak ← normal → aşırı nemli",
        "unit": "Standartlaştırılmış indis",
    },
    "NDVI": {
        "minimum": -1.0,
        "maximum": 1.0,
        "colors": ["#3b6fb6", "#c9b28f", "#f0dc65", "#88c96b", "#187a3d"],
        "index": [-1.0, 0.0, 0.2, 0.4, 0.6, 1.0],
        "caption": "NDVI: su/gölge ← düşük bitki örtüsü → yoğun bitki örtüsü",
        "unit": "NDVI",
    },
    "EVI": {
        "minimum": -1.0,
        "maximum": 1.0,
        "colors": ["#5b4b8a", "#d8c6a3", "#f0dc65", "#78c679", "#006837"],
        "index": [-1.0, 0.0, 0.2, 0.4, 0.6, 1.0],
        "caption": "EVI: düşük ← bitki canlılığı → yüksek",
        "unit": "EVI",
    },
    "LST": {
        "minimum": -10.0,
        "maximum": 50.0,
        "colors": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"],
        "caption": "Arazi yüzey sıcaklığı (°C): düşük → yüksek",
        "unit": "Yüzey sıcaklığı (°C)",
    },
    "Yağış": {
        "minimum": 0.0,
        "maximum": 10.0,
        "colors": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
        "caption": "Ortalama günlük yağış (mm/gün): düşük → yüksek",
        "unit": "Yağış (mm/gün)",
    },
    "Yıllık Yağış": {
        "minimum": 0.0,
        "maximum": 1500.0,
        "colors": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
        "caption": "Son 12 aylık toplam yağış (mm): düşük → yüksek",
        "unit": "12 aylık toplam yağış (mm)",
    },
    "Yağış Normali": {
        "minimum": 40.0,
        "maximum": 160.0,
        "colors": ["#8b1a1a", "#d6604d", "#f7f7f7", "#67a9cf", "#2166ac"],
        "caption": "Yağışın normale oranı (%): kurak ← normal → nemli",
        "unit": "Normal yağışın yüzdesi (%)",
    },
    "Sıcaklık": {
        "minimum": -10.0,
        "maximum": 40.0,
        "colors": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"],
        "caption": "Ortalama hava sıcaklığı (°C): düşük → yüksek",
        "unit": "Hava sıcaklığı (°C)",
    },
    "Sıcaklık Anomalisi": {
        "minimum": -5.0,
        "maximum": 5.0,
        "colors": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"],
        "caption": "Sıcaklık anomalisi (°C): soğuk ← normal → sıcak",
        "unit": "Sıcaklık anomalisi (°C)",
    },
    "Su Dengesi": {
        "minimum": -10.0,
        "maximum": 10.0,
        "colors": ["#8c510a", "#dfc27d", "#f5f5f5", "#80cdc1", "#01665e"],
        "caption": "Klimatik su dengesi P−PET (mm/gün): açık → fazla",
        "unit": "P−PET (mm/gün)",
    },
    "Yıllık Su Dengesi": {
        "minimum": -1200.0,
        "maximum": 1200.0,
        "colors": ["#8c510a", "#dfc27d", "#f5f5f5", "#80cdc1", "#01665e"],
        "caption": "Son 12 aylık P−PET (mm): açık → fazla",
        "unit": "12 aylık P−PET (mm)",
    },
    "Su Dengesi Anomalisi": {
        "minimum": -600.0,
        "maximum": 600.0,
        "colors": ["#8c510a", "#dfc27d", "#f5f5f5", "#80cdc1", "#01665e"],
        "caption": "Klimatik su dengesi anomalisi (mm): açık ← normal → fazla",
        "unit": "P−PET anomalisi (mm)",
    },
}


def map_visual_style(analysis: str) -> dict[str, object] | None:
    return MAP_VISUAL_STYLES.get(str(analysis))
