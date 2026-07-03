"""İzlenebilir metadata ve indirilebilir proje paketi."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone


def build_metadata(**kwargs) -> bytes:
    payload = {
        "application": "Zetriklim",
        "application_version": "0.7.7",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **kwargs,
        "provenance_note": (
            "Platform ile asıl veri ürünü ayrı kaydedilmelidir. Bilimsel atıfta "
            "ürünün üreticisi, sürümü, DOI'si ve erişim tarihi kullanılmalıdır."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def build_project_package(metadata: bytes, area_geojson: str, request_text: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("zetriklim-metadata.json", metadata)
        archive.writestr("calisma-alani.geojson", area_geojson.encode("utf-8"))
        archive.writestr("islem-ozeti.txt", request_text.encode("utf-8"))
        archive.writestr(
            "README.txt",
            (
                "Bu paket Zetriklim tarafından oluşturulmuştur.\n"
                "Canlı veri dosyaları yalnızca seçilen kaynak bağlayıcısı çalıştırıldığında "
                "pakete eklenir. metadata dosyası isteğin bilimsel izini taşır.\n"
            ).encode("utf-8"),
        )
    return output.getvalue()
