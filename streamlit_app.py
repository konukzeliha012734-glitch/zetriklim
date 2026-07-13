# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import date, time, timedelta
import json
import os
from pathlib import Path
import re

import folium
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap, StepColormap
from folium import plugins
from streamlit_folium import st_folium

from zetriklim.catalog import (
    ANALYSES,
    ANALYSIS_METHODS,
    SOURCES,
    VARIABLES,
    academic_data_package,
)
from zetriklim.academic import (
    build_academic_report_html,
    run_academic_analysis,
    safe_monthly_end,
)
from zetriklim.artifacts import (
    build_area_map_png,
    build_complete_package,
    build_excel,
    build_raster_png,
    build_timeseries_png,
    dataframe_to_csv,
    geodata_to_gpkg,
)
from zetriklim.climate_engine import (
    connection_label,
    fetch_map_tile as fetch_climate_engine_map_tile,
    fetch_timeseries as fetch_climate_engine_timeseries,
    validate_api_key,
)
from zetriklim.exports import build_metadata
from zetriklim.gadm import fetch_gadm_boundaries
from zetriklim.geometry import GeometryUploadError, UploadedPart, inspect_uploaded_files
from zetriklim.gee import (
    LAND_COVER_CLASSES,
    build_climate_geotiff,
    build_chirps_spi_geotiff,
    build_remote_analysis_geotiff,
    fetch_chirps_monthly_mean,
    fetch_gee_academic_series,
    fetch_gee_monthly_climate,
    create_user_auth_flow,
    exchange_user_auth_code,
    initialize_gee,
)
from zetriklim.open_meteo import fetch_centroid_series
from zetriklim.spi import calculate_spi_table


ROOT = Path(__file__).parent

st.set_page_config(
    page_title="Zetriklim | Havza ve iklim analizi",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

access_code = os.getenv("ZETRIKLIM_ACCESS_CODE")
if access_code and not st.session_state.get("access_granted"):
    st.title("Zetriklim")
    st.caption("Bu paylaşım GEE kotasını korumak için erişim koduyla sınırlandırılmıştır.")
    entered_code = st.text_input("Erişim kodu", type="password")
    if st.button("Uygulamaya gir", type="primary", use_container_width=True):
        if entered_code == access_code:
            st.session_state.access_granted = True
            st.rerun()
        else:
            st.error("Erişim kodu hatalı.")
    st.stop()


def cached_gee_status(project: str | None) -> tuple[bool, str]:
    return initialize_gee(project)


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_gadm_boundaries(country_iso3: str, level: int):
    return fetch_gadm_boundaries(country_iso3, level)


def checkbox_group(
    label: str,
    options: list,
    *,
    default: list | None = None,
    key_prefix: str,
    help: str | None = None,
    columns: int = 2,
    ui=st,
) -> list:
    ui.markdown(f"**{label}**")
    if help:
        ui.caption(help)
    selected = []
    default_values = set(default or [])
    column_count = max(1, min(columns, len(options)))
    groups = ui.columns(column_count) if column_count > 1 else [ui.container()]
    for index, option in enumerate(options):
        group = groups[index % column_count]
        if group.checkbox(
            str(option),
            value=option in default_values,
            key=f"{key_prefix}_{index}",
        ):
            selected.append(option)
    return selected


def _safe_multiselect_key(label: str, key: str | None) -> str:
    if key:
        return str(key)
    cleaned = re.sub(r"\W+", "_", str(label), flags=re.UNICODE).strip("_").lower()
    return f"safe_multiselect_{cleaned or 'selection'}"


def _safe_streamlit_multiselect(
    label,
    options,
    default=None,
    *,
    key=None,
    help=None,
    **_,
) -> list:
    return checkbox_group(
        str(label),
        list(options),
        default=list(default or []),
        key_prefix=_safe_multiselect_key(str(label), key),
        help=help,
        columns=3,
        ui=st,
    )


def _safe_delta_multiselect(
    self,
    label,
    options,
    default=None,
    *,
    key=None,
    help=None,
    **_,
) -> list:
    return checkbox_group(
        str(label),
        list(options),
        default=list(default or []),
        key_prefix=_safe_multiselect_key(str(label), key),
        help=help,
        columns=2,
        ui=self,
    )


st.multiselect = _safe_streamlit_multiselect
try:
    from streamlit.delta_generator import DeltaGenerator

    DeltaGenerator.multiselect = _safe_delta_multiselect
except Exception:
    pass


def add_cartographic_controls(
    fmap: folium.Map,
    analysis: str,
    source: str,
    start_date,
    end_date,
) -> None:
    """Analiz haritasına CBS lejantı, ölçek, koordinat ve kuzey oku ekler."""
    styles = {
        "NDVI": StepColormap(
            ["#3b6fb6", "#c9b28f", "#f0dc65", "#88c96b", "#187a3d"],
            index=[-1.0, 0.0, 0.2, 0.4, 0.6, 1.0],
            vmin=-1.0,
            vmax=1.0,
            caption="NDVI: su/gölge ← düşük bitki örtüsü → yoğun bitki örtüsü",
        ),
        "EVI": StepColormap(
            ["#5b4b8a", "#d8c6a3", "#f0dc65", "#78c679", "#006837"],
            index=[-1.0, 0.0, 0.2, 0.4, 0.6, 1.0],
            vmin=-1.0,
            vmax=1.0,
            caption="EVI: düşük ← bitki canlılığı → yüksek",
        ),
        "SPI": StepColormap(
            ["#8b1a1a", "#d6604d", "#f4a582", "#f7f7f7", "#92c5de", "#4393c3", "#2166ac"],
            index=[-3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0],
            vmin=-3.0,
            vmax=3.0,
            caption="SPI: aşırı kurak ← normal → aşırı nemli",
        ),
        "LST": LinearColormap(
            ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"],
            vmin=-10,
            vmax=50,
            caption="Arazi yüzey sıcaklığı (°C): düşük → yüksek",
        ),
    }
    colormap = styles.get(analysis)
    if colormap is not None:
        colormap.add_to(fmap)
    plugins.Fullscreen(
        position="topleft",
        title="Tam ekran",
        title_cancel="Tam ekrandan çık",
        force_separate_button=True,
    ).add_to(fmap)
    plugins.MousePosition(
        position="bottomright",
        separator=" · ",
        prefix="Koordinat",
        num_digits=5,
    ).add_to(fmap)
    north_arrow = """
    <div style="position:fixed;right:18px;top:85px;z-index:9999;background:white;
    border:1px solid #315b64;border-radius:8px;padding:5px 8px;text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,.16);font:700 13px sans-serif;color:#063447">
      N<br><span style="font-size:24px;line-height:20px">↑</span>
    </div>
    """
    title_box = f"""
    <div style="position:fixed;left:52px;top:10px;z-index:9999;background:rgba(255,255,255,.94);
    border-left:4px solid #00a6a6;border-radius:6px;padding:7px 11px;
    box-shadow:0 2px 8px rgba(0,0,0,.12);font:12px sans-serif;color:#063447">
      <b>{analysis} mekânsal analiz haritası</b><br>
      Kaynak: {source}<br>Dönem: {start_date} – {end_date}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(north_arrow + title_box))

st.markdown(
    """
    <style>
    :root { --ink:#062f40; --teal:#00a6a6; --cyan:#35c8d0; --amber:#ffad33; --coral:#ef6c57; }
    .stApp {
      background:
        radial-gradient(circle at 92% 8%, rgba(69,214,196,.20), transparent 24rem),
        radial-gradient(circle at 8% 92%, rgba(255,183,77,.15), transparent 22rem),
        linear-gradient(180deg, #f3fbf9 0%, #fffaf0 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1420px; padding-top: 1.25rem; }
    .hero {
      background: linear-gradient(120deg, #102f46 0%, #075b68 52%, #00a68f 100%);
      color: white; border-radius: 28px; padding: 32px 36px; margin-bottom: 18px;
      position: relative; overflow: hidden; box-shadow: 0 18px 50px rgba(6,47,64,.16);
    }
    .hero:before, .hero:after {
      content:""; position:absolute; border-radius:46% 54% 58% 42%;
      border:1px solid rgba(126,241,234,.38); transform:rotate(18deg);
    }
    .hero:before { width:340px; height:340px; right:-70px; top:-150px; }
    .hero:after { width:210px; height:210px; right:40px; bottom:-160px; }
    .eyebrow { color:#ffd27a; font-size:.76rem; letter-spacing:.16em; font-weight:850; }
    .hero h1 { font-size:2.35rem; line-height:1.04; margin:.42rem 0 .6rem; max-width:850px; }
    .hero p { color:#d7f2ef; max-width:820px; margin:0; font-size:1rem; }
    .step {
      font-size:.72rem; letter-spacing:.12em; color:#147984;
      text-transform:uppercase; font-weight:850; margin-bottom:.25rem;
    }
    .hint {
      background:linear-gradient(90deg, rgba(0,166,166,.10), rgba(53,200,208,.05));
      border-left:4px solid var(--teal); padding:12px 14px; border-radius:8px;
      color:#315b64; margin:.5rem 0 1rem;
    }
    .status-ready { color:#07806d; font-weight:800; }
    .status-wait { color:#b36b00; font-weight:800; }
    div[data-testid="stMetric"] {
      background:rgba(255,255,255,.82); border:1px solid rgba(0,128,136,.20);
      padding:12px 14px; border-radius:16px; box-shadow:0 6px 18px rgba(6,47,64,.05);
    }
    button[data-baseweb="tab"] { font-weight:750; }
    div[data-baseweb="tab-list"] {
      background:rgba(255,255,255,.72); padding:.35rem; border-radius:16px;
      border:1px solid rgba(0,128,136,.15);
    }
    .workflow { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:4px 0 20px; }
    .flow-card {
      background:rgba(255,255,255,.82); border:1px solid rgba(0,128,136,.18);
      border-radius:16px; padding:13px 14px; box-shadow:0 7px 20px rgba(6,47,64,.05);
    }
    .flow-card b { color:#075b68; display:block; margin-bottom:3px; }
    .flow-card span { color:#55727a; font-size:.82rem; }
    .brand-name {
      margin-top:.35rem; color:#063f55; font-size:1.02rem; line-height:1;
      font-weight:900; letter-spacing:.14em; text-align:center;
    }
    .brand-subtitle {
      margin-top:.35rem; color:#47717a; font-size:.68rem; line-height:1.25;
      font-weight:700; letter-spacing:.06em; text-align:center;
    }
    .brand-logo-shell {
      width:174px;max-width:100%;margin:0 auto;padding:.35rem;
      background:rgba(255,255,255,.72);border-radius:22px;
      box-shadow:0 9px 26px rgba(6,47,64,.08);
    }
    .brand-logo-shell svg {
      display:block;width:100%;height:auto;
    }
    @media (max-width:800px) { .workflow { grid-template-columns:1fr 1fr; } }
    .stButton > button, .stDownloadButton > button {
      border-radius:999px; min-height:2.85rem; font-weight:780;
    }
    .stButton > button[kind="primary"] {
      background:linear-gradient(90deg, #008f91, #00b7ad); border:0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "uploader_nonce" not in st.session_state:
    st.session_state.uploader_nonce = 0

top_logo, top_hero = st.columns([0.16, 0.84], vertical_alignment="center")
with top_logo:
    st.markdown(
        """
        <div class="brand-logo-shell" role="img" aria-label="Zetriklim coğrafi analiz logosu">
          <svg viewBox="0 0 220 210" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="zg" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#35c8d0"/>
                <stop offset="1" stop-color="#00a68f"/>
              </linearGradient>
              <clipPath id="zb">
                <path d="M48 18 L146 20 L166 57 L199 76 L196 142 L175 174
                         L129 188 L79 181 L31 151 L24 94 Z"/>
              </clipPath>
            </defs>
            <path d="M48 18 L146 20 L166 57 L199 76 L196 142 L175 174
                     L129 188 L79 181 L31 151 L24 94 Z"
                  fill="#fbfdf9" stroke="#063f55" stroke-width="8" stroke-linejoin="round"/>
            <g clip-path="url(#zb)" fill="none" stroke="#35c8d0" stroke-width="4">
              <path d="M8 52 C38 30 51 72 78 48 S124 28 150 45"/>
              <path d="M4 75 C34 52 49 95 78 69 S126 49 164 65"/>
              <path d="M7 101 C38 77 53 120 82 95 S126 75 166 88"/>
              <path d="M12 127 C39 104 56 145 84 121 S128 101 169 112"/>
            </g>
            <g clip-path="url(#zb)" fill="url(#zg)" stroke="#ffffff" stroke-width="2">
              <rect x="84" y="148" width="22" height="22"/><rect x="108" y="148" width="22" height="22"/>
              <rect x="132" y="148" width="22" height="22"/><rect x="156" y="148" width="22" height="22"/>
              <rect x="108" y="124" width="22" height="22"/><rect x="132" y="124" width="22" height="22"/>
              <rect x="156" y="124" width="22" height="22"/><rect x="132" y="100" width="22" height="22"/>
              <rect x="156" y="100" width="22" height="22"/><rect x="156" y="76" width="22" height="22"/>
            </g>
            <circle cx="104" cy="101" r="18" fill="#ffffff" stroke="#063f55" stroke-width="6"/>
            <circle cx="104" cy="101" r="9" fill="#ffb51b"/>
            <path d="M123 101 H151 M104 120 V142" stroke="#063f55" stroke-width="5"
                  stroke-linecap="round" stroke-dasharray="1 10"/>
          </svg>
        </div>
        <div class="brand-name">ZETRİKLİM</div>
        <div class="brand-subtitle">COĞRAFİ ANALİZ PLATFORMU</div>
        """,
        unsafe_allow_html=True,
    )
with top_hero:
    st.markdown(
        """
        <section class="hero">
          <div style="font-size:.76rem;font-weight:800;letter-spacing:.14em;color:#ffd67a">
            HAVZA · İKLİM · UZAKTAN ALGILAMA
          </div>
          <h1>Coğrafi sınırını tanımla, değişimi mekânsal olarak çözümle.</h1>
          <p>Kaynağı ve sürümü belgelenmiş açık veri ürünlerinden seçilen döneme ve analize uygun sonuçları;
          harita, GeoTIFF, Excel ve CBS proje paketi olarak üret.</p>
          <blockquote style="margin:1rem 0 0;padding:.75rem 1rem;border-left:3px solid #ffd67a;
          color:#efffff;background:rgba(255,255,255,.08);border-radius:0 10px 10px 0">
            “Her şey diğer her şeyle ilişkilidir; fakat yakın olanlar uzak olanlardan daha çok ilişkilidir.”
            <span style="display:block;margin-top:.3rem;font-size:.82rem;color:#bfe9e6">
              — Waldo Tobler, Coğrafyanın Birinci Yasası
            </span>
          </blockquote>
        </section>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="workflow">
      <div class="flow-card"><b>1 · Alanı yükle</b><span>SHP, GeoPackage veya GeoJSON</span></div>
      <div class="flow-card"><b>2 · Analizi seç</b><span>SPI, NDVI, EVI veya LST</span></div>
      <div class="flow-card"><b>3 · Veri kaynağını seç</b><span>Climate Engine veya Earth Engine</span></div>
      <div class="flow-card"><b>4 · CBS çıktısını al</b><span>Excel, GeoPackage, GeoTIFF ve harita</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode_col, mode_note = st.columns([0.44, 0.56], vertical_alignment="center")
with mode_col:
    application_mode = st.radio(
        "Çalışma modu",
        ["Standart Analiz", "Akademik Araştırma"],
        horizontal=True,
        help="Akademik mod; hipotez, çoklu indis, doğrulama, belirsizlik ve istatistiksel anlamlılık üretir.",
    )
academic_mode = application_mode == "Akademik Araştırma"
with mode_note:
    st.caption(
        "Akademik mod, mevcut sade iş akışını korur ve aynı çıktı paketine "
        "bilimsel yöntem, doğrulama ve raporlama katmanlarını ekler."
    )

academic_params: dict[str, object] = {}
academic_study: dict[str, object] = {}

tab_area, tab_analysis, tab_data, tab_research, tab_output = st.tabs(
    [
        "01 · Çalışma Alanı",
        "02 · Analiz Seçimi",
        "03 · Veri Kaynağı",
        "04 · Araştırma Tasarımı",
        "05 · Bulgular ve Çıktılar",
    ]
)

with tab_area:
    left, right = st.columns([0.43, 0.57], gap="large")
    with left:
        st.markdown('<div class="step">Çalışma alanı</div>', unsafe_allow_html=True)
        st.subheader("Coğrafi sınırınızı ekleyin")
        st.markdown(
            '<div class="hint">ZIP zorunlu değil. GeoPackage veya GeoJSON tek dosya; '
            'Shapefile ise .shp, .shx, .dbf ve .prj bileşenleri birlikte seçilebilir.</div>',
            unsafe_allow_html=True,
        )
        uploads = st.file_uploader(
            "Çalışma alanı dosyaları",
            type=["zip", "shp", "shx", "dbf", "prj", "cpg", "gpkg", "geojson", "json"],
            accept_multiple_files=True,
            key=f"area_upload_{st.session_state.uploader_nonce}",
            help="Havza, il, ilçe, bölge veya kendi çizdiğiniz poligonları yükleyin.",
        )
        single_shp_crs = st.text_input(
            "Tek SHP için koordinat sistemi",
            value="EPSG:4326",
            help=(
                "Yalnız .shp yüklenirse .prj bulunmadığı için CRS burada belirtilir. "
                "GeoPackage, GeoJSON ve tam SHP paketinde dosyanın kendi CRS bilgisi kullanılır."
            ),
        )
        clear_col, info_col = st.columns([0.42, 0.58], vertical_alignment="center")
        with clear_col:
            if st.button("Dosyaları temizle", icon=":material/delete:", use_container_width=True):
                st.session_state.pop("geometry_summary", None)
                st.session_state.pop("output_package", None)
                st.session_state.uploader_nonce += 1
                st.rerun()
        with info_col:
            st.caption("Hatalı dosyayı yükleme kutusundaki × ile tek tek de silebilirsiniz.")

        with st.expander("Tez haritası ayarları", expanded=academic_mode):
            map_title = st.text_input(
                "Harita başlığı",
                value="Çalışma Alanının Konumu",
                help="PNG alan haritasında kullanılacak şekil başlığıdır.",
            )
            map_use_gadm = st.checkbox(
                "GADM idari sınırlarını bağlam katmanı olarak ekle",
                value=True,
                help="GADM verisi yalnız harita çiziminde kullanılır; ham veri pakete eklenmez.",
            )
            g1, g2 = st.columns(2)
            gadm_iso3 = g1.text_input(
                "GADM ülke kodu (ISO3)",
                value="TUR",
                max_chars=3,
                help="Türkiye için TUR; başka ülkelerde üç harfli ISO3 kodunu girin.",
            ).strip().upper()
            gadm_level = int(
                g2.selectbox(
                    "İdari düzey",
                    [0, 1, 2],
                    index=1,
                    format_func=lambda value: {
                        0: "0 · Ülke sınırı",
                        1: "1 · Birinci düzey (Türkiye: il)",
                        2: "2 · İkinci düzey (Türkiye: ilçe)",
                    }[value],
                )
            )
            st.caption(
                "GADM 4.1 akademik ve ticari olmayan kullanım koşullarıyla kullanılır; "
                "haritada sürüm ve kaynak bilgisi gösterilir."
            )

        summary = st.session_state.get("geometry_summary")
        if uploads:
            try:
                parts = [UploadedPart(item.name, item.getvalue()) for item in uploads]
                with st.spinner("Geometri, koordinat sistemi ve alan denetleniyor..."):
                    summary = inspect_uploaded_files(parts, fallback_crs=single_shp_crs)
                st.session_state.geometry_summary = summary
            except GeometryUploadError as exc:
                summary = None
                st.session_state.pop("geometry_summary", None)
                st.error(str(exc))

        if summary:
            m1, m2 = st.columns(2)
            m1.metric("Toplam alan", f"{summary.area_km2:,.2f} km²")
            m2.metric("Coğrafi obje", f"{summary.feature_count:,}")
            st.success("Çalışma alanı doğrulandı.")

            with st.expander("Alan detayları", expanded=True):
                d1, d2 = st.columns(2)
                d1.write(f"**Çevre:** {summary.perimeter_km:,.2f} km")
                d1.write(f"**Köşe sayısı:** {summary.vertex_count:,}")
                d1.write(f"**Kaynak CRS:** {summary.source_crs}")
                d2.write(f"**Alan hesabı CRS:** {summary.area_crs}")
                d2.write(f"**Merkez:** {summary.centroid[0]:.5f}, {summary.centroid[1]:.5f}")
                d2.write(f"**Geometri onarımı:** {'Uygulandı' if summary.was_repaired else 'Gerekmedi'}")

            spatial_mode = st.selectbox(
                "Analiz alanı detayı",
                [
                    "Tüm objeleri tek çalışma alanı olarak birleştir",
                    "Her coğrafi objeyi ayrı raporla",
                    "Her obje + tüm alan özetini birlikte üret",
                    "Yalnızca alan ortalaması üret",
                ],
                help="Çok parçalı ilçe veya alt havza verilerinde sonuçların nasıl gruplanacağını belirler.",
            )
            spatial_stat = st.multiselect(
                "Mekânsal özetler",
                ["Ortalama", "Toplam", "Minimum", "Maksimum", "Medyan", "Standart sapma", "Yüzdelikler"],
                default=["Ortalama", "Minimum", "Maksimum"],
                help="Çalışma alanındaki raster hücrelerinin hangi istatistiklerle özetleneceğini belirler.",
            )
        else:
            spatial_mode = "Tüm objeleri tek çalışma alanı olarak birleştir"
            spatial_stat = ["Ortalama"]
            st.info("Harita ve alan ayrıntıları için bir coğrafi dosya yükleyin.")

    with right:
        st.markdown('<div class="step">Harita önizleme</div>', unsafe_allow_html=True)
        if summary:
            bounds = summary.bounds
            fmap = folium.Map(summary.centroid, zoom_start=8, tiles="CartoDB positron")
            folium.GeoJson(
                summary.gdf_wgs84.__geo_interface__,
                style_function=lambda _: {
                    "color": "#063447", "weight": 3, "fillColor": "#00a6a6", "fillOpacity": 0.30
                },
                highlight_function=lambda _: {"weight": 5, "fillOpacity": 0.42},
                tooltip=folium.GeoJsonTooltip(
                    fields=[c for c in summary.gdf_wgs84.columns if c != "geometry"][:4],
                    aliases=[c for c in summary.gdf_wgs84.columns if c != "geometry"][:4],
                ) if len(summary.gdf_wgs84.columns) > 1 else None,
            ).add_to(fmap)
            fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        else:
            fmap = folium.Map([39.0, 35.0], zoom_start=5, tiles="CartoDB positron")
        st_folium(fmap, height=570, width="stretch", returned_objects=[])

academic_components_for_source = st.session_state.get(
    "academic_analysis_components",
    ["SPI", "SPEI", "NDVI", "EVI", "LST"],
)
automatic_academic_package = academic_data_package(academic_components_for_source)
analysis_for_source = (
    (
        "SPI"
        if automatic_academic_package["components"][0] == "SPEI"
        else automatic_academic_package["components"][0]
    )
    if academic_mode
    else st.session_state.get("selected_analysis_widget", "SPI")
)
source_context_label = "Akademik çoklu analiz" if academic_mode else analysis_for_source

with tab_data:
    st.markdown('<div class="step">Kaynak, ürün ve değişken</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hint"><strong>{source_context_label}</strong> için yalnızca uyumlu kaynak ve '
        + (
            'ürünler listelenir. Çok kaynaklı akademik analiz için Earth Engine önerilen seçenektir.</div>'
            if academic_mode
            else 'ürünler listelenir. Climate Engine varsayılandır; Earth Engine alternatif olarak seçilebilir.</div>'
        ),
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="large")
    climate_engine_key = st.session_state.get(
        "climate_engine_api_key",
        os.getenv("CLIMATE_ENGINE_API_KEY", ""),
    )
    with c1:
        source_options = (
            ["Google Earth Engine", "Climate Engine"]
            if academic_mode
            else ["Climate Engine", "Google Earth Engine"]
        )
        provider = st.selectbox(
            "Veri kaynağı",
            source_options,
            index=0,
            help=(
                "Climate Engine kişisel API anahtarıyla, Google Earth Engine ise "
                "Project ID ve Google hesabı yetkilendirmesiyle çalışır."
            ),
        )
        source_info = SOURCES[provider]
        st.caption(source_info["description"])
        if academic_mode and provider == "Climate Engine":
            st.warning(
                "Climate Engine seçildiğinde SPI/SPEI, eğilim ve kaynak karşılaştırması üretilir; "
                "aylık NDVI/EVI/LST gecikme ve arazi örtüsü tabakaları için Earth Engine gerekir."
            )
        status_class = "status-ready" if source_info["status"] in {"Hazır", "Açık erişim"} else "status-wait"
        st.markdown(f'<span class="{status_class}">● {source_info["status"]}</span>', unsafe_allow_html=True)
        if provider == "Climate Engine":
            st.markdown("##### Climate Engine hesabını bağlayın")
            st.caption(
                "Climate Engine kullanıcı girişi Project ID ile değil, kişiye özel API anahtarıyla yapılır."
            )
            entered_ce_key = st.text_input(
                "Climate Engine API anahtarı",
                value="",
                type="password",
                placeholder="Anahtarınızı buraya yapıştırın",
                help="Anahtar yalnız bu tarayıcı oturumunda tutulur; GitHub'a ve çıktı dosyalarına yazılmaz.",
            )
            ce_left, ce_right = st.columns(2)
            ce_left.link_button(
                "API anahtarı iste",
                "https://www.climateengine.org/apis/requesting-an-authorization-key-token/",
                use_container_width=True,
            )
            ce_right.link_button(
                "Resmî API belgeleri",
                "https://api.climateengine.org/",
                use_container_width=True,
            )
            if st.button(
                "Climate Engine bağlantısını doğrula",
                disabled=not entered_ce_key,
                use_container_width=True,
            ):
                try:
                    with st.spinner("Climate Engine anahtarı doğrulanıyor..."):
                        ce_status = validate_api_key(entered_ce_key)
                    st.session_state.climate_engine_api_key = entered_ce_key
                    st.session_state.climate_engine_status = ce_status
                    st.success("Climate Engine API anahtarı doğrulandı.")
                    st.rerun()
                except Exception as ce_error:
                    st.session_state.pop("climate_engine_api_key", None)
                    st.session_state.pop("climate_engine_status", None)
                    st.error(f"Climate Engine bağlantısı kurulamadı: {ce_error}")
            climate_engine_key = st.session_state.get(
                "climate_engine_api_key",
                os.getenv("CLIMATE_ENGINE_API_KEY", ""),
            )
            if climate_engine_key:
                st.success(f"Climate Engine · {connection_label(climate_engine_key)}")
                expiration = st.session_state.get("climate_engine_status", {}).get("expiration")
                if expiration:
                    st.caption(f"Anahtar geçerlilik bilgisi: {expiration}")
                if st.button("Climate Engine bağlantısını kes", use_container_width=True):
                    st.session_state.pop("climate_engine_api_key", None)
                    st.session_state.pop("climate_engine_status", None)
                    st.rerun()
        gee_project = (
            st.text_input(
                "Google Earth Engine Project ID",
                value=os.getenv("GOOGLE_EARTH_ENGINE_PROJECT", ""),
                placeholder="ör. ee-kullanici-projesi",
                help="Proje adı veya numarası değil, Google Cloud Project ID değerini girin.",
            )
            if provider == "Google Earth Engine"
            else os.getenv("GOOGLE_EARTH_ENGINE_PROJECT", "")
        )
        if provider == "Google Earth Engine":
            project_valid = bool(
                re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", gee_project or "")
            )
            with st.expander("Project ID nasıl alınır?", expanded=not project_valid):
                st.markdown(
                    """
                    1. Google Cloud'da bir proje oluşturun veya mevcut projenizi seçin.
                    2. **Project ID** değerini proje seçiciden kopyalayın; proje adı ve proje numarası farklıdır.
                    3. Earth Engine API'yi etkinleştirin.
                    4. Projeyi ticari olmayan araştırma veya uygun kullanım türüyle Earth Engine'e kaydedin.
                    5. Buraya Project ID'yi girip Google hesabınızla yetkilendirin.
                    """
                )
                st.link_button(
                    "1 · Google Cloud projesi oluştur",
                    "https://console.cloud.google.com/projectcreate",
                    use_container_width=True,
                )
                if project_valid:
                    st.link_button(
                        "2 · Earth Engine API'yi etkinleştir",
                        "https://console.cloud.google.com/apis/library/earthengine.googleapis.com"
                        f"?project={gee_project}",
                        use_container_width=True,
                    )
                    st.link_button(
                        "3 · Projeyi Earth Engine'e kaydet",
                        "https://console.cloud.google.com/earth-engine/configuration"
                        f"?project={gee_project}",
                        use_container_width=True,
                    )
                    st.caption(
                        "Kayıt sayfasına yönlendirilmeniz normaldir; bu adım Google girişi değil, "
                        "Project ID'nin Earth Engine kullanımına açılmasıdır."
                    )

            if gee_project and not project_valid:
                st.error(
                    "Project ID biçimi geçerli görünmüyor. Yalnızca küçük harf, rakam ve kısa çizgi kullanın."
                )
                gee_ok, gee_message = False, "Geçersiz Project ID"
            elif project_valid:
                gee_ok, gee_message = cached_gee_status(gee_project)
                personal_auth = st.session_state.get("gee_user_auth", {})
                personally_connected = personal_auth.get("project") == gee_project
                if gee_ok and personally_connected:
                    st.success(f"Earth Engine {gee_message}")
                    if st.button("Google bağlantısını kes", use_container_width=True):
                        st.session_state.pop("gee_user_auth", None)
                        st.session_state.pop("gee_auth_flow", None)
                        st.rerun()
                elif gee_ok:
                    st.success(f"Earth Engine {gee_message}")
                    st.caption("Sunucu bağlantısı hazır. İsterseniz kendi Google hesabınızı bağlayabilirsiniz.")
                else:
                    st.warning("Bu Project ID için geçerli Google yetkilendirmesi bulunamadı.")

                if not personally_connected:
                    if st.button(
                        "Google hesabıyla Earth Engine'e bağlan",
                        type="primary",
                        use_container_width=True,
                    ):
                        auth_url, verifier = create_user_auth_flow()
                        st.session_state.gee_auth_flow = {
                            "project": gee_project,
                            "url": auth_url,
                            "verifier": verifier,
                        }
                    auth_flow = st.session_state.get("gee_auth_flow")
                    if auth_flow and auth_flow.get("project") == gee_project:
                        st.link_button(
                            "Google giriş ve izin ekranını aç",
                            auth_flow["url"],
                            use_container_width=True,
                        )
                        auth_code = st.text_input(
                            "Google'ın verdiği tek kullanımlık doğrulama kodu",
                            type="password",
                            help="Kod yalnızca bu oturumda bağlantı kurmak için kullanılır ve dosyaya yazılmaz.",
                        )
                        if st.button(
                            "Bağlantıyı tamamla ve test et",
                            disabled=not auth_code,
                            use_container_width=True,
                        ):
                            try:
                                with st.spinner("Google Earth Engine bağlantısı doğrulanıyor..."):
                                    st.session_state.gee_user_auth = exchange_user_auth_code(
                                        auth_code,
                                        auth_flow["verifier"],
                                        gee_project,
                                    )
                                st.session_state.pop("gee_auth_flow", None)
                                st.success("Google hesabı ve Project ID başarıyla doğrulandı.")
                                st.rerun()
                            except Exception as auth_error:
                                st.error(
                                    "Bağlantı kurulamadı. Project ID, Earth Engine kaydı ve Google "
                                    f"hesabı izinlerini kontrol edin. Ayrıntı: {auth_error}"
                                )
            else:
                gee_ok, gee_message = False, "Project ID bekleniyor"
                st.info("Önce kendi Google Cloud Project ID değerinizi girin.")
    with c2:
        ce_products = {
            "SPI": {
                "CHIRPS Daily (4,8 km)": ("CHIRPS_DAILY", ["precipitation"], date(1981, 1, 1)),
                "CHIRPS Pentad (4,8 km)": ("CHIRPS_PENTAD", ["precipitation"], date(1981, 1, 1)),
                "CHIRPS Preliminary Pentad": ("CHIRPS_PRELIM_PENTAD", ["precipitation"], date(2015, 1, 1)),
                "ERA5-Ag Daily (9,6 km)": ("ERA5_AG", ["total_precipitation"], date(1979, 1, 1)),
            },
            "NDVI": {
                "Sentinel-2 Surface Reflectance (10 m)": ("SENTINEL2_SR", ["NDVI"], date(2015, 1, 1)),
                "Harmonized Landsat–Sentinel-2 (30 m)": ("HLS_SR", ["NDVI"], date(2013, 4, 11)),
                "Landsat 5/7/8/9 Surface Reflectance (30 m)": ("LANDSAT_SR", ["NDVI"], date(1984, 1, 1)),
            },
            "EVI": {
                "Sentinel-2 Surface Reflectance (10 m)": ("SENTINEL2_SR", ["EVI"], date(2015, 1, 1)),
                "Harmonized Landsat–Sentinel-2 (30 m)": ("HLS_SR", ["EVI"], date(2013, 4, 11)),
                "Landsat 5/7/8/9 Surface Reflectance (30 m)": ("LANDSAT_SR", ["EVI"], date(1984, 1, 1)),
            },
            "LST": {
                "Landsat 8 Surface Reflectance (30 m)": ("LANDSAT8_SR", ["LST"], date(2013, 4, 11)),
                "Landsat 5/7/8/9 Surface Reflectance (30 m)": ("LANDSAT_SR", ["LST"], date(1984, 1, 1)),
                "MODIS Terra 8-day (1 km)": ("MODIS_TERRA_8DAY", ["LST_Day_1km"], date(2000, 2, 18)),
            },
        }
        gee_products = {
            "SPI": ["CHIRPS Daily"],
            "NDVI": ["Sentinel-2 SR Harmonized"],
            "EVI": ["Sentinel-2 SR Harmonized"],
            "LST": ["Landsat 8/9 Collection 2 Level-2"],
        }
        if provider == "Climate Engine":
            product = st.selectbox(
                "Veri ürünü",
                list(ce_products[analysis_for_source]),
                help=f"Yalnızca {analysis_for_source} üretebilen belgelenmiş Climate Engine ürünleri gösterilir.",
            )
            dataset_options = {
                value[0]: value[1]
                for value in ce_products[analysis_for_source].values()
            }
            selected_dataset = ce_products[analysis_for_source][product][0]
            product_start_date = ce_products[analysis_for_source][product][2]
            ce_dataset_id = st.selectbox(
                "Dataset parametresi",
                list(dataset_options),
                index=list(dataset_options).index(selected_dataset),
                help="Climate Engine API'nin kullandığı resmî dataset kodudur.",
            )
            product_start_date = {
                value[0]: value[2]
                for value in ce_products[analysis_for_source].values()
            }[ce_dataset_id]
            ce_variable_id = st.selectbox(
                "Değişken parametresi",
                dataset_options[ce_dataset_id],
                help=f"{analysis_for_source} hesabı için ürün içinde kullanılacak resmî değişken kodudur.",
            )
            ce_variable_ids = ce_variable_id
            st.link_button(
                "Dataset ve değişken parametrelerini incele",
                "https://www.climateengine.org/apis/apiDatasets/",
                use_container_width=True,
            )
        elif academic_mode:
            product = str(automatic_academic_package["label"])
            st.markdown("**Otomatik akademik veri paketi**")
            st.info(product, icon="📦")
            st.caption(
                "02 · Analiz Seçimi bölümündeki tercihinize göre gerekli veri "
                "koleksiyonları otomatik atanmıştır; ayrıca ürün seçmeniz gerekmez."
            )
            ce_dataset_id, ce_variable_ids = "", ""
            product_start_date = automatic_academic_package["start_date"]
        else:
            product = st.selectbox(
                "Veri ürünü",
                gee_products[analysis_for_source],
                help=f"{analysis_for_source} için belgelenmiş Earth Engine koleksiyonu.",
            )
            ce_dataset_id, ce_variable_ids = "", ""
            product_start_date = {
                "SPI": date(1981, 1, 1),
                "NDVI": date(2017, 3, 28),
                "EVI": date(2017, 3, 28),
                "LST": date(2013, 4, 11),
            }[analysis_for_source]
        variables = (
            list(automatic_academic_package["variables"])
            if academic_mode
            else {
                "SPI": ["Yağış"],
                "NDVI": ["NDVI / EVI"],
                "EVI": ["NDVI / EVI"],
                "LST": ["Yüzey sıcaklığı (LST)"],
            }[analysis_for_source]
        )
        quality_control = [
            "Eksik veri",
            "Birim dönüşümü",
            "Zaman sürekliliği",
            "Tamamlanmamış ay",
            "Fiziksel değer aralığı",
            "Uydu geçerli piksel oranı",
        ]
        st.info(
            f"Analiz değişkeni otomatik seçildi: {variables[0]}. "
            "Eksik veri, birim ve zaman sürekliliği kontrolleri uygulanacaktır."
        )
    with c3:
        period_mode = st.radio(
            "Dönem tanımlama",
            ["Yıl aralığı", "Kesin tarih aralığı"],
            horizontal=True,
            help="Yıl aralığında başlangıç 1 Ocak, bitiş 31 Aralık olarak uygulanır.",
        )
        if period_mode == "Yıl aralığı":
            year_left, year_right = st.columns(2)
            start_year = year_left.number_input(
                "Başlangıç yılı",
                min_value=product_start_date.year,
                max_value=date.today().year,
                value=product_start_date.year,
                step=1,
                help=f"Seçili ürünün belgelenmiş veri başlangıcı: {product_start_date:%d.%m.%Y}.",
                key=f"start_year_{provider}_{analysis_for_source}_{ce_dataset_id or product}",
            )
            end_year = year_right.number_input(
                "Bitiş yılı",
                min_value=product_start_date.year,
                max_value=date.today().year,
                value=date.today().year,
                step=1,
                help="Analize dahil edilecek son takvim yılı; mevcut yıl seçilirse bugün sona erer.",
                key=f"end_year_{provider}_{analysis_for_source}_{ce_dataset_id or product}",
            )
            start_date = max(date(int(start_year), 1, 1), product_start_date)
            end_date = (
                date.today()
                if int(end_year) == date.today().year
                else date(int(end_year), 12, 31)
            )
            st.caption(f"Uygulanacak dönem: {start_date:%d.%m.%Y} – {end_date:%d.%m.%Y}")
        else:
            start_date = st.date_input(
                "Başlangıç tarihi", product_start_date,
                min_value=product_start_date,
                max_value=date.today(),
                help=f"Seçili ürünün belgelenmiş veri başlangıcı: {product_start_date:%d.%m.%Y}.",
                key=f"start_date_{provider}_{analysis_for_source}_{ce_dataset_id or product}",
            )
            end_date = st.date_input(
                "Bitiş tarihi", date.today(),
                min_value=product_start_date,
                max_value=date.today(),
                help="Veri sorgusunun sona ereceği günü seçin.",
                key=f"end_date_{provider}_{analysis_for_source}_{ce_dataset_id or product}",
            )
        incomplete_month_removed = False
        if academic_mode or analysis_for_source == "SPI":
            end_date, incomplete_month_removed = safe_monthly_end(end_date)
            if incomplete_month_removed:
                st.warning(
                    f"Aylık toplamların yapay düşük görünmemesi için tamamlanmamış ay çıkarıldı. "
                    f"Son tarih {end_date:%d.%m.%Y} olarak uygulandı."
                )
        if provider == "Google Earth Engine" and analysis_for_source == "SPI":
            previous_month_end = date.today().replace(day=1) - timedelta(days=1)
            chirps_final_boundary = previous_month_end.replace(day=1) - timedelta(days=1)
            if end_date > chirps_final_boundary:
                end_date = chirps_final_boundary
                st.warning(
                    "CHIRPS v2.0 Final arşiv gecikmesi nedeniyle yalnız tamamlanmış ve "
                    f"yayımlanmış aylar kullanıldı. Son tarih {end_date:%d.%m.%Y}."
                )
        period_compatible = start_date >= product_start_date and start_date <= end_date
        st.caption(
            f"{product} kullanılabilir dönem başlangıcı: {product_start_date:%d.%m.%Y}. "
            "Bu tarihten önceki bir dönem için daha eski arşive sahip başka bir ürün seçilmelidir."
        )
        temporal_scale = "Aylık" if academic_mode or analysis_for_source == "SPI" else "Dönem kompoziti"
        aggregation = "Değişkene uygun aylık özet" if academic_mode else (
            "Toplam" if analysis_for_source == "SPI" else "Medyan"
        )
        start_time, end_time = None, None
        st.info(
            f"Zamansal işlem otomatik belirlendi: {temporal_scale} · {aggregation}.",
            icon="ℹ️",
        )
        if start_date > end_date:
            st.error("Başlangıç yılı/tarihi bitiş değerinden sonra olamaz.")
        if start_date < product_start_date:
            st.error(
                f"{product}, {product_start_date:%d.%m.%Y} öncesinde veri içermez. "
                "İstek gönderilmedi; ürün veya başlangıç tarihini değiştirin."
            )

    st.success(
        f"Uyumlu seçim: {analysis_for_source} · {provider} · {product}. "
        "Dataset ve değişken kodları analizle birlikte metadata dosyasına kaydedilecektir."
    )

with tab_analysis:
    st.markdown('<div class="step">Uygulanacak analizi belirleyin</div>', unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="hint">Akademik mod, kuraklık indislerini uydu tepkisi, eğilim, '
            'doğrulama ve belirsizlik analizleriyle aynı araştırma tasarımında birleştirir.</div>'
            if academic_mode
            else '<div class="hint">Her işlemde tek bir ana analiz seçilir. Böylece yalnızca o yönteme '
            'ait parametreler, uygun veri ürünleri ve çıktılar gösterilir.</div>'
        ),
        unsafe_allow_html=True,
    )
    if academic_mode:
        selected_analyses = st.multiselect(
            "Akademik analiz bileşenleri",
            ["SPI", "SPEI", "NDVI", "EVI", "LST"],
            default=["SPI", "SPEI", "NDVI", "EVI", "LST"],
            key="academic_analysis_components",
            help=(
                "SPI zorunlu değildir. Yalnız SPEI, yalnız NDVI/EVI/LST veya bunların "
                "istediğiniz birleşimini seçebilirsiniz."
            ),
        )
        selected_analysis = selected_analyses[0] if selected_analyses else "SPEI"
        if selected_analyses:
            st.success(
                "Seçilen bağımsız analiz bileşenleri: " + " + ".join(selected_analyses)
                + ". Eğilim analizi her seçili değişken için ayrı yürütülür."
            )
        else:
            st.error("En az bir akademik analiz bileşeni seçin.")
    else:
        selected_analysis = st.selectbox(
            "Uygulanacak analiz",
            ["SPI", "NDVI", "EVI", "LST"],
            key="selected_analysis_widget",
            help=(
                "SPI meteorolojik kuraklığı; NDVI ve EVI bitki örtüsü durumunu; "
                "LST arazi yüzey sıcaklığını inceler."
            ),
        )
        selected_analyses = [selected_analysis]
    method_rows = []
    for method in selected_analyses:
        method_info = ANALYSIS_METHODS[method]
        method_rows.append({
            "Yöntem": method,
            "Tam ad": method_info["title"],
            "Önerilen ürün": method_info["source"],
            "Doğal çözünürlük": method_info["resolution"],
            "Amaç": method_info["purpose"],
        })
    st.dataframe(
        method_rows,
        hide_index=True,
        width="stretch",
    )
    analysis_params = {"method": " + ".join(selected_analyses)}

    if selected_analysis == "SPI" and not academic_mode:
        st.markdown("##### SPI hesaplama ayarları")
        p1, p2, p3 = st.columns(3)
        analysis_params["scales"] = p1.multiselect(
            "SPI zaman ölçeği (ay)",
            [1, 3, 6, 9, 12, 18, 24],
            default=[3, 6, 12],
            help=(
                "Yağışın kaç aylık birikim üzerinden değerlendirileceğini belirler. "
                "SPI-3 mevsimsel, SPI-6 orta dönem, SPI-12 uzun dönem kuraklığı gösterir."
            ),
        )
        analysis_params["distribution"] = p2.selectbox(
            "Olasılık dağılımı",
            ["Gamma"],
            help="Yağış serisi için sıfır olasılığı düzeltilmiş Gamma dağılımı uygulanır.",
        )
        analysis_params["baseline"] = p3.selectbox(
            "Referans dönemi",
            ["1981–2024", "1991–2020", "1981–2010"],
            help="SPI değerlerinin karşılaştırıldığı klimatolojik dönemdir.",
        )
    elif not academic_mode:
        st.markdown(f"##### {selected_analysis} görüntü işleme ayarları")
        p1, p2 = st.columns(2)
        analysis_params["cloud_limit"] = p1.slider(
            "Azami sahne bulutluluğu (%)",
            0, 80, 30, 5,
            help="Bu orandan daha bulutlu uydu sahneleri analize alınmaz.",
        )
        analysis_params["composite"] = p2.selectbox(
            "Dönem kompoziti",
            ["Medyan"],
            help="Seçilen dönemdeki geçerli piksellerin medyanı alınarak tek raster üretilir.",
        )
        st.info(
            {
                "NDVI": "−1 ile +1 arasındadır; yüksek pozitif değerler daha yoğun ve sağlıklı bitki örtüsünü gösterir.",
                "EVI": "Yoğun bitki örtüsünde atmosfer ve toprak etkisini NDVI'ya göre daha güçlü düzeltir.",
                "LST": "Uydu termal bantlarından hesaplanan arazi yüzey sıcaklığıdır; hava sıcaklığı değildir.",
            }[selected_analysis]
        )

with tab_research:
    if not academic_mode:
        st.info(
            "Bu bölüm Akademik Araştırma modu seçildiğinde açılır. Standart moddaki "
            "mevcut SPI, NDVI, EVI ve LST iş akışı değişmeden kullanılabilir."
        )
        academic_params = {
            "scales": analysis_params.get("scales", [3, 6, 12]),
            "drought_indices": ["SPI"],
            "baseline_start": 1981,
            "baseline_end": 2024,
        }
    else:
        selected_drought_components = [
            item for item in selected_analyses if item in {"SPI", "SPEI"}
        ]
        selected_response_components = [
            item for item in selected_analyses if item in {"NDVI", "EVI", "LST"}
        ]
        has_drought_analysis = bool(selected_drought_components)
        has_response_analysis = bool(selected_response_components)
        research_profile_key = "_".join(selected_analyses) or "academic"

        if has_drought_analysis and has_response_analysis:
            default_title = "Havza ölçeğinde çok kaynaklı kuraklık ve ekosistem tepkisi analizi"
            default_question = (
                "Kuraklığın seçilen ekosistem değişkenleri üzerindeki etkisi hangi zaman "
                "ölçeğinde ve kaç aylık gecikmeyle ortaya çıkmaktadır?"
            )
            default_hypotheses = (
                "H1: Kuraklık indisleri ile ekosistem tepki değişkenleri arasında anlamlı ilişki vardır.\n"
                "H2: Ekosistem tepkisi kuraklıktan sonra gecikmeli ortaya çıkar.\n"
                "H3: Arazi örtüsü sınıfları kuraklık tepkisinin büyüklüğünü değiştirir."
            )
        elif has_drought_analysis:
            default_title = "Havza ölçeğinde kuraklık indisi ve eğilim analizi"
            default_question = (
                "Seçilen kuraklık indisleri çalışma döneminde hangi sıklık, şiddet ve "
                "uzun dönemli eğilim özelliklerini göstermektedir?"
            )
            default_hypotheses = (
                "H1: Çalışma döneminde kuraklık şiddeti veya sıklığında anlamlı değişim vardır.\n"
                "H2: Kuraklık özellikleri seçilen zaman ölçeklerine göre farklılaşır."
            )
        else:
            response_names = ", ".join(selected_response_components)
            default_title = f"Havza ölçeğinde {response_names} eğilim ve mekânsal değişim analizi"
            default_question = (
                f"{response_names} değerleri çalışma döneminde zamansal eğilim ve arazi "
                "örtüsü sınıflarına göre mekânsal farklılık göstermekte midir?"
            )
            default_hypotheses = (
                f"H1: {response_names} değerlerinde istatistiksel olarak anlamlı zamansal eğilim vardır.\n"
                f"H2: {response_names} değerleri arazi örtüsü sınıfları arasında anlamlı biçimde farklılaşır."
            )

        st.markdown('<div class="step">Araştırma sorusu ve hipotezler</div>', unsafe_allow_html=True)
        st.caption("Her alanın yanındaki ? simgesine tıklayarak kısa açıklamasını görebilirsiniz.")
        academic_study["title"] = st.text_input(
            "Çalışma başlığı",
            value=default_title,
            help="Tezde, raporda ve çıktı metadata dosyasında görünecek araştırma başlığıdır.",
            key=f"academic_title_{research_profile_key}",
        )
        academic_study["question"] = st.text_area(
            "Araştırma sorusu",
            value=default_question,
            help="Çalışmanın tek ve ölçülebilir ana sorusudur; hangi değişkenler arasındaki ilişkinin araştırıldığını belirtir.",
            key=f"academic_question_{research_profile_key}",
        )
        academic_study["hypotheses"] = st.text_area(
            "Hipotezler (her satıra bir hipotez)",
            value=default_hypotheses,
            height=145,
            help="Verilerle sınanacak bilimsel beklentilerdir. Her hipotezi H1, H2 şeklinde ayrı satıra yazın.",
            key=f"academic_hypotheses_{research_profile_key}",
        )

        drought_indices = selected_drought_components
        scales = [3]
        baseline_start, baseline_end = 1991, 2020
        spi_distribution, spei_distribution = "Gamma", "Log-logistic"
        event_threshold = -1.0

        if has_drought_analysis:
            st.markdown("##### Kuraklık indisleri ve referans dönemi")
            st.info(
                "Analiz Seçimi bölümünden otomatik aktarıldı: "
                + " + ".join(drought_indices)
            )
            r1, r2, r3 = st.columns(3)
            scales = r1.multiselect(
                "Zaman ölçeği (ay)", [1, 3, 6, 9, 12, 18, 24], default=[1, 3, 6, 12, 24],
                help="Kuraklığın kaç aylık birikimli koşullarla hesaplanacağını belirler. Kısa ölçekler hızlı, uzun ölçekler kalıcı etkileri gösterir.",
            )
            baseline_start = int(r2.number_input(
                "Referans başlangıcı", 1981, date.today().year - 9, 1991,
                help="SPI/SPEI değerlerinin normal koşullara göre standartlaştırılacağı klimatolojik dönemin ilk yılıdır.",
            ))
            baseline_end = int(r3.number_input(
                "Referans bitişi", baseline_start + 9, date.today().year, 2020,
                help="Klimatolojik referans döneminin son yılıdır. Güvenilir sonuç için tercihen en az 30 yıllık dönem seçilir.",
            ))
            if baseline_end - baseline_start + 1 < 30:
                st.warning("Kararlı standartlaştırma için en az 30 yıllık referans dönemi önerilir.")

            distribution_columns = st.columns(len(drought_indices) + 1)
            distribution_position = 0
            if "SPI" in drought_indices:
                spi_distribution = distribution_columns[distribution_position].selectbox(
                    "SPI dağılımı", ["Gamma", "Pearson Tip III"],
                    help="Yağış olasılıklarının SPI değerine dönüştürülmesinde kullanılacak istatistiksel dağılımdır; Gamma yaygın varsayılandır.",
                )
                distribution_position += 1
            if "SPEI" in drought_indices:
                spei_distribution = distribution_columns[distribution_position].selectbox(
                    "SPEI dağılımı", ["Log-logistic", "Pearson Tip III"],
                    help="Yağış eksi potansiyel evapotranspirasyon su dengesinin standartlaştırılmasında kullanılacak dağılımdır.",
                )
                distribution_position += 1
            event_threshold = distribution_columns[distribution_position].selectbox(
                "Kuraklık olayı eşiği", [-0.5, -1.0, -1.5, -2.0], index=1,
                help="İndis bu değerin altına düştüğünde dönem kuraklık olayı sayılır. −1,0 orta; −1,5 şiddetli; −2,0 olağanüstü kuraklığı temsil eder.",
            )
            if "SPEI" in drought_indices:
                st.caption(
                    "SPEI su dengesi, GEE iş akışında ERA5-Land model potansiyel buharlaşmasıyla "
                    "hesaplanır; bu değişken FAO-56 referans evapotranspirasyonu (ET₀) ile aynı değildir. "
                    "Kullanılan ürün ve dönüşüm adımları raporun yöntem/metadata bölümüne yazılır."
                )
        else:
            st.info(
                "Yalnız " + " + ".join(selected_response_components)
                + " seçildiği için SPI/SPEI, kuraklık eşiği ve referans dönemi ayarları uygulanmaz."
            )

        st.markdown("##### Eğilim ve mekânsal tabakalama")
        trend_column_count = 4 if has_drought_analysis and has_response_analysis else 3
        trend_columns = st.columns(trend_column_count)
        a1, a2, a3 = trend_columns[:3]
        prewhiten = a1.checkbox(
            "Otokorelasyon düzeltmesi", value=True,
            help="Ardışık ayların birbirine benzemesinden kaynaklanabilecek yapay eğilim anlamlılığını azaltır.",
        )
        seasonal_mk = a2.checkbox(
            "Mevsimsel Mann–Kendall", value=True,
            help="İlkbahar-yaz gibi doğal mevsim farklarını dikkate alarak uzun dönemli artış veya azalış eğilimini sınar.",
        )
        alpha = float(a3.selectbox(
            "Anlamlılık düzeyi", [0.01, 0.05, 0.10], index=1,
            help="İstatistiksel karar sınırıdır. 0,05 seçimi, yanlış pozitif sonuç için %5 kabul edilen hata olasılığı anlamına gelir.",
        ))
        max_lag, correlation_method = 0, "Spearman"
        if has_drought_analysis and has_response_analysis:
            max_lag = int(trend_columns[3].slider(
                "Azami gecikme (ay)", 0, 12, 6,
                help="Kuraklık ile bitki örtüsü veya yüzey sıcaklığı tepkisi arasında aranacak en uzun gecikmedir.",
            ))
            correlation_method = st.selectbox(
                "Kuraklık–ekosistem ilişki yöntemi", ["Spearman", "Pearson"],
                help="Spearman doğrusal olmayan sıralı ilişkiler için daha dayanıklıdır; Pearson doğrusal ilişkiyi ölçer.",
            )

        response_indices = selected_response_components
        land_cover_labels: list[str] = []
        if has_response_analysis:
            st.info(
                "Ekosistem değişkenleri Analiz Seçimi bölümünden otomatik aktarıldı: "
                + " + ".join(response_indices)
            )
            land_cover_labels = st.multiselect(
                "Arazi örtüsüne göre ayrı analiz",
                list(LAND_COVER_CLASSES.values()),
                default=["Ağaç örtüsü", "Otlak/mera", "Tarım alanı"],
                help="ESA WorldCover 2021 sınıflarıyla NDVI/EVI/LST zonal ortalamaları oluşturulur.",
            )
        land_cover_codes = [
            code for code, label in LAND_COVER_CLASSES.items() if label in land_cover_labels
        ]

        station_upload = None
        if has_drought_analysis:
            st.markdown("##### İstasyonla bağımsız doğrulama (isteğe bağlı)")
            station_upload = st.file_uploader(
                "Aylık/günlük istasyon yağış tablosu",
                type=["csv", "xlsx", "xls"],
                help="Tarih ve yağış sütunu içeren bir tablo yüklenirse aylık toplam alınarak doğrulamaya eklenir.",
            )
        if has_drought_analysis and station_upload is not None:
            try:
                station_raw = (
                    pd.read_excel(station_upload)
                    if station_upload.name.lower().endswith((".xlsx", ".xls"))
                    else pd.read_csv(station_upload, sep=None, engine="python")
                )
                s1, s2 = st.columns(2)
                station_date_column = s1.selectbox(
                    "İstasyon tarih sütunu", list(station_raw.columns),
                    help="Gözlemin gününü veya ayını içeren sütunu seçin; değerler aylık toplam için tarihe göre gruplanır.",
                )
                station_numeric = [
                    column
                    for column in station_raw.columns
                    if pd.to_numeric(station_raw[column], errors="coerce").notna().any()
                    and column != station_date_column
                ]
                station_value_column = s2.selectbox(
                    "İstasyon yağış sütunu", station_numeric,
                    help="Milimetre cinsinden günlük veya aylık yağış miktarını içeren sayısal sütunu seçin.",
                )
                station_dates = pd.to_datetime(station_raw[station_date_column], errors="coerce")
                station_values = pd.to_numeric(station_raw[station_value_column], errors="coerce")
                station_monthly = (
                    pd.DataFrame({"Tarih": station_dates, "İstasyon yağış (mm)": station_values})
                    .dropna(subset=["Tarih"])
                    .set_index("Tarih")
                    .resample("MS")
                    .sum(min_count=1)
                    .reset_index()
                )
                st.session_state.academic_station_data = station_monthly
                st.success(f"İstasyon serisi hazır: {len(station_monthly):,} aylık kayıt.")
            except Exception as station_error:
                st.session_state.pop("academic_station_data", None)
                st.error(f"İstasyon tablosu okunamadı: {station_error}")
        elif has_drought_analysis and st.button("Yüklü istasyon serisini temizle"):
            st.session_state.pop("academic_station_data", None)
            st.rerun()

        academic_params = {
            "drought_indices": drought_indices,
            "scales": scales,
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "spi_distribution": spi_distribution,
            "spei_distribution": spei_distribution,
            "event_threshold": event_threshold,
            "prewhiten": prewhiten,
            "seasonal_mk": seasonal_mk,
            "alpha": alpha,
            "max_lag": max_lag,
            "correlation_method": correlation_method,
            "remove_seasonality": True,
            "response_indices": response_indices,
            "land_cover_codes": land_cover_codes,
            "land_cover_labels": land_cover_labels,
        }
        analysis_params.update(academic_params)
        selected_analyses = [*drought_indices, *response_indices]
        st.success(
            "Araştırma tasarımı hazır. Kaynak sürümleri, yöntem parametreleri, kalite "
            "ölçütleri ve bütün test sonuçları metadata ile bilimsel rapora kaydedilecektir."
        )

with tab_output:
    st.markdown('<div class="step">İşlemi oluştur ve indir</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hint">
          <strong>Dosya nereye kaydedilir?</strong><br>
          İndir düğmesine bastığınızda dosya tarayıcınızın varsayılan
          <strong>İndirilenler (Downloads)</strong> klasörüne kaydedilir.
          Tarayıcınız “Her indirmede konum sor” ayarındaysa hedef klasörü siz seçersiniz.
          Zetriklim dosyayı sunucuda kalıcı olarak saklamaz.
        </div>
        """,
        unsafe_allow_html=True,
    )
    summary = st.session_state.get("geometry_summary")
    current_output_config = (
        selected_analysis,
        provider,
        product,
        str(start_date),
        str(end_date),
        round(summary.area_km2, 4) if summary else None,
        application_mode,
        json.dumps(
            {
                "academic": academic_params,
                "study": academic_study,
                "cartography": {
                    "title": map_title,
                    "use_gadm": map_use_gadm,
                    "gadm_iso3": gadm_iso3,
                    "gadm_level": gadm_level,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
    )
    if (
        st.session_state.get("output_config")
        and st.session_state.output_config != current_output_config
    ):
        for state_key in [
            "output_files", "output_package", "output_metadata", "output_data",
            "output_source", "output_analysis_errors", "output_tile_url",
            "output_academic_results",
        ]:
            st.session_state.pop(state_key, None)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Alan", f"{summary.area_km2:,.2f} km²" if summary else "Bekleniyor")
    q2.metric("Kaynak", provider)
    q3.metric("Değişken", len(variables))
    q4.metric("Yöntem", len(selected_analyses))

    output_options = [
        "Proje paketi (ZIP)", "CSV zaman serisi", "Excel çalışma kitabı",
        "GeoPackage", "GeoJSON", "PNG harita",
    ]
    if provider == "Google Earth Engine":
        output_options.insert(3, "GeoTIFF raster")
    else:
        output_options.append("Etkileşimli HTML harita")
    output_formats = st.multiselect(
        "İstenen çıktılar",
        output_options,
        default=output_options,
        help="Yalnızca seçilen kaynağın gerçekten üretebildiği çıktı türleri listelenir.",
        key=f"output_formats_{provider}",
    )
    include_items = st.multiselect(
        "Pakete eklenecek içerik",
        ["Ham veri", "İşlenmiş veri", "Analiz sonuçları", "Grafikler", "Kaynak ve yöntem metadata", "Kalite kontrol raporu"],
        default=["Analiz sonuçları", "Kaynak ve yöntem metadata", "Kalite kontrol raporu"],
        help="ZIP proje paketinin içinde bulunmasını istediğiniz içerikleri seçin.",
    )

    source_ready = (
        bool(climate_engine_key and ce_dataset_id and ce_variable_ids)
        if provider == "Climate Engine"
        else bool(gee_ok and gee_project)
    )
    can_build = bool(
        summary and variables and selected_analyses and period_compatible and source_ready
    )
    if not summary:
        st.warning("Önce Alan sekmesinden geçerli bir çalışma alanı yükleyin.")
    if not selected_analyses:
        st.warning("Analiz sekmesinden en az bir yöntem seçin.")
    if provider == "Climate Engine" and not source_ready:
        st.warning(
            "Climate Engine işlemi için doğrulanmış API anahtarı, dataset parametresi "
            "ve en az bir değişken parametresi gereklidir."
        )
    if provider == "Google Earth Engine" and not source_ready:
        st.warning(
            "Earth Engine işlemi için Project ID ve tamamlanmış Google yetkilendirmesi gereklidir."
        )

    if st.button(
        "İşlemi oluştur",
        type="primary",
        icon=":material/play_arrow:",
        disabled=not can_build,
        use_container_width=True,
    ):
        live_supported = provider in {
            "Otomatik en uygun açık kaynak",
            "Open-Meteo Historical",
            "Google Earth Engine",
            "Climate Engine",
        }
        if not live_supported:
            st.error(
                f"{provider} için canlı indirme bağlayıcısı henüz etkin değil. "
                "Gerçek veri almak için şimdilik “Otomatik en uygun açık kaynak” "
                "veya “Open-Meteo Historical” seçin."
            )
        else:
            try:
                with st.spinner("Seçilen kaynaktan gerçek veri indiriliyor ve çıktılar hazırlanıyor..."):
                    source_request_metadata = {}
                    academic_source_metadata = {}
                    if provider == "Google Earth Engine":
                        climate_latitude, climate_longitude = summary.centroid
                        climate_elevation = None
                        unsupported = []
                        if academic_mode:
                            climate_data, academic_source_metadata = fetch_gee_academic_series(
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                response_indices=list(academic_params.get("response_indices", [])),
                                land_cover_codes=list(academic_params.get("land_cover_codes", [])),
                                project=gee_project,
                            )
                            climate_model = (
                                "CHIRPS + ERA5-Land + Sentinel-2/Landsat "
                                "via Google Earth Engine"
                            )
                            climate_url = "https://developers.google.com/earth-engine/datasets/catalog"
                            source_request_metadata["academic_datasets"] = academic_source_metadata
                        elif selected_analysis == "SPI":
                            climate_data, unsupported = fetch_gee_monthly_climate(
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                ["Yağış"],
                                project=gee_project,
                            )
                            climate_model = "CHIRPS Daily via Google Earth Engine"
                            climate_url = (
                                "https://developers.google.com/earth-engine/datasets/catalog/"
                                "UCSB-CHG_CHIRPS_DAILY"
                            )
                        else:
                            climate_data = pd.DataFrame(
                                [{
                                    "Tarih": pd.Timestamp(end_date),
                                    "Örnek ID": 1,
                                    "Enlem": climate_latitude,
                                    "Boylam": climate_longitude,
                                    "Analiz": selected_analysis,
                                    "Dönem başlangıcı": str(start_date),
                                    "Dönem bitişi": str(end_date),
                                }]
                            )
                            climate_model = f"{product} via Google Earth Engine"
                            climate_url = "https://developers.google.com/earth-engine/datasets/catalog"
                    elif provider == "Climate Engine":
                        climate_data, ce_metadata = fetch_climate_engine_timeseries(
                            climate_engine_key,
                            summary.gdf_wgs84,
                            start_date,
                            end_date,
                            ce_dataset_id,
                            ce_variable_ids,
                            area_reducer="mean",
                        )
                        climate_model = (
                            f"Climate Engine API · {ce_dataset_id} · {ce_variable_ids}"
                        )
                        climate_url = ce_metadata["endpoint"]
                        source_request_metadata = ce_metadata
                        climate_latitude, climate_longitude = summary.centroid
                        climate_elevation = None
                        unsupported = []
                        if academic_mode:
                            supporting_climate = fetch_centroid_series(
                                latitude=summary.centroid[0],
                                longitude=summary.centroid[1],
                                start_date=start_date,
                                end_date=end_date,
                                variables=["Hava sıcaklığı", "Potansiyel evapotranspirasyon"],
                                temporal_scale="Aylık",
                            )
                            supporting_data = supporting_climate.data.drop(
                                columns=["Örnek ID", "Enlem", "Boylam"], errors="ignore"
                            )
                            climate_data["Tarih"] = pd.to_datetime(
                                climate_data["Tarih"], errors="coerce"
                            )
                            supporting_data["Tarih"] = pd.to_datetime(
                                supporting_data["Tarih"], errors="coerce"
                            )
                            climate_data = climate_data.merge(
                                supporting_data, on="Tarih", how="outer"
                            ).sort_values("Tarih")
                            climate_model += " + ERA5/Open-Meteo PET ve sıcaklık"
                            source_request_metadata["academic_support_source"] = (
                                supporting_climate.source_url
                            )
                    else:
                        climate = fetch_centroid_series(
                            latitude=summary.centroid[0],
                            longitude=summary.centroid[1],
                            start_date=start_date,
                            end_date=end_date,
                            variables=variables,
                            temporal_scale=temporal_scale,
                        )
                        climate_data = climate.data
                        climate_model = climate.model
                        climate_url = climate.source_url
                        climate_latitude = climate.latitude
                        climate_longitude = climate.longitude
                        climate_elevation = climate.elevation_m
                        unsupported = climate.unsupported_variables

                    analysis_tables = {}
                    spi_table = None
                    academic_results: dict[str, pd.DataFrame] = {}
                    station_data = st.session_state.get("academic_station_data")
                    if academic_mode and drought_indices and isinstance(station_data, pd.DataFrame):
                        climate_data["Tarih"] = pd.to_datetime(
                            climate_data["Tarih"], errors="coerce"
                        )
                        station_data = station_data.copy()
                        station_data["Tarih"] = pd.to_datetime(
                            station_data["Tarih"], errors="coerce"
                        )
                        climate_data = climate_data.merge(
                            station_data, on="Tarih", how="outer"
                        ).sort_values("Tarih")

                    if academic_mode:
                        precipitation_columns = [
                            column
                            for column in climate_data.columns
                            if any(token in column.lower() for token in ["yağış", "precip"])
                            and pd.to_numeric(climate_data[column], errors="coerce").notna().any()
                        ]
                        precipitation_columns.sort(
                            key=lambda column: (
                                0 if "chirps" in column.lower() else
                                1 if "toplam yağış" in column.lower() else 2
                            )
                        )
                        needs_precipitation = any(
                            item in {"SPI", "SPEI"}
                            for item in academic_params.get("drought_indices", [])
                        )
                        if needs_precipitation and not precipitation_columns:
                            raise ValueError(
                                "Seçilen SPI/SPEI analizi için sayısal yağış sütunu bulunamadı. "
                                "Yalnız NDVI/EVI/LST seçildiyse kuraklık indislerini boş bırakın."
                            )
                        pet_columns = [
                            column
                            for column in climate_data.columns
                            if any(token in column.lower() for token in ["pet", "et₀", "evapotrans"])
                            and pd.to_numeric(climate_data[column], errors="coerce").notna().any()
                        ]
                        response_columns = [
                            column
                            for column in climate_data.columns
                            if any(
                                column == response or column.startswith(f"{response}|")
                                for response in academic_params.get("response_indices", [])
                            )
                        ]
                        validation_columns = [
                            column for column in precipitation_columns[1:]
                            if column != precipitation_columns[0]
                        ]
                        academic_results = run_academic_analysis(
                            climate_data,
                            precipitation_column=precipitation_columns[0] if precipitation_columns else None,
                            pet_column=pet_columns[0] if pet_columns else None,
                            response_columns=response_columns,
                            validation_columns=validation_columns,
                            config=academic_params,
                        )
                        analysis_tables.update(
                            {
                                name: table
                                for name, table in academic_results.items()
                                if table is not None and not table.empty
                            }
                        )
                        academic_series = academic_results.get("Kuraklık Serisi", pd.DataFrame())
                        spi_columns = [
                            column for column in academic_series.columns
                            if column.startswith("SPI-")
                        ]
                        spi_table = (
                            academic_series[["Tarih", *spi_columns]].copy()
                            if spi_columns else None
                        )
                    elif "SPI" in selected_analyses:
                        spi_input_data = climate_data
                        spi_source = climate_model
                        gee_ok, _ = cached_gee_status(gee_project or None)
                        if provider == "Otomatik en uygun açık kaynak" and gee_ok:
                            chirps_data = fetch_chirps_monthly_mean(
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                project=gee_project,
                            )
                            era_rain_columns = [
                                column for column in climate_data.columns
                                if "yağış" in column.lower() and climate_data[column].dtype.kind in "fi"
                            ]
                            if era_rain_columns:
                                era_monthly = (
                                    climate_data.set_index("Tarih")[era_rain_columns[0]]
                                    .resample("MS")
                                    .sum(min_count=1)
                                    .rename("ERA5 yağış (mm)")
                                    .reset_index()
                                )
                                comparison = chirps_data[["Tarih", "Toplam yağış (mm)"]].rename(
                                    columns={"Toplam yağış (mm)": "CHIRPS yağış (mm)"}
                                ).merge(era_monthly, on="Tarih", how="outer")
                                comparison["Fark (CHIRPS - ERA5)"] = (
                                    comparison["CHIRPS yağış (mm)"] - comparison["ERA5 yağış (mm)"]
                                )
                                analysis_tables["Yağış Karşılaştırma"] = comparison
                            spi_input_data = chirps_data
                            spi_source = "CHIRPS Daily via Google Earth Engine"
                        precipitation_columns = [
                            column for column in spi_input_data.columns
                            if any(token in column.lower() for token in ["yağış", "precip"])
                            and pd.to_numeric(spi_input_data[column], errors="coerce").notna().any()
                        ]
                        if not precipitation_columns:
                            available_columns = ", ".join(map(str, spi_input_data.columns))
                            raise ValueError(
                                "SPI hesaplanamadı: veri tablosunda sayısal yağış sütunu bulunamadı. "
                                f"Mevcut sütunlar: {available_columns}"
                            )
                        spi_scales = analysis_params.get("scales") or [1, 3, 6, 12]
                        spi_table = calculate_spi_table(
                            spi_input_data,
                            precipitation_columns[0],
                            spi_scales,
                        )
                        spi_table.insert(1, "SPI yağış kaynağı", spi_source)
                        analysis_tables["SPI Sonuçları"] = spi_table
                    gadm_context = None
                    gadm_metadata = {}
                    gadm_error = None
                    if map_use_gadm:
                        try:
                            gadm_context, gadm_metadata = cached_gadm_boundaries(
                                gadm_iso3,
                                gadm_level,
                            )
                        except Exception as context_error:
                            gadm_error = str(context_error)
                    source_request_metadata["cartography"] = {
                        "title": map_title,
                        "gadm": gadm_metadata or None,
                        "gadm_error": gadm_error,
                    }

                    metadata = build_metadata(
                        area={
                            "area_km2": round(summary.area_km2, 6),
                            "perimeter_km": round(summary.perimeter_km, 6),
                            "feature_count": summary.feature_count,
                            "source_crs": summary.source_crs,
                            "calculation_crs": summary.area_crs,
                            "spatial_mode": spatial_mode,
                            "spatial_statistics": spatial_stat,
                            "sampling_note": "İlk çalışan bağlayıcı alan merkezindeki ERA5-Land grid hücresini kullanır.",
                        },
                        request={
                            "provider": provider,
                            "product": climate_model,
                            "source_url": climate_url,
                            "variables": variables,
                            "unsupported_variables": unsupported,
                            "start_date": str(start_date),
                            "end_date": str(end_date),
                            "temporal_scale": temporal_scale,
                            "aggregation": aggregation,
                            "quality_control": quality_control,
                            "source_request_metadata": source_request_metadata,
                        },
                        analysis={
                            "mode": application_mode,
                            "methods": selected_analyses,
                            "parameters": analysis_params,
                            "study": academic_study if academic_mode else None,
                        },
                        outputs={"formats": output_formats, "contents": include_items},
                    )
                    area_geojson = summary.gdf_wgs84.to_json(drop_id=True).encode("utf-8")
                    metadata_rows = [
                        ("Veri kaynağı", provider),
                        ("Asıl ürün", climate_model),
                        ("Kaynak URL", climate_url),
                        ("Başlangıç", start_date),
                        ("Bitiş", end_date),
                        ("Zaman çözünürlüğü", temporal_scale),
                        ("Örnekleme", "Çalışma alanı merkezindeki ERA5-Land grid hücresi"),
                        ("Enlem", climate_latitude),
                        ("Boylam", climate_longitude),
                        ("Model yüksekliği (m)", climate_elevation),
                        ("Desteklenmeyen seçimler", ", ".join(unsupported) or "Yok"),
                        ("Çalışma modu", application_mode),
                        (
                            "Referans dönemi",
                            f"{academic_params.get('baseline_start')}–{academic_params.get('baseline_end')}"
                            if academic_mode else analysis_params.get("baseline", "—"),
                        ),
                    ]
                    area_rows = [
                        ("Toplam alan (km²)", summary.area_km2),
                        ("Çevre (km)", summary.perimeter_km),
                        ("Coğrafi obje", summary.feature_count),
                        ("Köşe sayısı", summary.vertex_count),
                        ("Kaynak CRS", summary.source_crs),
                        ("Alan hesabı CRS", summary.area_crs),
                        ("Merkez enlem", summary.centroid[0]),
                        ("Merkez boylam", summary.centroid[1]),
                    ]
                    export_data = (
                        academic_results.get("Birleşik Analiz Serisi", climate_data)
                        if academic_mode else climate_data
                    )
                    excel = build_excel(
                        export_data,
                        metadata_rows=metadata_rows,
                        area_rows=area_rows,
                        analysis_tables=analysis_tables,
                    )
                    csv_data = dataframe_to_csv(export_data)
                    csv_excel_tr = dataframe_to_csv(export_data, excel_tr=True)
                    graph_png = build_timeseries_png(export_data)
                    map_png = build_area_map_png(
                        summary.gdf_wgs84,
                        summary.centroid,
                        context=gadm_context,
                        title=map_title,
                        subtitle=f"{start_date} – {end_date} · {', '.join(selected_analyses)}",
                        context_label=f"GADM {gadm_iso3} düzey {gadm_level}",
                        source_note=(
                            "Çalışma alanı: kullanıcı verisi · İdari sınırlar: GADM 4.1"
                            if gadm_context is not None
                            else "Çalışma alanı: kullanıcı verisi"
                        ),
                    )
                    gpkg = geodata_to_gpkg(summary.gdf_wgs84, summary.centroid)
                    readme = (
                        "ZETRİKLİM GERÇEK VERİ VE ANALİZ PAKETİ\n\n"
                        f"Kaynak: {climate_model}\n"
                        f"Dönem: {start_date} – {end_date}\n"
                        f"Kayıt sayısı: {len(export_data):,}\n"
                        f"Örnekleme: {'Havza alan ortalaması' if provider == 'Google Earth Engine' else 'Çalışma alanının merkezindeki ERA5-Land grid hücresi'}.\n\n"
                        "DOSYALAR\n"
                        "- zetriklim-veri.xlsx: veri, kaynak, alan ve özet sayfaları\n"
                        "- zetriklim-veri.csv: CBS ve istatistik yazılımları için tablo\n"
                        "- zetriklim-veri-excel-tr.csv: Türkçe Excel için noktalı virgüllü tablo\n"
                        "- zetriklim-cbs.gpkg: çalışma alanı ve örnekleme noktası\n"
                        "- calisma-alani.geojson: çalışma alanı sınırı\n"
                        "- zaman-serisi.png: iklim grafiği\n"
                        "- calisma-alani-haritasi.png: çalışma alanı / havza sınırı\n"
                        "- [YONTEM]_[DONEM].tif: havza sınırına kırpılmış CBS raster katmanı\n"
                        "- [YONTEM]_[DONEM].png: lejantlı akademik harita önizlemesi\n"
                        "- raster-analiz-metadata.json: raster kaynağı, formül, dönem, çözünürlük ve sahne sayısı\n"
                        "- zetriklim-metadata.json: veri kaynağı ve işlem izi\n"
                    ).encode("utf-8")
                    if academic_mode:
                        readme += (
                            "\nAKADEMİK ARAŞTIRMA DOSYALARI\n"
                            "- bilimsel-rapor.html: araştırma sorusu, yöntem, bulgu tabloları ve kaynaklar\n"
                            "- akademik-kuraklik-serisi.csv: SPI ve SPEI zaman serileri\n"
                            "- akademik-analiz-serisi.csv: kaynak değişkenleri ve seçili indislerin birleşik aylık serisi\n"
                            "- kuraklik-olaylari.csv: olay başlangıcı, bitişi, süre, şiddet ve yoğunluk\n"
                            "- egilim-ve-degisim.csv: Mann–Kendall, Sen eğimi ve Pettitt sonuçları\n"
                            "- gecikmeli-iliski.csv: kuraklık–NDVI/EVI/LST gecikmeli korelasyonları\n"
                            "- kaynak-dogrulama.csv: Bias, MAE, RMSE, korelasyon ve KGE\n"
                            "- belirsizlik.csv: kaynaklar arası ensemble yayılımı\n"
                            "- kalite-kontrol.csv: eksik kayıt, aralık ve zaman sürekliliği kontrolleri\n"
                        ).encode("utf-8")
                    files = {
                        "zetriklim-veri.xlsx": excel,
                        "zetriklim-veri.csv": csv_data,
                        "zetriklim-veri-excel-tr.csv": csv_excel_tr,
                        "zetriklim-cbs.gpkg": gpkg,
                        "calisma-alani.geojson": area_geojson,
                        "zaman-serisi.png": graph_png,
                        "calisma-alani-haritasi.png": map_png,
                        "zetriklim-metadata.json": metadata,
                        "BENI-OKU.txt": readme,
                    }
                    if academic_mode:
                        academic_file_names = {
                            "Kuraklık Serisi": "akademik-kuraklik-serisi.csv",
                            "Kuraklık Olayları": "kuraklik-olaylari.csv",
                            "Eğilim ve Değişim": "egilim-ve-degisim.csv",
                            "Gecikmeli İlişki": "gecikmeli-iliski.csv",
                            "Kaynak Doğrulama": "kaynak-dogrulama.csv",
                            "Belirsizlik": "belirsizlik.csv",
                            "Kalite Kontrol": "kalite-kontrol.csv",
                            "Dağılım Uyum": "dagilim-uyum-testleri.csv",
                            "Birleşik Analiz Serisi": "akademik-analiz-serisi.csv",
                            "Analiz Uyarıları": "akademik-analiz-uyarilari.csv",
                        }
                        for result_name, file_name in academic_file_names.items():
                            result_table = academic_results.get(result_name)
                            if result_table is not None and not result_table.empty:
                                files[file_name] = dataframe_to_csv(result_table)
                        files["bilimsel-rapor.html"] = build_academic_report_html(
                            study=academic_study,
                            config=academic_params,
                            results=academic_results,
                            source_note=climate_model,
                        )
                    if gadm_metadata:
                        files["gadm-harita-kaynagi.json"] = json.dumps(
                            gadm_metadata,
                            ensure_ascii=False,
                            indent=2,
                        ).encode("utf-8")
                    if gadm_error:
                        files["harita-uyarisi.txt"] = (
                            "GADM bağlam katmanı indirilemedi; çalışma alanı haritası kullanıcı sınırıyla üretildi.\n"
                            f"Ayrıntı: {gadm_error}"
                        ).encode("utf-8")
                    remote_errors = []
                    climate_engine_tile_url = None
                    if provider == "Climate Engine":
                        try:
                            (
                                climate_engine_tile_url,
                                ce_map_metadata,
                            ) = fetch_climate_engine_map_tile(
                                climate_engine_key,
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                ce_dataset_id,
                                ce_variable_ids,
                                selected_analysis,
                            )
                            ce_map = folium.Map(
                                summary.centroid,
                                zoom_start=8,
                                tiles="CartoDB positron",
                                control_scale=True,
                            )
                            folium.TileLayer(
                                tiles=climate_engine_tile_url,
                                attr="Climate Engine / Google Earth Engine",
                                name=f"{selected_analysis} · {ce_dataset_id}",
                                overlay=True,
                                control=True,
                                opacity=0.85,
                            ).add_to(ce_map)
                            folium.GeoJson(
                                summary.gdf_wgs84.__geo_interface__,
                                name="Çalışma alanı",
                                style_function=lambda _: {
                                    "color": "#052f42",
                                    "weight": 3,
                                    "fillOpacity": 0,
                                },
                            ).add_to(ce_map)
                            bounds = summary.bounds
                            ce_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                            add_cartographic_controls(
                                ce_map,
                                selected_analysis,
                                f"Climate Engine · {ce_dataset_id}",
                                start_date,
                                end_date,
                            )
                            folium.LayerControl().add_to(ce_map)
                            files[
                                f"{selected_analysis}_ClimateEngine_etkilesimli_harita.html"
                            ] = ce_map.get_root().render().encode("utf-8")
                            files["climate-engine-harita-metadata.json"] = json.dumps(
                                ce_map_metadata,
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ).encode("utf-8")
                        except Exception as map_error:
                            remote_errors.append(
                                f"{selected_analysis} Climate Engine haritası: {map_error}"
                            )

                    gee_ok = False
                    if provider == "Google Earth Engine":
                        gee_ok, _ = cached_gee_status(gee_project or None)
                    if provider == "Google Earth Engine" and gee_ok:
                        if "Yağış" in variables:
                            precipitation_tif = build_climate_geotiff(
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                "precipitation",
                                project=gee_project,
                            )
                            precipitation_name = f"yagis_toplam_{start_date}_{end_date}"
                            files[f"{precipitation_name}.tif"] = precipitation_tif
                            files[f"{precipitation_name}.png"] = build_raster_png(
                                precipitation_tif,
                                f"CHIRPS Toplam Yağış · {start_date} – {end_date}",
                                boundary=summary.gdf_wgs84,
                                palette="YlGnBu",
                                colorbar_label="Yağış (mm)",
                                fixed_range=None,
                                source_note="CHIRPS Daily",
                            )
                        if "Hava sıcaklığı" in variables:
                            temperature_tif = build_climate_geotiff(
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                "temperature",
                                project=gee_project,
                            )
                            temperature_name = f"sicaklik_ortalama_{start_date}_{end_date}"
                            files[f"{temperature_name}.tif"] = temperature_tif
                            files[f"{temperature_name}.png"] = build_raster_png(
                                temperature_tif,
                                f"ERA5-Land Ortalama Sıcaklık · {start_date} – {end_date}",
                                boundary=summary.gdf_wgs84,
                                palette="coolwarm",
                                colorbar_label="Sıcaklık (°C)",
                                fixed_range=None,
                                source_note="ERA5-Land",
                            )
                    remote_methods = {
                        "NDVI", "NDWI", "NDMI", "NDBI", "EVI", "SAVI",
                        "LST", "DEM", "SLOPE", "ASPECT", "TWI",
                    }
                    remote_selected = [
                        method for method in selected_analyses if method in remote_methods
                    ]
                    remote_metadata = []
                    raster_styles = {
                        "NDVI": ("RdYlGn", "NDVI", (-1.0, 1.0)),
                        "NDWI": ("Blues", "NDWI", (-1.0, 1.0)),
                        "NDMI": ("BrBG", "NDMI", (-1.0, 1.0)),
                        "NDBI": ("magma", "NDBI", (-1.0, 1.0)),
                        "EVI": ("YlGn", "EVI", (-1.0, 1.0)),
                        "SAVI": ("summer", "SAVI", (-1.0, 1.0)),
                        "LST": ("inferno", "Yüzey sıcaklığı (°C)", None),
                        "DEM": ("terrain", "Yükselti (m)", None),
                        "SLOPE": ("YlOrBr", "Eğim (°)", (0.0, 60.0)),
                        "ASPECT": ("hsv", "Bakı (°)", (0.0, 360.0)),
                        "TWI": ("viridis", "TWI", None),
                    }
                    for method in (
                        remote_selected if provider == "Google Earth Engine" else []
                    ):
                        try:
                            raster_tif, method_metadata = build_remote_analysis_geotiff(
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                method,
                                project=gee_project or None,
                                cloud_limit=int(analysis_params.get("cloud_limit", 30)),
                            )
                            base_name = (
                                f"{method}_{start_date}_{end_date}"
                                if method in {"NDVI", "NDWI", "NDMI", "NDBI", "EVI", "SAVI", "LST"}
                                else f"{method}_statik"
                            )
                            palette, colorbar_label, fixed_range = raster_styles[method]
                            files[f"{base_name}.tif"] = raster_tif
                            files[f"{base_name}.png"] = build_raster_png(
                                raster_tif,
                                f"{method} · {ANALYSIS_METHODS[method]['title']}",
                                boundary=summary.gdf_wgs84,
                                palette=palette,
                                colorbar_label=colorbar_label,
                                fixed_range=fixed_range,
                                source_note=str(method_metadata.get("source", climate_model)),
                            )
                            remote_metadata.append(method_metadata)
                        except Exception as method_error:
                            remote_errors.append(f"{method}: {method_error}")
                    if remote_metadata:
                        files["raster-analiz-metadata.json"] = json.dumps(
                            remote_metadata,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ).encode("utf-8")
                    if remote_errors:
                        files["analiz-uyarilari.txt"] = (
                            "Üretilemeyen analizler\n\n" + "\n".join(remote_errors)
                        ).encode("utf-8")
                    if spi_table is not None and "SPI" in selected_analyses:
                        files["spi-sonuclari.csv"] = dataframe_to_csv(spi_table)
                        if provider == "Google Earth Engine" and gee_ok:
                            for selected_scale in (analysis_params.get("scales") or [3]):
                                selected_scale = int(selected_scale)
                                spi_tif = build_chirps_spi_geotiff(
                                    summary.gdf_wgs84,
                                    end_date,
                                    selected_scale,
                                    baseline_start=int(academic_params.get("baseline_start", 1981)),
                                    baseline_end=int(academic_params.get("baseline_end", 2024)),
                                    project=gee_project,
                                )
                                files[f"SPI-{selected_scale}_{end_date:%Y-%m}.tif"] = spi_tif
                                files[f"SPI-{selected_scale}_{end_date:%Y-%m}.png"] = build_raster_png(
                                    spi_tif,
                                    f"CHIRPS SPI-{selected_scale} · {end_date:%Y-%m}",
                                    boundary=summary.gdf_wgs84,
                                    source_note="CHIRPS Daily · SPI sınıfları WMO eşikleri",
                                )
                    st.session_state.output_files = files
                    st.session_state.output_package = build_complete_package(files)
                    st.session_state.output_metadata = metadata
                    st.session_state.output_data = export_data
                    st.session_state.output_source = climate_model
                    st.session_state.output_analysis_errors = remote_errors
                    st.session_state.output_tile_url = climate_engine_tile_url
                    st.session_state.output_boundary = summary.gdf_wgs84.to_json()
                    st.session_state.output_config = current_output_config
                    st.session_state.output_academic_results = academic_results
                if st.session_state.get("output_analysis_errors"):
                    st.warning(
                        f"Tablo verisi hazırlandı ({len(climate_data):,} kayıt), ancak seçilen "
                        "analizin bütün görsel çıktıları tamamlanamadı."
                    )
                else:
                    st.success(
                        f"{' + '.join(selected_analyses)} analizi tamamlandı: {len(export_data):,} aylık kayıt. "
                        "Tablolar, haritalar ve CBS paketi hazır."
                    )
                if unsupported:
                    st.warning(
                        "Bu bağlayıcıda desteklenmeyen değişkenler pakete eklenmedi: "
                        + ", ".join(unsupported)
                    )
                if st.session_state.get("output_analysis_errors"):
                    st.warning(
                        "Bazı rasterlar üretilemedi; başarıyla üretilen diğer çıktılar korundu:\n\n"
                        + "\n".join(
                            f"- {item}"
                            for item in st.session_state.output_analysis_errors
                        )
                    )
            except Exception as exc:
                st.error(f"Veri indirme veya çıktı oluşturma başarısız: {exc}")

    if st.session_state.get("output_package"):
        st.info(
            f"Kaynak: {st.session_state.get('output_source', '—')} · "
            f"Kayıt: {len(st.session_state.get('output_data', [])):,} · "
            "İndirme hedefi: tarayıcınızın İndirilenler klasörü"
        )
        academic_output = st.session_state.get("output_academic_results", {})
        if academic_mode and academic_output:
            st.subheader("Akademik bulgular özeti")
            academic_warnings = academic_output.get("Analiz Uyarıları", pd.DataFrame())
            if not academic_warnings.empty:
                st.warning("Bazı seçili bileşenler gerekli değişken bulunamadığı için hesaplanamadı.")
                st.dataframe(academic_warnings, width="stretch", hide_index=True)
            event_table = academic_output.get("Kuraklık Olayları", pd.DataFrame())
            trend_table = academic_output.get("Eğilim ve Değişim", pd.DataFrame())
            lag_table = academic_output.get("Gecikmeli İlişki", pd.DataFrame())
            validation_table = academic_output.get("Kaynak Doğrulama", pd.DataFrame())
            am1, am2, am3, am4 = st.columns(4)
            am1.metric("Kuraklık olayı", len(event_table))
            am2.metric(
                "Anlamlı eğilim",
                int(trend_table.get("Anlamlı eğilim", pd.Series(dtype=bool)).fillna(False).sum()),
            )
            am3.metric(
                "Anlamlı gecikmeli ilişki",
                int(lag_table.get("Anlamlı", pd.Series(dtype=bool)).fillna(False).sum()),
            )
            am4.metric("Doğrulama çifti", len(validation_table))

            result_tabs = st.tabs(
                ["Kuraklık olayları", "Eğilim ve değişim", "Gecikmeli tepki", "Doğrulama ve belirsizlik"]
            )
            with result_tabs[0]:
                if event_table.empty:
                    st.info("Seçilen eşikte kuraklık olayı bulunmadı veya seri yetersiz.")
                else:
                    st.dataframe(event_table, width="stretch", hide_index=True)
            with result_tabs[1]:
                if trend_table.empty:
                    st.info("Eğilim testi için yeterli geçerli gözlem bulunmadı.")
                else:
                    st.dataframe(trend_table, width="stretch", hide_index=True)
            with result_tabs[2]:
                best_lags = (
                    lag_table[lag_table.get("En güçlü gecikme", False) == True]  # noqa: E712
                    if not lag_table.empty else lag_table
                )
                if best_lags.empty:
                    st.info("Uydu tepkisi için yeterli ortak dönem bulunmadı.")
                else:
                    st.dataframe(best_lags, width="stretch", hide_index=True)
            with result_tabs[3]:
                if not validation_table.empty:
                    st.markdown("**Kaynak karşılaştırması**")
                    st.dataframe(validation_table, width="stretch", hide_index=True)
                uncertainty = academic_output.get("Belirsizlik", pd.DataFrame())
                if not uncertainty.empty:
                    st.markdown("**Kaynaklar arası belirsizlik serisi**")
                    st.dataframe(uncertainty.tail(120), width="stretch", hide_index=True)

        st.subheader("Analiz zaman serisi")
        st.image(
            st.session_state.output_files["zaman-serisi.png"],
            caption=(
                f"{selected_analysis} · {start_date} – {end_date} · "
                "NoData değerleri kalite kontrolünde grafik dışında bırakılmıştır."
            ),
            width="stretch",
        )
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button(
            "Tüm paketi indir (ZIP)",
            st.session_state.output_package,
            file_name="zetriklim-analiz-paketi.zip",
            mime="application/zip",
            use_container_width=True,
        )
        d2.download_button(
            "Excel indir",
            st.session_state.output_files["zetriklim-veri.xlsx"],
            file_name="zetriklim-veri.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        d3.download_button(
            "Standart CSV indir",
            st.session_state.output_files["zetriklim-veri.csv"],
            file_name="zetriklim-veri.csv",
            mime="text/csv",
            use_container_width=True,
        )
        d4.download_button(
            "Türkçe Excel CSV",
            st.session_state.output_files["zetriklim-veri-excel-tr.csv"],
            file_name="zetriklim-veri-excel-tr.csv",
            mime="text/csv",
            use_container_width=True,
            help="Noktalı virgül ayırıcı ve ondalık virgül kullanır; Türkçe Excel'de sütunlara doğru ayrılır.",
        )
        cbs1, cbs2, cbs3 = st.columns(3)
        cbs1.download_button(
            "GeoPackage indir",
            st.session_state.output_files["zetriklim-cbs.gpkg"],
            file_name="zetriklim-cbs.gpkg",
            mime="application/geopackage+sqlite3",
            use_container_width=True,
        )
        cbs2.download_button(
            "Zaman serisi grafiği",
            st.session_state.output_files["zaman-serisi.png"],
            file_name="zaman-serisi.png",
            mime="image/png",
            use_container_width=True,
        )
        cbs3.download_button(
            "Alan haritası",
            st.session_state.output_files["calisma-alani-haritasi.png"],
            file_name="calisma-alani-haritasi.png",
            mime="image/png",
            use_container_width=True,
        )
        with st.expander("Tez formatında çalışma alanı haritasını önizle", expanded=True):
            st.image(
                st.session_state.output_files["calisma-alani-haritasi.png"],
                caption="Ölçek, kuzey oku, koordinat ağı, lejant, projeksiyon ve kaynak bilgisi içeren konum haritası",
                width="stretch",
            )
        extra1, extra2 = st.columns(2)
        extra1.download_button(
            "GeoJSON sınırını indir",
            st.session_state.output_files["calisma-alani.geojson"],
            file_name="calisma-alani.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )
        extra2.download_button(
            "Metadata indir",
            st.session_state.output_files["zetriklim-metadata.json"],
            file_name="zetriklim-metadata.json",
            mime="application/json",
            use_container_width=True,
        )
        if "bilimsel-rapor.html" in st.session_state.output_files:
            st.download_button(
                "Bilimsel raporu indir (HTML)",
                st.session_state.output_files["bilimsel-rapor.html"],
                file_name="zetriklim-bilimsel-rapor.html",
                mime="text/html",
                use_container_width=True,
            )
        raster_names = [name for name in st.session_state.output_files if name.lower().endswith(".tif")]
        if raster_names:
            st.subheader("CBS raster katmanları")
            raster_columns = st.columns(min(3, len(raster_names)))
            for index, raster_name in enumerate(raster_names):
                label = (
                    "Yağış GeoTIFF"
                    if raster_name.startswith("yagis_")
                    else "Sıcaklık GeoTIFF"
                    if raster_name.startswith("sicaklik_")
                    else f"{raster_name.split('_')[0]} GeoTIFF"
                )
                raster_columns[index % len(raster_columns)].download_button(
                    label,
                    st.session_state.output_files[raster_name],
                    file_name=raster_name,
                    mime="image/tiff",
                    use_container_width=True,
                    key=f"raster_download_{index}_{raster_name}",
                )
        html_map_names = [
            name for name in st.session_state.output_files
            if name.lower().endswith(".html") and "harita" in name.lower()
        ]
        if html_map_names:
            st.subheader("Etkileşimli analiz haritası")
            tile_url = st.session_state.get("output_tile_url")
            if tile_url and summary:
                result_map = folium.Map(
                    summary.centroid,
                    zoom_start=8,
                    tiles="CartoDB positron",
                    control_scale=True,
                )
                folium.TileLayer(
                    tiles=tile_url,
                    attr="Climate Engine / Google Earth Engine",
                    name=selected_analysis,
                    overlay=True,
                    opacity=0.85,
                ).add_to(result_map)
                folium.GeoJson(
                    summary.gdf_wgs84.__geo_interface__,
                    name="Çalışma alanı / havza sınırı",
                    style_function=lambda _: {
                        "color": "#052f42",
                        "weight": 3.5,
                        "fillColor": "#ffffff",
                        "fillOpacity": 0.04,
                    },
                ).add_to(result_map)
                bounds = summary.bounds
                result_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                add_cartographic_controls(
                    result_map,
                    selected_analysis,
                    st.session_state.get("output_source", provider),
                    start_date,
                    end_date,
                )
                folium.LayerControl(collapsed=False).add_to(result_map)
                st_folium(result_map, height=560, width="stretch", returned_objects=[])
                st.caption(
                    "Lejant renkleri analiz sınıflarını gösterir. Climate Engine rasterı "
                    "karo servisinde dikdörtgen kapsam olarak sunulur; koyu çizgi gerçek "
                    "çalışma alanı/havza sınırıdır."
                )
            for index, map_name in enumerate(html_map_names):
                st.download_button(
                    "Etkileşimli haritayı indir",
                    st.session_state.output_files[map_name],
                    file_name=map_name,
                    mime="text/html",
                    use_container_width=True,
                    key=f"html_map_{index}",
                )
        preview_names = [
            name
            for name in st.session_state.output_files
            if name.lower().endswith(".png")
            and name not in {"zaman-serisi.png", "calisma-alani-haritasi.png"}
        ]
        if preview_names:
            with st.expander("Üretilen iklim, indis ve topoğrafya haritalarını önizle", expanded=True):
                preview_columns = st.columns(min(2, len(preview_names)))
                for index, preview_name in enumerate(preview_names):
                    preview_columns[index % len(preview_columns)].image(
                        st.session_state.output_files[preview_name],
                        caption=preview_name.replace("_", " ").replace(".png", ""),
                        width="stretch",
                    )
        with st.expander("İndirilecek veriyi önizle", expanded=True):
            st.dataframe(st.session_state.output_data.head(500), width="stretch", hide_index=True)
            st.caption("Önizleme ilk 500 kaydı gösterir; indirilen Excel ve CSV tüm kayıtları içerir.")

st.markdown(
    """
    <div style="
      margin-top:3rem;padding:1.25rem 1.5rem;border-top:1px solid rgba(0,128,136,.22);
      text-align:center;color:#496b73;background:rgba(255,255,255,.45);border-radius:18px 18px 0 0">
      <strong style="color:#075b68;letter-spacing:.08em">ZETRİKLİM</strong><br>
      Havza, iklim ve uzaktan algılama analiz platformu<br>
      <span style="font-size:.82rem">Zeliha Konuk · 2026</span>
    </div>
    """,
    unsafe_allow_html=True,
)

