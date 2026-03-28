from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import folium
import numpy as np
import streamlit as st
from PIL import Image
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

from inference import run_inference
from router import apply_damage_cost, find_routes, load_graph, recommend_demo_route_points, recommend_operations_base, snap_points
from utils.geospatial import image_bounds_for_folium
from utils.metadata import parse_metadata
from utils.visualization import colorize_damage_presence, colorize_mask, overlay_mask

DEMO_DIR = Path(__file__).resolve().parent / "data" / "demo"
DEMO_PRE = DEMO_DIR / "pre_disaster.png"
DEMO_POST = DEMO_DIR / "post_disaster.png"
DEMO_METADATA = DEMO_DIR / "metadata.json"
DEMO_REMOTE_BASE = "https://raw.githubusercontent.com/Bariskaya07/disaster-routing/main/data/demo"
DEMO_LIBRARY_DIR = Path(__file__).resolve().parent / "data" / "demo_library"
DEMO_LIBRARY_MANIFEST = DEMO_LIBRARY_DIR / "manifest.json"

st.set_page_config(
    page_title="TUA Taktiksel Tahliye Sistemi",
    page_icon="🛰️",
    layout="wide",
)


def _read_uploaded_image(upload) -> np.ndarray | None:
    if upload is None:
        return None
    with Image.open(BytesIO(upload.getvalue())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_image_path(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_metadata_input(upload, text_value: str) -> dict:
    if upload is not None:
        return parse_metadata(upload.getvalue())
    if text_value.strip():
        return parse_metadata(text_value)
    return {}


def _default_point(metadata: dict, key: str, fallback: tuple[float, float]) -> tuple[float, float]:
    point = metadata.get(key)
    if point is None:
        return fallback
    return float(point[0]), float(point[1])


def _download_demo_asset_if_missing(path: Path, remote_name: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    remote_url = f"{DEMO_REMOTE_BASE}/{remote_name}"
    try:
        with urlopen(remote_url, timeout=20) as response:
            path.write_bytes(response.read())
    except URLError as error:
        raise FileNotFoundError(f"Demo dosyası indirilemedi: {remote_url}") from error


def _ensure_local_demo_assets() -> None:
    if DEMO_METADATA.exists() and DEMO_PRE.exists() and DEMO_POST.exists():
        return
    if not DEMO_METADATA.exists():
        raise FileNotFoundError(
            "Demo metadata dosyası eksik. 'data/demo/metadata.json' bulunamadı."
        )
    _download_demo_asset_if_missing(DEMO_PRE, "pre_disaster.png")
    _download_demo_asset_if_missing(DEMO_POST, "post_disaster.png")


def _demo_assets_available() -> bool:
    return DEMO_PRE.exists() and DEMO_POST.exists() and DEMO_METADATA.exists()


@st.cache_data(show_spinner=False)
def _load_demo_manifest() -> dict:
    if not DEMO_LIBRARY_MANIFEST.exists():
        return {}
    return json.loads(DEMO_LIBRARY_MANIFEST.read_text(encoding="utf-8"))


def _demo_library_scenes() -> list[dict]:
    manifest = _load_demo_manifest()
    scenes = manifest.get("scenes", [])
    if not isinstance(scenes, list):
        return []
    return [dict(scene) for scene in scenes if isinstance(scene, dict)]


def _demo_library_available() -> bool:
    return len(_demo_library_scenes()) > 0


def _format_disaster_name(disaster: str) -> str:
    return disaster.replace("-", " ").title()


def _demo_scene_dir(scene: dict) -> Path:
    relative_dir = scene.get("relative_dir")
    if relative_dir:
        return DEMO_LIBRARY_DIR / str(relative_dir)
    return DEMO_LIBRARY_DIR / str(scene.get("disaster", "")) / str(scene.get("scene_id", ""))


def _demo_scene_assets_available(scene: dict) -> bool:
    scene_dir = _demo_scene_dir(scene)
    return (
        (scene_dir / "pre_disaster.png").exists()
        and (scene_dir / "post_disaster.png").exists()
        and (scene_dir / "metadata.json").exists()
    )


def _selected_demo_metadata(scene: dict | None) -> dict:
    if scene is None:
        if not DEMO_METADATA.exists():
            raise FileNotFoundError(
                "Demo metadata dosyası eksik. 'data/demo/metadata.json' bulunamadı."
            )
        return parse_metadata(DEMO_METADATA)

    scene_dir = _demo_scene_dir(scene)
    metadata_path = scene_dir / "metadata.json"
    if metadata_path.exists():
        metadata = parse_metadata(metadata_path)
    else:
        metadata = parse_metadata(
            {
                "bbox": scene.get("bbox"),
                "start": scene.get("start"),
                "goal": scene.get("goal"),
                "scene_id": scene.get("scene_id"),
                "disaster": scene.get("disaster"),
            }
        )
    metadata.setdefault("scene_id", scene.get("scene_id"))
    metadata.setdefault("disaster", scene.get("disaster"))
    return metadata


def _load_demo_assets(scene: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    if scene is None:
        _ensure_local_demo_assets()
        return _read_image_path(DEMO_PRE), _read_image_path(DEMO_POST), _selected_demo_metadata(None)

    if not _demo_scene_assets_available(scene):
        raise FileNotFoundError(
            f"Seçili demo sahnesi eksik: {_demo_scene_dir(scene)} altında pre_disaster.png, post_disaster.png veya metadata.json bulunamadı."
        )
    scene_dir = _demo_scene_dir(scene)
    return (
        _read_image_path(scene_dir / "pre_disaster.png"),
        _read_image_path(scene_dir / "post_disaster.png"),
        _selected_demo_metadata(scene),
    )


def _format_scene_option(scene: dict) -> str:
    scene_suffix = str(scene.get("scene_id", "")).split("_")[-1]
    score = int(scene.get("score", 0))
    destroyed = int(scene.get("destroyed", 0))
    major = int(scene.get("major", 0))
    return f"{scene_suffix} | Skor {score} | Yıkık {destroyed} | Ağır {major}"


def _estimate_water_exclusion_mask(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    building_mask: np.ndarray,
) -> np.ndarray:
    def _border_connected(mask: np.ndarray) -> np.ndarray:
        binary = mask.astype(np.uint8)
        if binary.max() == 0:
            return np.zeros_like(binary, dtype=bool)
        _, labels = cv2.connectedComponents(binary)
        border_labels = np.unique(
            np.concatenate(
                [
                    labels[0, :],
                    labels[-1, :],
                    labels[:, 0],
                    labels[:, -1],
                ]
            )
        )
        border_labels = border_labels[border_labels != 0]
        if border_labels.size == 0:
            return np.zeros_like(binary, dtype=bool)
        return np.isin(labels, border_labels)

    def _water_from_rgb(image_rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hue = hsv[..., 0]
        sat = hsv[..., 1]
        val = hsv[..., 2]
        rgb = image_rgb.astype(np.int16)
        red = rgb[..., 0]
        green = rgb[..., 1]
        blue = rgb[..., 2]

        blue_cyan_hue = (hue >= 68) & (hue <= 128)
        water_like_color = ((green >= red + 8) & (blue >= red - 6)) | ((blue >= red + 6) & (green >= red))
        not_too_bright = val <= 205
        enough_saturation = sat >= 18
        return blue_cyan_hue & water_like_color & not_too_bright & enough_saturation

    persistent_water = _water_from_rgb(pre_image) & _water_from_rgb(post_image) & (building_mask == 0)
    border_water = _border_connected(persistent_water)
    water_mask = border_water.astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    shoreline_buffer = cv2.dilate(water_mask, np.ones((15, 15), dtype=np.uint8), iterations=3)
    return shoreline_buffer > 0


def _add_legend(fmap: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 6px 20px rgba(15,23,42,0.18);
        font-size: 13px;
        line-height: 1.45;
        min-width: 250px;
    ">
      <div style="font-weight:700; margin-bottom:8px;">Harita Açıklaması</div>
      <div style="margin-bottom:6px;">
        <span style="display:inline-block; width:30px; height:0; border-top:4px dashed #dc2626; margin-right:8px; vertical-align:middle;"></span>
        Afet Öncesi Referans En Kısa Yol
      </div>
      <div style="margin-bottom:8px;">
        <span style="display:inline-block; width:30px; height:0; border-top:6px solid #16a34a; margin-right:8px; vertical-align:middle;"></span>
        Afet Sonrası Güvenli Tahliye Koridoru
      </div>
      <div style="margin-bottom:2px;">
        <span style="display:inline-block; width:30px; height:12px; border-radius:999px; background:#8b5cf6; margin-right:8px; vertical-align:middle;"></span>
        Hasarlı / Yıkık Yapılar
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))


@st.cache_resource(show_spinner=False)
def _cached_graph(west: float, south: float, east: float, north: float):
    return load_graph((west, south, east, north))


def _build_map(artifacts, metadata: dict, route_bundle=None, snap_result=None, operations_base=None):
    bbox = metadata.get("bbox")
    if bbox:
        west, south, east, north = bbox
        center = [(south + north) / 2.0, (west + east) / 2.0]
        zoom_start = 15
    else:
        center = [39.0, 35.0]
        zoom_start = 6

    fmap = folium.Map(location=center, zoom_start=zoom_start, control_scale=True, tiles="CartoDB positron")

    if bbox:
        west, south, east, north = bbox
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#0f172a",
            weight=2,
            fill=False,
            tooltip="Operasyon Sahası Sınırı",
        ).add_to(fmap)

    if artifacts.georef is not None:
        overlay = colorize_damage_presence(artifacts.damage_mask)
        ImageOverlay(
            image=overlay,
            bounds=image_bounds_for_folium(artifacts.georef),
            name="Hasarlı / Yıkık Yapılar",
            opacity=0.85,
            interactive=False,
            cross_origin=False,
            zindex=2,
        ).add_to(fmap)

    if route_bundle is not None:
        same_route = route_bundle.shortest_nodes == route_bundle.safest_nodes
        folium.PolyLine(
            route_bundle.shortest_geometry,
            color="#dc2626",
            weight=10 if same_route else 5,
            opacity=0.90 if same_route else 0.85,
            dash_array="10",
            tooltip="Afet Öncesi Referans En Kısa Yol",
        ).add_to(fmap)
        folium.PolyLine(
            route_bundle.safest_geometry,
            color="#16a34a",
            weight=6 if same_route else 6,
            opacity=0.92,
            tooltip="Afet Sonrası Güvenli Tahliye Koridoru",
        ).add_to(fmap)

    start_marker = snap_result.snapped_start if snap_result is not None and snap_result.snapped_start is not None else metadata.get("start")
    goal_marker = snap_result.snapped_goal if snap_result is not None and snap_result.snapped_goal is not None else metadata.get("goal")
    if start_marker:
        folium.Marker(start_marker, tooltip="Başlangıç", icon=folium.Icon(color="green")).add_to(fmap)
    if goal_marker:
        folium.Marker(goal_marker, tooltip="Hedef", icon=folium.Icon(color="black")).add_to(fmap)
    if operations_base is not None:
        folium.Marker(
            [operations_base.lat, operations_base.lon],
            tooltip="Önerilen Operasyon Üssü / Triage Noktası",
            icon=folium.Icon(color="blue", icon="plus-sign"),
        ).add_to(fmap)

    _add_legend(fmap)
    return fmap


def _render_analysis(artifacts, metadata: dict, route_bundle=None, snap_result=None, operations_base=None, warnings=None) -> None:
    warnings = list(warnings or [])
    heavy_pixels = int(artifacts.heavy_damage_mask.sum())
    building_pixels = int(artifacts.building_mask.sum())
    total_pixels = int(np.prod(artifacts.heavy_damage_mask.shape))
    heavy_ratio = (heavy_pixels / building_pixels * 100.0) if building_pixels else 0.0
    building_coverage = (building_pixels / total_pixels * 100.0) if total_pixels else 0.0
    if heavy_ratio >= 60.0:
        risk_level = "Yüksek"
    elif heavy_ratio >= 25.0:
        risk_level = "Orta"
    else:
        risk_level = "Düşük"

    building_overlay = overlay_mask(
        artifacts.pre_image,
        np.stack([artifacts.building_mask * 255] * 3, axis=-1),
        alpha=0.35,
    )
    damage_overlay = overlay_mask(
        artifacts.post_image,
        colorize_mask(artifacts.damage_mask, artifacts.building_mask),
        alpha=0.50,
    )
    extra_distance_pct = 0.0
    hazard_reduction = 0.0
    if route_bundle is not None and snap_result is not None:
        if route_bundle.shortest_length_m > 0:
            extra_distance_pct = (route_bundle.safest_length_m / route_bundle.shortest_length_m - 1.0) * 100.0
        if route_bundle.shortest_cost > 0:
            hazard_reduction = max(0.0, (1.0 - route_bundle.safest_cost / route_bundle.shortest_cost) * 100.0)

    main_warnings: list[str] = []
    if route_bundle is None:
        main_warnings = warnings
    elif any("Rota planlama" in warning for warning in warnings):
        main_warnings = warnings

    left_col, right_col = st.columns([1.0, 1.2], gap="large")

    with left_col:
        st.subheader("Uydu Görüntüleri ve YZ Analizi")
        top_left, top_right = st.columns(2)
        bottom_left, bottom_right = st.columns(2)
        top_left.image(artifacts.pre_image, caption="Afet Öncesi", use_container_width=True)
        top_right.image(artifacts.post_image, caption="Afet Sonrası", use_container_width=True)
        bottom_left.image(building_overlay, caption="Yapay Zeka: Yapı Tespiti", use_container_width=True)
        bottom_right.image(
            damage_overlay,
            caption="Yapay Zeka: Hasar Durumu (Yeşil=Ayakta, Sarı/Turuncu=Hasarlı, Kırmızı=Yıkık)",
            use_container_width=True,
        )

        scene_metrics = st.columns(3)
        scene_metrics[0].metric(
            "Ağır / Yıkık Yapı Oranı",
            f"%{heavy_ratio:.1f}",
            help="Yalnızca tespit edilen yapı alanı içinde ağır hasarlı veya yıkılmış görünen bölüm.",
        )
        scene_metrics[1].metric("Saha Hasar Seviyesi", risk_level)
        scene_metrics[2].metric("Model Durumu", "Gerçek Model" if not artifacts.used_fallback else "Yedek Tahmin")

        support_metrics = st.columns(2)
        support_metrics[0].metric(
            "Tespit Edilen Yapı Alanı",
            f"%{building_coverage:.1f}",
            help="Tüm görüntünün ne kadarının yapı olarak tespit edildiği.",
        )
        support_metrics[1].metric(
            "Ağır Hasarlı Piksel",
            f"{heavy_pixels:,}".replace(",", "."),
            help="Ağır hasarlı veya yıkık sınıfına düşen yapı pikselleri.",
        )

    with right_col:
        st.subheader("Taktiksel Operasyon Haritası")
        st.caption("Kırmızı kesikli çizgi afet öncesi referans en kısa hattı gösterir. Yeşil çizgi afet sonrası önerilen güvenli koridordur. Mor alanlar ise hasarlı veya yıkık yapı kümelerini gösterir.")
        fmap = _build_map(
            artifacts,
            metadata,
            route_bundle=route_bundle,
            snap_result=snap_result,
            operations_base=operations_base,
        )
        st_folium(fmap, use_container_width=True, height=760)
        if route_bundle is not None and snap_result is not None:
            st.markdown("**Sistem Kararı**")
            summary_container = st.container()
            with summary_container:
                if route_bundle.safest_nodes == route_bundle.shortest_nodes or hazard_reduction < 1.0:
                    st.info("Bu sahnede afet öncesi referans en kısa yol ile afet sonrası güvenli rota aynı koridordur. Kırmızı kesikli çizgi görünür olsun diye yeşil rotanın altında taban çizgisi olarak gösterilmektedir.")
                else:
                    st.success(
                        f"Afet öncesi referans en kısa yola kıyasla, afet sonrası güvenli koridor hasarlı yapı kümelerine maruziyeti yaklaşık %{hazard_reduction:.0f} azalttı. Ek mesafe %{max(extra_distance_pct, 0.0):.0f}."
                    )

            route_row_1 = st.columns(2)
            route_row_1[0].metric(
                "Afet Sonrası Güvenli Rota",
                f"{route_bundle.safest_length_m:.0f} m",
                help="Afet sonrası hasarlı yapı katmanı işlendiğinde sistemin önerdiği gidilebilir koridor.",
            )
            route_row_1[1].metric(
                "Afet Öncesi Referans Rota",
                f"{route_bundle.shortest_length_m:.0f} m",
                help="Hasar etkisi hesaba katılmadan yalnızca mesafeye göre çıkan referans en kısa hat.",
            )
            if route_bundle.shortest_nodes == route_bundle.safest_nodes:
                route_row_1[1].caption("Kırmızı kesikli hat, afet öncesi referansı göstermek için aynı güzergah üzerinde altta çizilir.")
            else:
                route_row_1[1].caption("Kırmızı kesikli çizgi, afet öncesi referans durumda en kısa olan hattır.")

            route_row_2 = st.columns(2)
            route_row_2[0].metric(
                "Güvenlik İçin Fazladan Yol",
                f"{extra_distance_pct:+.0f}%",
                help="Afet sonrası güvenli rota seçildiğinde, afet öncesi referans en kısa yola göre ne kadar daha fazla yol gidildiği.",
            )
            route_row_2[1].metric(
                "Tehlikeden Uzaklaşma Kazancı",
                f"{hazard_reduction:.0f}%",
                help="Afet öncesi referans en kısa yol yerine güvenli koridor seçildiğinde hasarlı yapı kümelerinden uzaklaşma avantajı.",
            )
            explanation_col_1, explanation_col_2 = st.columns(2)
            explanation_col_1.caption("Afet sonrası güvenli koridor seçildiğinde, afet öncesi referans hatta göre ek mesafe ne kadar arttı.")
            explanation_col_2.caption("Afet öncesi referans en kısa hatta göre, hasarlı yapı kümelerinden uzaklaşınca elde edilen güvenlik kazancı.")

        if operations_base is not None:
            st.markdown("**Önerilen Operasyon Üssü**")
            ops_cols = st.columns(2)
            ops_cols[0].metric(
                "Yol Erişimi",
                f"{operations_base.road_snap_m:.0f} m",
                help="Önerilen üssün en yakın araç yoluna olan yaklaşık mesafesi.",
            )
            ops_cols[1].metric(
                "Açık Alan Büyüklüğü",
                f"{operations_base.zone_area_px:,}".replace(",", "."),
                help="Bina dışı ve düşük riskli uygun alanın yaklaşık piksel büyüklüğü.",
            )
            road_label = operations_base.road_class.replace("_", " ").title()
            st.info(
                "Sistem, deniz ve su alanlarını dışlayıp bina dışı uygun açık alanları tarayarak mavi işaretçiyle bir operasyon üssü adayı öneriyor. "
                f"Bu aday {road_label} bağlantısına yakın, erişimi kolay ve çevresindeki hasarlı yapı yoğunluğu daha düşük bir boş alanı temsil ediyor."
            )
        elif artifacts.georef is not None:
            st.caption("Bu sahnede su alanları hariç tutulduğunda, yol erişimi iyi ve yeterince geniş bir açık operasyon alanı otomatik seçilemedi.")

        if main_warnings:
            for warning in main_warnings:
                st.warning(warning)

    with st.expander("Teknik Ayrıntılar", expanded=False):
        st.json(
            {
                "metadata": metadata,
                "used_fallback": artifacts.used_fallback,
                "device": artifacts.device,
                "heavy_damage_pixels": heavy_pixels,
                "building_pixels": building_pixels,
                "heavy_damage_ratio_over_buildings": heavy_ratio,
                "building_coverage_ratio_over_image": building_coverage,
                "warnings": warnings,
                "start_snap_m": snap_result.start_snap_m if snap_result is not None else None,
                "goal_snap_m": snap_result.goal_snap_m if snap_result is not None else None,
                "operations_base": {
                    "lat": operations_base.lat,
                    "lon": operations_base.lon,
                    "support_score": operations_base.support_score,
                    "zone_area_px": operations_base.zone_area_px,
                    "road_snap_m": operations_base.road_snap_m,
                    "road_class": operations_base.road_class,
                    "damage_clearance": operations_base.damage_clearance,
                }
                if operations_base is not None
                else None,
                "shortest_length_m": route_bundle.shortest_length_m if route_bundle is not None else None,
                "safest_length_m": route_bundle.safest_length_m if route_bundle is not None else None,
                "shortest_cost": route_bundle.shortest_cost if route_bundle is not None else None,
                "safest_cost": route_bundle.safest_cost if route_bundle is not None else None,
                "localization_weights": str(artifacts.localization_weights) if artifacts.localization_weights else None,
                "damage_weights": str(artifacts.damage_weights) if artifacts.damage_weights else None,
            }
        )


if "use_demo_assets" not in st.session_state:
    st.session_state["use_demo_assets"] = True
if "analysis_state" not in st.session_state:
    st.session_state["analysis_state"] = None
if "selected_demo_disaster" not in st.session_state:
    st.session_state["selected_demo_disaster"] = ""
if "selected_demo_scene_id" not in st.session_state:
    st.session_state["selected_demo_scene_id"] = ""
if "demo_route_seed_scene_id" not in st.session_state:
    st.session_state["demo_route_seed_scene_id"] = None
if "pending_route_points" not in st.session_state:
    st.session_state["pending_route_points"] = None

st.title("TUA Afet Yönetimi: Otonom Taktiksel Tahliye Sistemi")
st.caption(
    "Uydu görüntüsünden hasarlı yapı çıkarımı ve güvenli tahliye koridoru planlama tek operatör ekranında birleşir."
)

demo_scenes = _demo_library_scenes()
scenes_by_disaster: dict[str, list[dict]] = {}
for scene in demo_scenes:
    disaster = str(scene.get("disaster", "diger"))
    scenes_by_disaster.setdefault(disaster, []).append(scene)

demo_disaster_options = list(scenes_by_disaster.keys())
selected_demo_scene = None
if demo_disaster_options:
    if st.session_state["selected_demo_disaster"] not in scenes_by_disaster:
        st.session_state["selected_demo_disaster"] = demo_disaster_options[0]
    current_demo_scenes = scenes_by_disaster[st.session_state["selected_demo_disaster"]]
    current_demo_scene_ids = [str(scene.get("scene_id", "")) for scene in current_demo_scenes]
    if st.session_state["selected_demo_scene_id"] not in current_demo_scene_ids:
        st.session_state["selected_demo_scene_id"] = current_demo_scene_ids[0]
    selected_demo_scene = next(
        (scene for scene in current_demo_scenes if str(scene.get("scene_id", "")) == st.session_state["selected_demo_scene_id"]),
        current_demo_scenes[0],
    )

with st.sidebar:
    st.header("Girdiler")
    if demo_scenes:
        st.caption(f"Demo sahne kütüphanesi: Hazır ({len(demo_scenes)} sahne)")
    else:
        st.caption(f"Demo veri seti durumu: {'Hazır' if _demo_assets_available() else 'Eksik'}")

    if demo_disaster_options:
        st.selectbox(
            "Demo Afet Tipi",
            options=demo_disaster_options,
            key="selected_demo_disaster",
            format_func=_format_disaster_name,
        )
        current_demo_scenes = scenes_by_disaster[st.session_state["selected_demo_disaster"]]
        current_demo_scene_ids = [str(scene.get("scene_id", "")) for scene in current_demo_scenes]
        if st.session_state["selected_demo_scene_id"] not in current_demo_scene_ids:
            st.session_state["selected_demo_scene_id"] = current_demo_scene_ids[0]
        scene_lookup = {str(scene.get("scene_id", "")): scene for scene in current_demo_scenes}
        st.selectbox(
            "Demo Sahnesi",
            options=current_demo_scene_ids,
            key="selected_demo_scene_id",
            format_func=lambda scene_id: _format_scene_option(scene_lookup[scene_id]),
        )
        selected_demo_scene = scene_lookup[st.session_state["selected_demo_scene_id"]]
        st.caption(
            "Seçili sahne: "
            f"{selected_demo_scene['scene_id']} | "
            f"Yıkık: {int(selected_demo_scene.get('destroyed', 0))} | "
            f"Ağır: {int(selected_demo_scene.get('major', 0))} | "
            f"Az: {int(selected_demo_scene.get('minor', 0))} | "
            f"Sağlam: {int(selected_demo_scene.get('no_damage', 0))}"
        )

    if st.button("Demo Yükle", use_container_width=True):
        st.session_state["use_demo_assets"] = True
    pre_upload = None
    post_upload = None
    metadata_upload = None
    metadata_text = ""

    with st.expander("Manuel Yükleme", expanded=False):
        st.caption("Demo yerine kendi görüntü ve meta verinizi kullanmak isterseniz bu alanı açın.")
        if st.button("Elle Yükle", use_container_width=True, key="manual_upload_button"):
            st.session_state["use_demo_assets"] = False

        pre_upload = st.file_uploader("Afet Öncesi Görüntü", type=["png", "jpg", "jpeg"])
        post_upload = st.file_uploader("Afet Sonrası Görüntü", type=["png", "jpg", "jpeg"])
        metadata_upload = st.file_uploader("Meta Veri JSON", type=["json"])
        metadata_text = st.text_area(
            "Meta Veri JSON (İsteğe Bağlı Yapıştır)",
            value="",
            placeholder='{"bbox": [32.80, 39.85, 32.81, 39.86], "start": [39.851, 32.801], "goal": [39.859, 32.809]}',
            height=160,
        )

try:
    if st.session_state["use_demo_assets"]:
        metadata = _selected_demo_metadata(selected_demo_scene)
    else:
        metadata = _read_metadata_input(metadata_upload, metadata_text)
except Exception as exc:
    metadata = {}
    st.sidebar.error(f"Meta veri çözümlenemedi: {exc}")

bbox = metadata.get("bbox")
default_start = _default_point(metadata, "start", (bbox[1], bbox[0]) if bbox else (39.0, 35.0))
default_goal = _default_point(metadata, "goal", (bbox[3], bbox[2]) if bbox else (39.01, 35.01))

if "start_lat" not in st.session_state:
    st.session_state["start_lat"] = float(default_start[0])
if "start_lon" not in st.session_state:
    st.session_state["start_lon"] = float(default_start[1])
if "goal_lat" not in st.session_state:
    st.session_state["goal_lat"] = float(default_goal[0])
if "goal_lon" not in st.session_state:
    st.session_state["goal_lon"] = float(default_goal[1])

pending_route_points = st.session_state.get("pending_route_points")
if pending_route_points is not None:
    pending_start = pending_route_points["start"]
    pending_goal = pending_route_points["goal"]
    st.session_state["start_lat"] = float(pending_start[0])
    st.session_state["start_lon"] = float(pending_start[1])
    st.session_state["goal_lat"] = float(pending_goal[0])
    st.session_state["goal_lon"] = float(pending_goal[1])
    st.session_state["pending_route_points"] = None

selected_demo_scene_key = str(selected_demo_scene.get("scene_id")) if selected_demo_scene is not None else "fixed-demo"
if st.session_state["use_demo_assets"] and st.session_state.get("demo_route_seed_scene_id") != selected_demo_scene_key:
    st.session_state["start_lat"] = float(default_start[0])
    st.session_state["start_lon"] = float(default_start[1])
    st.session_state["goal_lat"] = float(default_goal[0])
    st.session_state["goal_lon"] = float(default_goal[1])
    st.session_state["demo_route_seed_scene_id"] = selected_demo_scene_key
elif not st.session_state["use_demo_assets"]:
    st.session_state["demo_route_seed_scene_id"] = None

with st.sidebar:
    st.subheader("Rota Koordinatları")
    start_lat = st.number_input("Başlangıç Enlemi", format="%.6f", key="start_lat")
    start_lon = st.number_input("Başlangıç Boylamı", format="%.6f", key="start_lon")
    goal_lat = st.number_input("Hedef Enlemi", format="%.6f", key="goal_lat")
    goal_lon = st.number_input("Hedef Boylamı", format="%.6f", key="goal_lon")
    run_button = st.button("Analizi Başlat", type="primary", use_container_width=True)

if run_button:
    if st.session_state["use_demo_assets"]:
        try:
            pre_image, post_image, demo_metadata = _load_demo_assets(selected_demo_scene)
            metadata = dict(demo_metadata)
        except Exception as exc:
            st.error(f"Demo veri seti yüklenemedi: {exc}")
            st.stop()
    else:
        if pre_upload is None or post_upload is None:
            st.error("Afet öncesi ve afet sonrası görüntüler birlikte yüklenmelidir.")
            st.stop()
        pre_image = _read_uploaded_image(pre_upload)
        post_image = _read_uploaded_image(post_upload)

    metadata = dict(metadata)
    metadata["start"] = (float(start_lat), float(start_lon))
    metadata["goal"] = (float(goal_lat), float(goal_lon))

    with st.spinner("Çift aşamalı afet çıkarımı çalıştırılıyor..."):
        artifacts = run_inference(pre_image, post_image, metadata)

    warnings = list(artifacts.warnings)
    route_bundle = None
    snap_result = None
    operations_base = None

    if artifacts.georef is not None and bbox:
        with st.spinner("Yol ağı çekiliyor ve taktiksel rota hesaplanıyor..."):
            try:
                graph_bundle = _cached_graph(*bbox)
                baseline_graph = graph_bundle.graph_proj
                if st.session_state["use_demo_assets"] and selected_demo_scene is not None:
                    recommended_points = recommend_demo_route_points(
                        baseline_graph,
                        graph_bundle.graph_latlon,
                    )
                    if recommended_points is not None:
                        metadata["start"], metadata["goal"] = recommended_points
                        st.session_state["pending_route_points"] = {
                            "start": recommended_points[0],
                            "goal": recommended_points[1],
                        }
                        warnings.append(
                            "Demo sahnesindeki başlangıç ve hedef noktaları, yol ağına daha anlamlı otursun diye otomatik olarak gerçek yol düğümlerinden seçildi."
                        )
                water_exclusion_mask = _estimate_water_exclusion_mask(
                    artifacts.pre_image,
                    artifacts.post_image,
                    artifacts.building_mask,
                )
                operations_base = recommend_operations_base(
                    baseline_graph,
                    artifacts.georef,
                    artifacts.building_mask,
                    artifacts.damage_raster.array,
                    exclusion_mask=water_exclusion_mask,
                )
                damage_graph = apply_damage_cost(baseline_graph, artifacts.damage_raster)
                snap_result = snap_points(damage_graph, metadata["start"], metadata["goal"])
                warnings.extend(snap_result.warnings)
                route_bundle = find_routes(
                    damage_graph,
                    snap_result.start_node,
                    snap_result.goal_node,
                    baseline_graph_proj=baseline_graph,
                    graph_latlon=graph_bundle.graph_latlon,
                )
            except Exception as exc:
                warnings.append(f"Rota planlama şu an kullanılamıyor: {exc}")
    else:
        warnings.append("BBox veya coğrafi hizalama meta verisi olmadığı için rota planlama kapatıldı.")

    st.session_state["analysis_state"] = {
        "artifacts": artifacts,
        "metadata": metadata,
        "route_bundle": route_bundle,
        "snap_result": snap_result,
        "operations_base": operations_base,
        "warnings": warnings,
    }
    if st.session_state.get("pending_route_points") is not None:
        st.rerun()

analysis_state = st.session_state.get("analysis_state")
if analysis_state is not None:
    _render_analysis(
        analysis_state["artifacts"],
        analysis_state["metadata"],
        route_bundle=analysis_state.get("route_bundle"),
        snap_result=analysis_state.get("snap_result"),
        operations_base=analysis_state.get("operations_base"),
        warnings=analysis_state.get("warnings"),
    )
else:
    st.info("Görüntüleri yükleyin veya hazır demoyu seçin, ardından analizi başlatın.")
