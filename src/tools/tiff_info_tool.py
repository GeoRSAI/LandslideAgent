from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any
from PIL import Image


GEOTIFF_GEO_KEY_DIRECTORY_TAG = 34735
GEOTIFF_GEO_ASCII_PARAMS_TAG = 34737
GEOTIFF_MODEL_PIXEL_SCALE_TAG = 33550
GEOTIFF_GDAL_NODATA_TAG = 42113
TIFF_X_RESOLUTION_TAG = 282
TIFF_Y_RESOLUTION_TAG = 283
TIFF_RESOLUTION_UNIT_TAG = 296


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, Fraction):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_geo_key_directory(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        values = []
        for item in raw:
            try:
                values.append(int(item))
            except (TypeError, ValueError):
                return []
        return values
    return []


def _extract_geotiff_crs_from_tags(tags: dict[int, Any]) -> str | None:
    directory = _normalize_geo_key_directory(tags.get(GEOTIFF_GEO_KEY_DIRECTORY_TAG))
    if len(directory) < 4:
        return None

    key_count = directory[3]
    if key_count <= 0:
        return None

    ascii_params = str(tags.get(GEOTIFF_GEO_ASCII_PARAMS_TAG, "") or "")
    key_entries: dict[int, tuple[int, int, int]] = {}
    start = 4
    for idx in range(key_count):
        pos = start + idx * 4
        if pos + 3 >= len(directory):
            break
        key_id = directory[pos]
        tiff_tag_location = directory[pos + 1]
        count = directory[pos + 2]
        value_offset = directory[pos + 3]
        key_entries[key_id] = (tiff_tag_location, count, value_offset)

    def _read_key(key_id: int) -> str | int | None:
        entry = key_entries.get(key_id)
        if not entry:
            return None
        location, count, offset = entry
        if location == 0:
            return int(offset)
        if location == GEOTIFF_GEO_ASCII_PARAMS_TAG and ascii_params:
            text = ascii_params[offset : offset + max(0, count)]
            return text.replace("|", " ").strip(" \x00")
        return None

    # Prefer explicit EPSG-like keys when available.
    projected_cs = _read_key(3072)  # ProjectedCSTypeGeoKey
    if isinstance(projected_cs, int) and projected_cs not in {0, 32767}:
        return f"EPSG:{projected_cs}"

    geographic_cs = _read_key(2048)  # GeographicTypeGeoKey
    if isinstance(geographic_cs, int) and geographic_cs not in {0, 32767}:
        return f"EPSG:{geographic_cs}"

    for citation_key in (3073, 2049, 1026):
        text = _read_key(citation_key)
        if isinstance(text, str) and text:
            return text
    return None


def _extract_resolution_and_nodata_from_pillow(
    img: Image.Image,
    tags: dict[int, Any],
) -> tuple[dict[str, Any] | None, Any]:
    resolution: dict[str, Any] | None = None
    nodata: Any = None

    model_scale = tags.get(GEOTIFF_MODEL_PIXEL_SCALE_TAG)
    if isinstance(model_scale, (list, tuple)) and len(model_scale) >= 2:
        x_res = _as_float(model_scale[0])
        y_res = _as_float(model_scale[1])
        if x_res is not None and y_res is not None and x_res > 0.0 and y_res > 0.0:
            resolution = {
                "x": x_res,
                "y": y_res,
                "unit": "map_units",
                "source": "model_pixel_scale",
            }

    if resolution is None:
        x_tiff = _as_float(tags.get(TIFF_X_RESOLUTION_TAG))
        y_tiff = _as_float(tags.get(TIFF_Y_RESOLUTION_TAG))
        if x_tiff is not None and y_tiff is not None and x_tiff > 0.0 and y_tiff > 0.0:
            unit_code = int(tags.get(TIFF_RESOLUTION_UNIT_TAG, 2) or 2)
            unit = {1: "none", 2: "inch", 3: "centimeter"}.get(unit_code, "unknown")
            resolution = {
                "x": x_tiff,
                "y": y_tiff,
                "unit": unit,
                "source": "tiff_resolution",
            }

    if resolution is None:
        dpi = img.info.get("dpi")
        if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
            x_dpi = _as_float(dpi[0])
            y_dpi = _as_float(dpi[1])
            if x_dpi is not None and y_dpi is not None and x_dpi > 0.0 and y_dpi > 0.0:
                resolution = {
                    "x": x_dpi,
                    "y": y_dpi,
                    "unit": "dpi",
                    "source": "image_dpi",
                }

    raw_nodata = tags.get(GEOTIFF_GDAL_NODATA_TAG)
    if raw_nodata is not None:
        text = str(raw_nodata).strip()
        as_float = _as_float(text)
        nodata = as_float if as_float is not None else text

    return resolution, nodata


def _try_read_with_rasterio(path: Path) -> tuple[str | None, dict[str, Any] | None, Any] | None:
    suffix = path.suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        return None
    try:
        import rasterio
    except Exception:
        return None

    try:
        with rasterio.open(path) as ds:
            crs_text = ds.crs.to_string() if ds.crs else None
            res = ds.res if isinstance(ds.res, tuple) and len(ds.res) == 2 else None
            resolution = None
            if res:
                x_res = _as_float(res[0])
                y_res = _as_float(res[1])
                if x_res is not None and y_res is not None:
                    resolution = {
                        "x": abs(x_res),
                        "y": abs(y_res),
                        "unit": "map_units",
                        "source": "rasterio",
                    }
            return crs_text, resolution, ds.nodata
    except Exception:
        return None


def read_tiff_info(image_path: str) -> dict:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    crs: str | None = None
    resolution: dict[str, Any] | None = None
    nodata: Any = None
    image_format = ""
    mode = ""

    with Image.open(path) as img:
        width, height = img.size
        bands = len(img.getbands()) if img.getbands() else 0
        image_format = str(img.format or "").upper()
        mode = str(img.mode or "")
        tags = {}
        if hasattr(img, "tag_v2"):
            try:
                tags = {int(k): v for k, v in img.tag_v2.items()}
            except Exception:
                tags = {}

        crs = _extract_geotiff_crs_from_tags(tags)
        resolution, nodata = _extract_resolution_and_nodata_from_pillow(img, tags)

    # Prefer rasterio values when available for GeoTIFF because they are more reliable.
    rio_meta = _try_read_with_rasterio(path)
    if rio_meta is not None:
        rio_crs, rio_resolution, rio_nodata = rio_meta
        crs = rio_crs or crs
        resolution = rio_resolution or resolution
        if rio_nodata is not None:
            nodata = rio_nodata

    return {
        "image_path": str(path.resolve()),
        "exists": True,
        "format": image_format,
        "mode": mode,
        "crs": crs,
        "width": int(width),
        "height": int(height),
        "bands": int(bands),
        "resolution": resolution,
        "nodata": nodata,
    }
