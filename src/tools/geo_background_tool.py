from __future__ import annotations

import json
import math
import os
import ssl
import time
from io import BytesIO
import zipfile
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

def _http_get_json(url: str, params: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    query = parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query}" if query else url
    req = request.Request(
        full_url,
        headers={
            "User-Agent": os.getenv(
                "GEO_BACKGROUND_USER_AGENT",
                "agent-landslide/0.1 (geo-background-tool)",
            ),
            "Accept": "application/json",
        },
        method="GET",
    )

    context = None
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = None

    retries = max(1, int(os.getenv("GEO_BACKGROUND_HTTP_RETRIES", "3")))
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=timeout, context=context) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urlerror.URLError as exc:
            last_exc = exc
            # SSL EOF is often transient on public APIs; retry with backoff.
            if attempt < retries:
                time.sleep(min(1.5, 0.25 * attempt))
                continue
            break
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(1.5, 0.25 * attempt))
                continue
            break

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("http request failed without exception")


def _http_get_bytes(url: str, params: dict[str, Any], timeout: float = 10.0, accept: str = "*/*") -> bytes:
    query = parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query}" if query else url
    req = request.Request(
        full_url,
        headers={
            "User-Agent": os.getenv(
                "GEO_BACKGROUND_USER_AGENT",
                "agent-landslide/0.1 (geo-background-tool)",
            ),
            "Accept": accept,
        },
        method="GET",
    )

    context = None
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = None

    retries = max(1, int(os.getenv("GEO_BACKGROUND_HTTP_RETRIES", "3")))
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=timeout, context=context) as resp:
                return resp.read()
        except urlerror.URLError as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(1.5, 0.25 * attempt))
                continue
            break
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(1.5, 0.25 * attempt))
                continue
            break

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("http request failed without exception")


def _looks_like_tiff(data: bytes) -> bool:
    if len(data) < 4:
        return False
    return data[:4] in (b"II*\x00", b"MM\x00*")


def _extract_raster_payload(raw_bytes: bytes) -> tuple[bytes | None, str]:
    if not raw_bytes:
        return None, "empty raster response body"

    if _looks_like_tiff(raw_bytes):
        return raw_bytes, ""

    # Some providers may return a zip payload with a GeoTIFF inside.
    if raw_bytes[:2] == b"PK":
        try:
            with zipfile.ZipFile(BytesIO(raw_bytes)) as zf:
                for name in zf.namelist():
                    lower = name.lower()
                    if lower.endswith(".tif") or lower.endswith(".tiff"):
                        return zf.read(name), ""
            return None, "zip payload has no .tif/.tiff file"
        except Exception as exc:
            return None, f"zip payload parse failed: {exc}"

    # Often the API returns a text/json error body while HTTP status is still 200.
    snippet = raw_bytes[:320].decode("utf-8", errors="ignore").strip().replace("\n", " ")
    if snippet:
        return None, f"non-raster payload: {snippet[:220]}"
    return None, f"non-raster payload head={raw_bytes[:16]!r}"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _query_nominatim_reverse(lat: float, lon: float) -> dict[str, Any]:
    data = _http_get_json(
        "https://nominatim.openstreetmap.org/reverse",
        {
            "format": "jsonv2",
            "lat": lat,
            "lon": lon,
            "zoom": 14,
            "addressdetails": 1,
        },
        timeout=float(os.getenv("GEO_BACKGROUND_TIMEOUT", "10")),
    )
    address = data.get("address") if isinstance(data.get("address"), dict) else {}
    return {
        "display_name": str(data.get("display_name", "") or ""),
        "address": {
            "country": address.get("country", ""),
            "state": address.get("state", ""),
            "county": address.get("county", ""),
            "city": address.get("city", "") or address.get("town", "") or address.get("village", ""),
            "road": address.get("road", ""),
            "postcode": address.get("postcode", ""),
        },
        "source": "nominatim",
    }


def _query_open_meteo_elevations(points: list[tuple[float, float]], timeout: float = 10.0) -> list[float | None]:
    if not points:
        return []
    lat_str = ",".join(f"{lat:.8f}" for lat, _ in points)
    lon_str = ",".join(f"{lon:.8f}" for _, lon in points)
    data = _http_get_json(
        "https://api.open-meteo.com/v1/elevation",
        {"latitude": lat_str, "longitude": lon_str},
        timeout=timeout,
    )
    values = data.get("elevation")
    if isinstance(values, list):
        return [_safe_float(v) for v in values]
    if values is None:
        return [None for _ in points]
    one = _safe_float(values)
    return [one for _ in points]


def _opentopography_dem_type() -> str:
    return os.getenv("OPENTOPOGRAPHY_DEMTYPE", "COP30").strip() or "COP30"


def _opentopography_resolution_m(dem_type: str) -> float | None:
    resolution_map = {
        "COP30": 30.0,
        "COP90": 90.0,
        "SRTMGL1": 30.0,
        "SRTMGL3": 90.0,
        "AW3D30": 30.0,
    }
    return resolution_map.get(dem_type.upper())


def _open_meteo_fallback_enabled() -> bool:
    raw = str(os.getenv("GEO_BACKGROUND_ENABLE_OPEN_METEO_FALLBACK", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _summarize_dem_issue(message: str) -> str:
    text = str(message or "").strip()
    lower = text.lower()
    if "empty raster response body" in lower:
        return "empty raster response body"
    if "non-raster payload" in lower:
        return "non-raster payload returned"
    if "timeout" in lower:
        return "request timeout"
    if not text:
        return "unknown reason"
    return text[:120]


def _query_opentopography_elevations(points: list[tuple[float, float]], timeout: float = 10.0) -> dict[str, Any]:
    api_key = os.getenv("OPENTOPOGRAPHY_API_KEY", "").strip()
    if not api_key:
        return {"elevations": [], "error": "missing OPENTOPOGRAPHY_API_KEY"}
    if not points:
        return {"elevations": [], "dem_type": _opentopography_dem_type(), "resolution_m": _opentopography_resolution_m(_opentopography_dem_type()), "raw": {}}

    dem_type = _opentopography_dem_type()
    resolution_m = _opentopography_resolution_m(dem_type)
    base_url = os.getenv(
        "OPENTOPOGRAPHY_POINT_URL",
        "https://portal.opentopography.org/API/globaldem",
    ).strip()
    padding_m = max(15.0, float(resolution_m or 30.0) / 2.0)
    min_side_m = float(os.getenv("OPENTOPOGRAPHY_MIN_BBOX_METERS", "260"))
    lats = [lat for lat, _ in points]
    lons = [lon for _, lon in points]
    center_lat = sum(lats) / len(lats)
    south = min(lats) - _meters_to_lat_delta(padding_m, center_lat)
    north = max(lats) + _meters_to_lat_delta(padding_m, center_lat)
    west = min(lons) - _meters_to_lon_delta(padding_m, center_lat)
    east = max(lons) + _meters_to_lon_delta(padding_m, center_lat)

    lat_span_m = max(0.0, (north - south) * _meters_per_degree_lat(center_lat))
    lon_span_m = max(0.0, (east - west) * _meters_per_degree_lon(center_lat))
    if lat_span_m < min_side_m:
        extra_m = (min_side_m - lat_span_m) / 2.0
        south -= _meters_to_lat_delta(extra_m, center_lat)
        north += _meters_to_lat_delta(extra_m, center_lat)
    if lon_span_m < min_side_m:
        extra_m = (min_side_m - lon_span_m) / 2.0
        west -= _meters_to_lon_delta(extra_m, center_lat)
        east += _meters_to_lon_delta(extra_m, center_lat)

    try:
        tif_bytes = _http_get_bytes(
            base_url,
            {
                "demtype": dem_type,
                "south": south,
                "north": north,
                "west": west,
                "east": east,
                "outputFormat": "GTiff",
                "API_Key": api_key,
            },
            timeout=max(timeout, 20.0),
        )
    except Exception as exc:
        return {
            "elevations": [None for _ in points],
            "dem_type": dem_type,
            "resolution_m": resolution_m,
            "error": f"opentopography query failed: {exc}",
            "raw": {},
        }

    raster_bytes, payload_error = _extract_raster_payload(tif_bytes)
    if not raster_bytes:
        return {
            "elevations": [None for _ in points],
            "dem_type": dem_type,
            "resolution_m": resolution_m,
            "error": f"opentopography raster payload invalid: {payload_error}",
            "raw": {
                "bbox": {"south": south, "north": north, "west": west, "east": east},
                "content_bytes": len(tif_bytes),
            },
        }

    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(raster_bytes) as memfile:
            with memfile.open(driver="GTiff") as dataset:
                samples = []
                for value in dataset.sample([(lon, lat) for lat, lon in points]):
                    if len(value) == 0:
                        samples.append(None)
                        continue
                    cell = value[0]
                    nodata = dataset.nodata
                    if nodata is not None and float(cell) == float(nodata):
                        samples.append(None)
                    else:
                        samples.append(_safe_float(cell))
                return {
                    "elevations": samples,
                    "dem_type": dem_type,
                    "resolution_m": resolution_m,
                    "raw": {
                        "bbox": {"south": south, "north": north, "west": west, "east": east},
                        "content_bytes": len(raster_bytes),
                        "width": dataset.width,
                        "height": dataset.height,
                    },
                }
    except Exception as exc:
        return {
            "elevations": [None for _ in points],
            "dem_type": dem_type,
            "resolution_m": resolution_m,
            "error": f"opentopography raster parse failed: {exc}",
            "raw": {},
        }


def _query_opentopography_elevation(lat: float, lon: float, timeout: float = 10.0) -> dict[str, Any]:
    sampled = _query_opentopography_elevations([(lat, lon)], timeout=timeout)
    elevations = sampled.get("elevations", [])
    elev = elevations[0] if elevations else None
    return {
        "elevation_m": elev,
        "raw": sampled.get("raw", {}),
        "dem_type": sampled.get("dem_type", _opentopography_dem_type()),
        "resolution_m": sampled.get("resolution_m"),
        "error": sampled.get("error") if elev is None else None,
    }


def _query_dem_opentopography_or_fallback(lat: float, lon: float) -> dict[str, Any]:
    warnings: list[str] = []
    api_key = os.getenv("OPENTOPOGRAPHY_API_KEY", "").strip()
    timeout = float(os.getenv("GEO_BACKGROUND_TIMEOUT", "10"))

    if api_key:
        try:
            ot_data = _query_opentopography_elevation(lat, lon, timeout=timeout)
            elev = ot_data.get("elevation_m")
            if elev is not None:
                return {
                    "elevation_m": elev,
                    "dem_source": f"opentopography:{ot_data.get('dem_type', _opentopography_dem_type())}",
                    "dem_resolution_m": ot_data.get("resolution_m"),
                    "raw": ot_data.get("raw", {}),
                    "warnings": warnings,
                }
            ot_err = str(ot_data.get("error") or "opentopography response parsed but elevation value not found")
            warnings.append(f"30m DEM unavailable ({_summarize_dem_issue(ot_err)})")
        except Exception as exc:
            warnings.append(f"opentopography query failed: {exc}")

    if _open_meteo_fallback_enabled():
        try:
            elevations = _query_open_meteo_elevations([(lat, lon)], timeout=timeout)
            elev = elevations[0] if elevations else None
            if elev is not None:
                return {
                    "elevation_m": elev,
                    "dem_source": "open-meteo-fallback",
                    "dem_resolution_m": 90.0,
                    "raw": {},
                    # Primary 30m DEM failure is suppressed when fallback succeeds.
                    "warnings": [],
                }
            warnings.append("open-meteo fallback returned null elevation")
        except Exception as exc:
            warnings.append(f"fallback elevation query failed: {exc}")

    return {
        "elevation_m": None,
        "dem_source": "",
        "dem_resolution_m": None,
        "raw": {},
        "warnings": warnings,
        "error": "30m DEM query failed",
    }


def _meters_per_degree_lat(lat_deg: float) -> float:
    lat_rad = math.radians(lat_deg)
    return (
        111132.92
        - 559.82 * math.cos(2.0 * lat_rad)
        + 1.175 * math.cos(4.0 * lat_rad)
    )


def _meters_per_degree_lon(lat_deg: float) -> float:
    lat_rad = math.radians(lat_deg)
    return (
        111412.84 * math.cos(lat_rad)
        - 93.5 * math.cos(3.0 * lat_rad)
    )


def _meters_to_lat_delta(distance_m: float, lat_deg: float) -> float:
    return float(distance_m) / max(1.0, _meters_per_degree_lat(lat_deg))


def _meters_to_lon_delta(distance_m: float, lat_deg: float) -> float:
    return float(distance_m) / max(1.0, _meters_per_degree_lon(lat_deg))


def _estimate_slope_aspect(lat: float, lon: float, timeout: float = 10.0) -> dict[str, Any]:
    sample_distance_m = float(os.getenv("GEO_BACKGROUND_SLOPE_SAMPLE_METERS", "30"))
    lat_delta = _meters_to_lat_delta(sample_distance_m, lat)
    lon_delta = _meters_to_lon_delta(sample_distance_m, lat)
    p_center = (lat, lon)
    p_north = (lat + lat_delta, lon)
    p_south = (lat - lat_delta, lon)
    p_east = (lat, lon + lon_delta)
    p_west = (lat, lon - lon_delta)
    points = [p_center, p_north, p_south, p_east, p_west]

    dem_method = "central_difference_opentopography"
    dem_resolution_m: float | None = None
    elevations: list[float | None] = []
    warnings: list[str] = []
    if os.getenv("OPENTOPOGRAPHY_API_KEY", "").strip():
        sampled = _query_opentopography_elevations(points, timeout=timeout)
        elevations = sampled.get("elevations", [])
        dem_type = sampled.get("dem_type", _opentopography_dem_type())
        dem_resolution_m = sampled.get("resolution_m")
        dem_method = f"central_difference_opentopography_{str(dem_type).lower()}"
        if sampled.get("error"):
            ot_err = str(sampled.get("error") or "30m DEM query failed")
            warnings.append(f"30m DEM unavailable ({_summarize_dem_issue(ot_err)})")
    else:
        warnings.append("missing OPENTOPOGRAPHY_API_KEY")

    ot_usable = (
        len(elevations) == len(points)
        and all(v is not None for v in elevations[1:])
    )

    if not ot_usable and _open_meteo_fallback_enabled():
        try:
            fallback = _query_open_meteo_elevations(points, timeout=timeout)
            if len(fallback) == len(points) and all(v is not None for v in fallback[1:]):
                elevations = fallback
                dem_method = "central_difference_open_meteo_fallback"
                dem_resolution_m = 90.0
            else:
                if len(fallback) != len(points):
                    warnings.append("open-meteo fallback returned insufficient DEM neighbor samples")
                else:
                    warnings.append("open-meteo fallback neighbor elevation contains null")
        except Exception as exc:
            warnings.append(f"open-meteo slope fallback failed: {exc}")

    if len(elevations) != len(points):
        return {
            "slope_deg": None,
            "aspect_deg": None,
            "sample_distance_m": sample_distance_m,
            "dem_resolution_m": dem_resolution_m,
            "error": " | ".join(warnings[:3]) if warnings else "insufficient DEM neighbor samples",
        }
    zc, zn, zs, ze, zw = elevations
    if any(v is None for v in (zn, zs, ze, zw)):
        return {
            "slope_deg": None,
            "aspect_deg": None,
            "sample_distance_m": sample_distance_m,
            "dem_resolution_m": dem_resolution_m,
            "error": " | ".join(warnings[:3]) if warnings else "neighbor elevation contains null",
        }

    dx_m = max(1.0, 2.0 * sample_distance_m)
    dy_m = max(1.0, 2.0 * sample_distance_m)
    dzdx = (float(ze) - float(zw)) / dx_m
    dzdy = (float(zn) - float(zs)) / dy_m

    slope_rad = math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy))
    slope_deg = math.degrees(slope_rad)

    east_component = -dzdx
    north_component = -dzdy
    aspect = (math.degrees(math.atan2(east_component, north_component)) + 360.0) % 360.0

    surfaced_warnings = warnings[:3]
    if dem_method == "central_difference_open_meteo_fallback":
        # Fallback succeeded; keep UI clean by not surfacing the upstream 30m failure.
        surfaced_warnings = []
    return {
        "slope_deg": round(slope_deg, 4),
        "aspect_deg": round(aspect, 4),
        "center_elevation_m": _safe_float(zc),
        "method": dem_method,
        "sample_delta_deg": max(lat_delta, lon_delta),
        "sample_distance_m": sample_distance_m,
        "dem_resolution_m": dem_resolution_m,
        "warnings": surfaced_warnings,
    }


def _query_macrostrat_geology(lat: float, lon: float) -> dict[str, Any]:
    timeout = float(os.getenv("GEO_BACKGROUND_TIMEOUT", "10"))
    errors: list[str] = []
    candidate_endpoints = [
        os.getenv(
            "MACROSTRAT_POINT_URL",
            "https://macrostrat.org/api/v2/geologic_units/map",
        ).strip(),
        "https://v2.macrostrat.org/api/v2/geologic_units/map",
    ]
    candidate_params = [
        {"lat": lat, "lng": lon},
        {"lat": lat, "lon": lon},
        {"latitude": lat, "longitude": lon},
    ]

    for endpoint in candidate_endpoints:
        for params in candidate_params:
            try:
                data = _http_get_json(endpoint, params, timeout=timeout)
                records = data.get("data")
                if not isinstance(records, list):
                    success_obj = data.get("success")
                    if isinstance(success_obj, dict):
                        records = success_obj.get("data")
                if isinstance(records, list) and records:
                    unit = records[0] if isinstance(records[0], dict) else {}
                    return {
                        "unit_name": str(unit.get("name", "") or unit.get("strat_name", "") or ""),
                        "lithology": str(unit.get("lith", "") or unit.get("lithology", "") or ""),
                        "description": str(unit.get("descrip", "") or unit.get("description", "") or ""),
                        "age": str(unit.get("age", "") or ""),
                        "source": "macrostrat",
                        "raw": unit,
                    }
                features = data.get("features")
                if isinstance(features, list) and features:
                    props = (features[0] or {}).get("properties", {})
                    if isinstance(props, dict):
                        return {
                            "unit_name": str(props.get("name", "") or props.get("strat_name", "") or ""),
                            "lithology": str(props.get("lith", "") or props.get("lithology", "") or ""),
                            "description": str(props.get("descrip", "") or props.get("description", "") or ""),
                            "age": str(props.get("age", "") or ""),
                            "source": "macrostrat",
                            "raw": props,
                        }
            except Exception as exc:
                errors.append(f"{endpoint} {params}: {exc}")

    return {
        "unit_name": "",
        "lithology": "",
        "description": "",
        "age": "",
        "source": "macrostrat",
        "raw": {},
        "error": "macrostrat query failed",
        "warnings": errors[:3],
    }


def query_geo_background_safe(lat: float, lon: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "observation_point": {"lat": float(lat), "lon": float(lon)},
        "address": {},
        "terrain": {},
        "geology": {},
        "warnings": [],
    }

    try:
        result["address"] = _query_nominatim_reverse(lat, lon)
    except (urlerror.URLError, TimeoutError, ConnectionError) as exc:
        result["address"] = {"source": "nominatim", "error": f"address lookup failed: {exc}"}
    except Exception as exc:
        result["address"] = {"source": "nominatim", "error": f"address lookup failed: {exc}"}

    try:
        terrain = _query_dem_opentopography_or_fallback(lat, lon)
        result["terrain"] = {
            "elevation_m": terrain.get("elevation_m"),
            "dem_source": terrain.get("dem_source", ""),
            "dem_resolution_m": terrain.get("dem_resolution_m"),
        }
        if isinstance(terrain.get("warnings"), list):
            result["warnings"].extend(str(w) for w in terrain["warnings"][:3])
        if terrain.get("error"):
            result["warnings"].append(str(terrain.get("error")))
    except Exception as exc:
        result["terrain"] = {"elevation_m": None, "dem_source": "", "error": f"dem lookup failed: {exc}"}

    try:
        slope_aspect = _estimate_slope_aspect(
            lat,
            lon,
            timeout=float(os.getenv("GEO_BACKGROUND_TIMEOUT", "10")),
        )
        result["terrain"]["slope_deg"] = slope_aspect.get("slope_deg")
        result["terrain"]["aspect_deg"] = slope_aspect.get("aspect_deg")
        result["terrain"]["slope_aspect_method"] = slope_aspect.get("method", "")
        result["terrain"]["sample_delta_deg"] = slope_aspect.get("sample_delta_deg")
        result["terrain"]["sample_distance_m"] = slope_aspect.get("sample_distance_m")
        if slope_aspect.get("dem_resolution_m") is not None and result["terrain"].get("dem_resolution_m") is None:
            result["terrain"]["dem_resolution_m"] = slope_aspect.get("dem_resolution_m")
        if isinstance(slope_aspect.get("warnings"), list):
            result["warnings"].extend(str(w) for w in slope_aspect["warnings"][:2])
        if slope_aspect.get("error"):
            result["warnings"].append(f"slope_aspect_failed: {slope_aspect['error']}")
    except Exception as exc:
        result["terrain"]["slope_deg"] = None
        result["terrain"]["aspect_deg"] = None
        result["warnings"].append(f"slope_aspect_failed: {exc}")

    try:
        geology = _query_macrostrat_geology(lat, lon)
        result["geology"] = {
            "unit_name": geology.get("unit_name", ""),
            "lithology": geology.get("lithology", ""),
            "description": geology.get("description", ""),
            "age": geology.get("age", ""),
            "source": geology.get("source", "macrostrat"),
        }
        if geology.get("error"):
            result["warnings"].append(str(geology["error"]))
        if isinstance(geology.get("warnings"), list):
            result["warnings"].extend(str(w) for w in geology["warnings"][:2])
    except Exception as exc:
        result["geology"] = {"source": "macrostrat", "error": f"geology lookup failed: {exc}"}

    seen = set()
    dedup_warnings = []
    for item in result["warnings"]:
        if item in seen:
            continue
        seen.add(item)
        dedup_warnings.append(item)
    result["warnings"] = dedup_warnings
    return result
