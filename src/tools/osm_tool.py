from __future__ import annotations

import copy
import json
import os
import socket
import ssl
import time
from collections import OrderedDict
from typing import Any
from urllib import error as urlerror
from urllib import parse
from urllib import request


_DEFAULT_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
_OSM_CACHE: "OrderedDict[tuple[float, float, int], tuple[float, dict[str, Any]]]" = OrderedDict()


def _osm_ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _osm_endpoints() -> list[str]:
    raw = str(os.getenv("OVERPASS_API_URLS", "") or "").strip()
    if raw:
        endpoints = [item.strip() for item in raw.split(",") if item.strip()]
        if endpoints:
            return list(dict.fromkeys(endpoints))
    return list(dict.fromkeys(_DEFAULT_OVERPASS_ENDPOINTS))


def _base_result(lat: float, lon: float, radius: int) -> dict[str, Any]:
    return {
        "observation_point": {"lat": float(lat), "lon": float(lon)},
        "radius_m": int(radius),
        "count": 0,
        "features": [],
        "warnings": [],
        "source": "",
        "source_status": "ok",
    }


def _build_overpass_query(lat: float, lon: float, radius: int) -> str:
    query_timeout = int(float(os.getenv("OVERPASS_QUERY_TIMEOUT", "18")))
    road_radius = min(radius, int(float(os.getenv("OVERPASS_ROAD_RADIUS_M", str(radius))) or radius))
    amenity_radius = min(radius, int(float(os.getenv("OVERPASS_AMENITY_RADIUS_M", "800")) or 800))
    building_radius = min(radius, int(float(os.getenv("OVERPASS_BUILDING_RADIUS_M", "250")) or 250))
    return f"""
    [out:json][timeout:{query_timeout}];
    (
      node(around:{radius},{lat},{lon})[place~"^(village|town|city|hamlet|suburb|neighbourhood)$"];
      way(around:{road_radius},{lat},{lon})[highway~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|track)$"];
      node(around:{amenity_radius},{lat},{lon})[amenity~"^(school|hospital|clinic|fire_station|police)$"];
      way(around:{amenity_radius},{lat},{lon})[amenity~"^(school|hospital|clinic|fire_station|police)$"];
      node(around:{building_radius},{lat},{lon})[building][name];
      way(around:{building_radius},{lat},{lon})[building][name];
      relation(around:{building_radius},{lat},{lon})[building][name];
    );
    out center tags qt;
    """


def _fetch_overpass_payload(endpoint: str, overpass_query: str) -> dict[str, Any]:
    encoded_query = parse.urlencode({"data": overpass_query}).encode("utf-8")
    req = request.Request(
        endpoint,
        data=encoded_query,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": "agent-landslide/0.1 (osm-nearby-tool)",
        },
        method="POST",
    )
    ssl_context = _osm_ssl_context()
    http_timeout = float(os.getenv("OVERPASS_HTTP_TIMEOUT", "25"))
    with request.urlopen(req, timeout=http_timeout, context=ssl_context) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_features(payload: dict[str, Any]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for item in payload.get("elements", []):
        tags = item.get("tags") or {}
        item_lat = item.get("lat")
        item_lon = item.get("lon")
        center = item.get("center") or {}
        if item_lat is None:
            item_lat = center.get("lat")
        if item_lon is None:
            item_lon = center.get("lon")
        if item_lat is None or item_lon is None:
            continue

        feature_type = "other"
        if "place" in tags:
            feature_type = "settlement"
        elif "highway" in tags:
            feature_type = "road"
        elif "amenity" in tags:
            feature_type = "amenity"
        elif "building" in tags:
            feature_type = "building"

        features.append(
            {
                "id": f"{item.get('type', 'element')}/{item.get('id', '')}",
                "type": feature_type,
                "subtype": tags.get("place")
                or tags.get("highway")
                or tags.get("amenity")
                or tags.get("building")
                or "",
                "name": tags.get("name", ""),
                "lat": float(item_lat),
                "lon": float(item_lon),
                "tags": tags,
            }
        )
    return features


def _cache_key(lat: float, lon: float, radius: int) -> tuple[float, float, int]:
    return (round(float(lat), 6), round(float(lon), 6), int(radius))


def _cache_ttl_seconds() -> float:
    return max(0.0, float(os.getenv("OVERPASS_CACHE_TTL_SECONDS", "300") or 0.0))


def _cache_max_entries() -> int:
    return max(1, int(float(os.getenv("OVERPASS_CACHE_MAX_ENTRIES", "128") or 128)))


def _get_cached_result(lat: float, lon: float, radius: int) -> dict[str, Any] | None:
    ttl = _cache_ttl_seconds()
    if ttl <= 0.0:
        return None

    key = _cache_key(lat, lon, radius)
    cached = _OSM_CACHE.get(key)
    if not cached:
        return None

    cached_at, cached_result = cached
    if time.time() - cached_at > ttl:
        _OSM_CACHE.pop(key, None)
        return None

    _OSM_CACHE.move_to_end(key)
    result = copy.deepcopy(cached_result)
    result["source_status"] = "cached"
    result["cache_hit"] = True
    return result


def _store_cached_result(lat: float, lon: float, radius: int, result: dict[str, Any]) -> None:
    ttl = _cache_ttl_seconds()
    if ttl <= 0.0:
        return

    key = _cache_key(lat, lon, radius)
    _OSM_CACHE[key] = (time.time(), copy.deepcopy(result))
    _OSM_CACHE.move_to_end(key)
    while len(_OSM_CACHE) > _cache_max_entries():
        _OSM_CACHE.popitem(last=False)


def _retry_attempts_per_endpoint() -> int:
    return max(1, int(float(os.getenv("OVERPASS_RETRIES_PER_ENDPOINT", "2") or 2)))


def _retry_backoff_seconds() -> float:
    return max(0.0, float(os.getenv("OVERPASS_RETRY_BACKOFF_SECONDS", "1.0") or 0.0))


def _is_retryable_http_error(exc: urlerror.HTTPError) -> bool:
    return int(exc.code) in {429, 500, 502, 503, 504}


def query_osm_nearby(lat: float, lon: float, radius: int = 300) -> dict[str, Any]:
    cached = _get_cached_result(lat, lon, radius)
    if cached is not None:
        return cached

    overpass_query = _build_overpass_query(lat, lon, radius)
    errors: list[str] = []
    last_exc: Exception | None = None
    attempts_per_endpoint = _retry_attempts_per_endpoint()
    backoff_seconds = _retry_backoff_seconds()

    for endpoint in _osm_endpoints():
        for attempt in range(1, attempts_per_endpoint + 1):
            try:
                payload = _fetch_overpass_payload(endpoint, overpass_query)
                features = _normalize_features(payload)
                result = _base_result(lat, lon, radius)
                result["count"] = len(features)
                result["features"] = features
                result["source"] = endpoint
                _store_cached_result(lat, lon, radius, result)
                return result
            except urlerror.HTTPError as exc:
                last_exc = exc
                errors.append(f"{endpoint} -> HTTP {exc.code}: {exc.reason}")
                if not _is_retryable_http_error(exc) or attempt >= attempts_per_endpoint:
                    break
            except (urlerror.URLError, TimeoutError, ConnectionError, socket.timeout, ssl.SSLError) as exc:
                last_exc = exc
                errors.append(f"{endpoint} -> {exc}")
                if attempt >= attempts_per_endpoint:
                    break
            except Exception as exc:
                last_exc = exc
                errors.append(f"{endpoint} -> {exc}")
                break

            if backoff_seconds > 0.0:
                time.sleep(backoff_seconds * attempt)

    detail = " | ".join(errors) if errors else str(last_exc or "all Overpass endpoints failed")
    raise RuntimeError(detail)


def query_osm_nearby_safe(lat: float, lon: float, radius: int = 300) -> dict[str, Any]:
    try:
        return query_osm_nearby(lat, lon, radius)
    except Exception as exc:
        result = _base_result(lat, lon, radius)
        result["source_status"] = "unavailable"
        result["warnings"] = [
            "OSM nearby context is temporarily unavailable; the report continues without nearby mapped features."
        ]
        result["error_detail"] = str(exc)
        return result
