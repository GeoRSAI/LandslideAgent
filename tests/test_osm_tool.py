from src.tools import osm_tool


def test_query_osm_nearby_safe_returns_generic_warning(monkeypatch):
    def fake_query(lat: float, lon: float, radius: int = 1000):
        raise RuntimeError("https://overpass-api.de/api/interpreter -> HTTP 504: Gateway Timeout")

    monkeypatch.setattr(osm_tool, "query_osm_nearby", fake_query)

    result = osm_tool.query_osm_nearby_safe(30.0, 103.0, 1000)

    assert result["source_status"] == "unavailable"
    assert result["count"] == 0
    assert "temporarily unavailable" in result["warnings"][0]
    assert "504" not in result["warnings"][0]
    assert "504" in result["error_detail"]
