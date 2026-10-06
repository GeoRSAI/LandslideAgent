import pytest
from fastapi import HTTPException

from scripts import llm_service as service


def test_media_accepts_file_under_configured_root(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_ALLOWED_ROOT", str(tmp_path))
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    assert service._safe_resolve_image(str(image)) == image.resolve()


def test_media_rejects_sibling_with_same_path_prefix(monkeypatch, tmp_path):
    allowed = tmp_path / "images"
    sibling = tmp_path / "images-private"
    allowed.mkdir()
    sibling.mkdir()
    image = sibling / "private.png"
    image.write_bytes(b"fixture")
    monkeypatch.setenv("IMAGE_ALLOWED_ROOT", str(allowed))
    with pytest.raises(HTTPException) as exc:
        service._safe_resolve_image(str(image))
    assert exc.value.status_code == 400
