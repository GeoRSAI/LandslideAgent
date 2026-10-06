from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ImageInfo(DomainModel):
    image_path: str
    width: int | None = None
    height: int | None = None
    bands: int | None = None
    dtype: str | None = None
    crs: str | None = None
    resolution: dict[str, Any] | list[float] | None = None
    bounds: Any = None


class FirstPassResult(DomainModel):
    has_landslide: bool | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    assessment_label: Literal["likely", "unlikely", "uncertain", "error"] | None = None
    scene_description: str = ""
    evidence: str = ""


class SegmentationResult(DomainModel):
    mask_path: str = ""
    overlay_path: str = ""
    landslide_pixels: int = 0
    area_ratio: float = Field(default=0.0, ge=0.0)
    polygon_count: int = 0


class Region(DomainModel):
    tile_id: int = 0
    bbox: list[float] = Field(min_length=4, max_length=4)
    score: float = 0.0
    class_id: int = 0
    source: str = ""


class RefinementResult(DomainModel):
    regions: list[Region] = Field(default_factory=list)
    area_ratio: float = Field(default=0.0, ge=0.0)
    overlay_path: str = ""
    mask_path: str = ""


class ClassificationResult(DomainModel):
    class_name: str = "unknown"
    class_id: int | None = None
    confidence: float = 0.0
    topk: list[dict[str, Any]] = Field(default_factory=list)


class GeoContext(DomainModel):
    background: dict[str, Any] = Field(default_factory=dict)
    nearby: dict[str, Any] = Field(default_factory=dict)
    available: bool = False
    warnings: list[str] = Field(default_factory=list)


class AnalysisState(DomainModel):
    image_info: ImageInfo
    stage1: FirstPassResult | None = None
    segmentation: SegmentationResult | None = None
    refinement: RefinementResult | None = None
    classification: ClassificationResult | None = None
    geo_context: GeoContext | None = None
    llm_second_pass: dict[str, Any] | None = None
    final_report: dict[str, Any] | None = None
    report_path: str = ""
    report_out_path: str = ""
    latitude: float | None = None
    longitude: float | None = None
    nearby_radius: int = 300
    enable_second_pass: bool = False
    second_pass_area_ratio: float = 0.20
    errors: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
