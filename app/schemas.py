"""Pydantic-модели запроса и ответа."""
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    filter: str = Field("product_boost", description="Имя фильтра или пресета")
    intensity: float = Field(1.0, ge=0.0, le=1.0, description="Интенсивность 0..1")
    limit: int = Field(12, ge=1, le=20, description="Сколько изображений обработать")


class FilterInfo(BaseModel):
    name: str
    title: str
    engine: str  # numpy | torch | preset
    description: str


class AppliedFilter(BaseModel):
    name: str
    engine: str
    intensity: float


class ImageStats(BaseModel):
    width: int
    height: int
    aspect_ratio: float
    brightness: float  # 0..255
    contrast: float    # std яркости
    mean_color: tuple[int, int, int]
    dark_ratio: float   # доля пикселей с яркостью < 40
    light_ratio: float  # доля пикселей с яркостью > 215
    histogram: list[float]  # 16 бинов gray, сумма = 1
    flags: list[str]


class ImageResult(BaseModel):
    url: str
    title: str
    stats: ImageStats
    result_stats: ImageStats
    filters: list[AppliedFilter]
    original: str   # data URI (превью)
    original_full: str   # URL полноразмерного JPEG
    processed_full: str
    processed: str  # data URI


class ErrorItem(BaseModel):
    url: str | None = None
    stage: str  # page | download | decode | process
    message: str


class Summary(BaseModel):
    processed: int
    failed: int
    avg_brightness: float
    avg_contrast: float
    avg_result_brightness: float
    avg_result_contrast: float
    histogram: list[float]
    result_histogram: list[float]


class AnalyzeResponse(BaseModel):
    source_name: str
    source_url: str
    pages_fetched: int
    images_found: int
    filter: str
    intensity: float
    applied: list[AppliedFilter]
    summary: Summary
    images: list[ImageResult]
    errors: list[ErrorItem]
    elapsed_sec: float
