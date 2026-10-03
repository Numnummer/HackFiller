"""FastAPI: /health, /api/filters, /api/analyze и раздача интерфейса."""
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, parser, processing, store
from .schemas import (AnalyzeRequest, AnalyzeResponse, ErrorItem, FilterInfo,
                      ImageResult, Summary)

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="ImageScope", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/source")
async def source():
    return {"name": config.SOURCE_NAME, "url": config.SOURCE_URL}


@app.get("/api/img/{token}.jpg")
async def full_image(token: str):
    data = store.get(token)
    if data is None:
        raise HTTPException(404, "Изображение устарело — запустите сбор заново")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/filters", response_model=list[FilterInfo])
async def filters():
    return processing.list_filters()


def _process_one(item: parser.Downloaded, chain):
    """Синхронная CPU-работа над одним изображением (запускается в пуле потоков)."""
    arr = processing.decode_rgb(item.data)
    stats = processing.compute_stats(arr)
    out = processing.apply_chain(arr, chain)
    return ImageResult(
        url=item.url, title=item.title, stats=stats,
        result_stats=processing.compute_stats(out),
        filters=processing.describe_chain(chain),
        original=processing.to_data_uri(arr),
        processed=processing.to_data_uri(out),
        original_full=f"/api/img/{store.put(processing.to_jpeg(arr))}.jpg",
        processed_full=f"/api/img/{store.put(processing.to_jpeg(out))}.jpg",
    )


def _mean_hist(hists):
    n = len(hists)
    return [round(sum(h[i] for h in hists) / n, 4) for i in range(processing.HIST_BINS)] if n else [0.0] * processing.HIST_BINS


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    try:
        chain = processing.resolve_chain(req.filter, req.intensity)
    except KeyError:
        raise HTTPException(422, f"Неизвестный фильтр: {req.filter}")

    t0 = time.perf_counter()
    errors: list[ErrorItem] = []
    limit = min(req.limit, config.MAX_IMAGES)

    async with parser.make_client() as client:
        cands, fails, pages_ok = await parser.collect_candidates(client, config.SOURCE_PAGES)
        errors += [ErrorItem(**f.__dict__) for f in fails]
        if not cands:
            raise HTTPException(502, "Не удалось получить ни одного изображения с источника: "
                                + "; ".join(f.message for f in fails))
        chosen = parser.pick_spread(cands, limit)
        downloaded, dfails = await parser.download_images(client, chosen)
        errors += [ErrorItem(**f.__dict__) for f in dfails]

    results: list[ImageResult] = []
    for item in downloaded:
        try:
            results.append(await run_in_threadpool(_process_one, item, chain))
        except Exception as e:  # noqa: BLE001 — битое изображение пропускаем
            errors.append(ErrorItem(url=item.url, stage="decode",
                                    message=f"не удалось обработать: {type(e).__name__}: {e}"))

    n = len(results)
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0.0  # noqa: E731
    summary = Summary(
        processed=n, failed=len(chosen) - n,
        avg_brightness=mean([r.stats.brightness for r in results]),
        avg_contrast=mean([r.stats.contrast for r in results]),
        avg_result_brightness=mean([r.result_stats.brightness for r in results]),
        avg_result_contrast=mean([r.result_stats.contrast for r in results]),
        histogram=_mean_hist([r.stats.histogram for r in results]),
        result_histogram=_mean_hist([r.result_stats.histogram for r in results]),
    )
    return AnalyzeResponse(
        source_name=config.SOURCE_NAME, source_url=config.SOURCE_URL,
        pages_fetched=pages_ok, images_found=len(cands),
        filter=req.filter, intensity=req.intensity,
        applied=processing.describe_chain(chain),
        summary=summary, images=results, errors=errors,
        elapsed_sec=round(time.perf_counter() - t0, 2),
    )
