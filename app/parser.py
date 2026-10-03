"""Получение HTML, извлечение ссылок на изображения и их загрузка."""
import asyncio
import re
import time
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from . import config


@dataclass
class Candidate:
    url: str
    title: str


@dataclass
class Downloaded:
    url: str
    title: str
    data: bytes


@dataclass
class Failure:
    url: str | None
    stage: str
    message: str


def extract_images(html: str, base_url: str, selector: str = "img",
                   rewrite: tuple[str, str] | None = None) -> list[Candidate]:
    """Находит <img> и превращает относительные ссылки в абсолютные."""
    soup = BeautifulSoup(html, "html.parser")
    found: list[Candidate] = []
    for img in soup.select(selector):
        src = img.get("src") or img.get("data-src")
        if not src or src.startswith("data:"):
            continue
        url = urldefrag(urljoin(base_url, src.strip()))[0]
        if urlparse(url).scheme not in ("http", "https"):
            continue
        if rewrite:
            url = re.sub(rewrite[0], rewrite[1], url)
        title = (img.get("alt") or img.get("title") or "").strip()
        title = title.removeprefix("Preview wallpaper").strip()
        found.append(Candidate(url, title))
    return found


async def _fetch_page(client: httpx.AsyncClient, url: str):
    try:
        r = await client.get(url, timeout=config.HTML_TIMEOUT)
        r.raise_for_status()
        return extract_images(r.text, str(r.url), config.IMAGE_SELECTOR,
                                              config.IMAGE_URL_REWRITE), None
    except Exception as e:  # noqa: BLE001 — любая ошибка страницы не должна ронять запрос
        return [], Failure(url, "page", f"{type(e).__name__}: {e}")


_cache: dict[tuple, tuple[float, list, int]] = {}


async def collect_candidates(client: httpx.AsyncClient, pages: list[str]):
    """Последовательно (с паузой по Crawl-delay) загружает страницы каталога.
    Возвращает (кандидаты, ошибки, число страниц)."""
    key = tuple(pages)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < config.PAGES_CACHE_TTL:
        return list(hit[1]), [], hit[2]  # список страниц каталога недавно уже собирали
    results = []
    for i, p in enumerate(pages):
        if i:
            await asyncio.sleep(config.CRAWL_DELAY)
        results.append(await _fetch_page(client, p))
    seen: set[str] = set()
    candidates: list[Candidate] = []
    failures: list[Failure] = []
    ok_pages = 0
    for imgs, fail in results:
        if fail:
            failures.append(fail)
            continue
        ok_pages += 1
        for c in imgs:
            if c.url not in seen:
                seen.add(c.url)
                candidates.append(c)
    if not failures and candidates:
        _cache[key] = (time.monotonic(), list(candidates), ok_pages)
    return candidates, failures, ok_pages


def pick_spread(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Равномерно выбирает limit кандидатов по всем страницам (для разнообразия)."""
    if len(candidates) <= limit:
        return candidates
    step = len(candidates) / limit
    return [candidates[int(i * step)] for i in range(limit)]


async def _download(client, sem, cand: Candidate):
    async with sem:
        try:
            async with client.stream("GET", cand.url, timeout=config.IMAGE_TIMEOUT) as r:
                r.raise_for_status()
                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > config.MAX_IMAGE_BYTES:
                        return None, Failure(cand.url, "download", "файл слишком большой, пропущен")
            if not buf:
                return None, Failure(cand.url, "download", "пустой ответ")
            return Downloaded(cand.url, cand.title, bytes(buf)), None
        except httpx.TimeoutException:
            return None, Failure(cand.url, "download", "тайм-аут загрузки")
        except Exception as e:  # noqa: BLE001
            return None, Failure(cand.url, "download", f"{type(e).__name__}: {e}")


async def download_images(client: httpx.AsyncClient, cands: list[Candidate]):
    sem = asyncio.Semaphore(config.CONCURRENCY)
    results = await asyncio.gather(*(_download(client, sem, c) for c in cands))
    return [d for d, _ in results if d], [f for _, f in results if f]


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": config.USER_AGENT}, follow_redirects=True
    )
