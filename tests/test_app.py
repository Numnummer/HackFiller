import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import processing
from app.main import app
from app.parser import extract_images

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_relative_links():
    html = '<img src="../media/a.jpg" alt="A"><img src="/b.png"><img src="data:x"><img>'
    got = extract_images(html, "https://x.com/cat/page-1.html")
    assert [c.url for c in got] == ["https://x.com/media/a.jpg", "https://x.com/b.png"]


def test_all_filters_change_image_and_keep_shape():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (40, 30, 3), dtype=np.uint8)
    for name in list(processing.FILTERS) + list(processing.PRESETS):
        out = processing.apply_chain(arr, processing.resolve_chain(name, 1.0))
        assert out.shape == arr.shape and out.dtype == np.uint8
        assert not np.array_equal(out, arr), name


def test_stats():
    arr = np.full((10, 20, 3), 255, np.uint8)
    s = processing.compute_stats(arr)
    assert (s.width, s.height, s.aspect_ratio) == (20, 10, 2.0)
    assert s.light_ratio == 1.0 and s.contrast == 0


def test_unknown_filter():
    assert client.post("/api/analyze", json={"filter": "nope"}).status_code == 422
    assert client.post("/api/analyze", json={"filter": "blur", "intensity": 3}).status_code == 422


def test_bad_image_is_skipped(monkeypatch):
    import httpx
    from app import main, parser

    buf = io.BytesIO()
    Image.new("RGB", (50, 40), (200, 30, 30)).save(buf, "PNG")

    def handler(req):
        if req.url.path == "/p.html":
            return httpx.Response(200, text='<img src="ok.png"><img src="bad.png"><img src="404.png">')
        if req.url.path == "/ok.png":
            return httpx.Response(200, content=buf.getvalue())
        if req.url.path == "/bad.png":
            return httpx.Response(200, content=b"not an image")
        return httpx.Response(404)

    monkeypatch.setattr(main.config, "SOURCE_PAGES", ["http://t/p.html"])
    monkeypatch.setattr(main.config, "IMAGE_SELECTOR", "img")
    monkeypatch.setattr(parser, "make_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True))
    r = client.post("/api/analyze", json={"filter": "edges", "limit": 5})
    assert r.status_code == 200
    d = r.json()
    assert len(d["images"]) == 1 and len(d["errors"]) == 2
