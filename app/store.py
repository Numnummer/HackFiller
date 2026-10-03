"""Небольшое in-memory хранилище полноразмерных JPEG для просмотра по клику."""
import secrets
from collections import OrderedDict

MAX_ITEMS = 120  # ≈ 5 запросов по 12 изображений × 2; старые вытесняются
_items: "OrderedDict[str, bytes]" = OrderedDict()


def put(data: bytes) -> str:
    token = secrets.token_urlsafe(12)
    _items[token] = data
    while len(_items) > MAX_ITEMS:
        _items.popitem(last=False)
    return token


def get(token: str) -> bytes | None:
    return _items.get(token)
