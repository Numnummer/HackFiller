"""Анализ (NumPy) и фильтры: NumPy-операции над массивами и PyTorch conv2d."""
import base64
import io
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .schemas import AppliedFilter, FilterInfo, ImageStats

def _pick_device() -> torch.device:
    """IMAGESCOPE_DEVICE=auto|cpu|cuda (по умолчанию auto: CUDA, если доступна)."""
    want = os.environ.get("IMAGESCOPE_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if want == "cuda":
        raise RuntimeError("IMAGESCOPE_DEVICE=cuda, но CUDA недоступна (нужна CUDA-сборка torch)")
    return torch.device("cpu")


DEVICE = _pick_device()
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
HIST_BINS = 16


# ---------- загрузка / кодирование ----------
def decode_rgb(data: bytes) -> np.ndarray:
    """bytes -> RGB uint8 массив H x W x 3 (бросает исключение для битых файлов)."""
    img = Image.open(io.BytesIO(data))
    img.load()
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def to_jpeg(arr: np.ndarray, max_side: int = 1920, quality: int = 90) -> bytes:
    img = Image.fromarray(arr)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_data_uri(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    img.thumbnail((config.PREVIEW_SIDE, config.PREVIEW_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------- статистика (NumPy) ----------
def compute_stats(arr: np.ndarray) -> ImageStats:
    h, w, _ = arr.shape
    gray = arr.astype(np.float32) @ LUMA
    brightness = float(gray.mean())
    contrast = float(gray.std())
    hist, _ = np.histogram(gray, bins=HIST_BINS, range=(0, 256))
    hist = (hist / gray.size).round(4).tolist()
    mean_color = tuple(int(v) for v in arr.reshape(-1, 3).mean(axis=0))
    flags = []
    if brightness < 70:
        flags.append("слишком тёмное")
    elif brightness > 190:
        flags.append("слишком светлое")
    if contrast < 25:
        flags.append("низкий контраст")
    return ImageStats(
        width=w, height=h, aspect_ratio=round(w / h, 3),
        brightness=round(brightness, 2), contrast=round(contrast, 2),
        mean_color=mean_color,
        dark_ratio=round(float((gray < 40).mean()), 4),
        light_ratio=round(float((gray > 215).mean()), 4),
        histogram=hist, flags=flags,
    )


# ---------- NumPy-фильтры (float32 0..255 -> float32 0..255) ----------
def _blend(orig, new, k):
    return orig * (1 - k) + new * k


def np_grayscale(a, k):
    g = (a @ LUMA)[..., None].repeat(3, axis=2)
    return _blend(a, g, k)


def np_invert(a, k):
    return _blend(a, 255 - a, k)


def np_sepia(a, k):
    m = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]], dtype=np.float32)
    return _blend(a, np.clip(a @ m.T, 0, 255), k)


def np_posterize(a, k):
    levels = max(3, int(round(8 - 5 * k)))  # k=0 -> 8 уровней, k=1 -> 3 уровня
    step = 255 / (levels - 1)
    return np.round(a / step) * step


def np_pixelate(a, k):
    h, w, _ = a.shape
    block = max(2, int(round((0.004 + 0.012 * k) * max(h, w))))  # 0.4–1.6% стороны (≈8–31 px на Full HD)
    hp, wp = -(-h // block) * block, -(-w // block) * block
    padded = np.pad(a, ((0, hp - h), (0, wp - w), (0, 0)), mode="edge")
    small = padded.reshape(hp // block, block, wp // block, block, 3).mean(axis=(1, 3))
    big = small.repeat(block, axis=0).repeat(block, axis=1)
    return big[:h, :w]


def np_threshold(a, k):
    g = a @ LUMA
    t = 128 - 60 * (k - 0.5)
    bw = np.where(g > t, 255.0, 0.0)[..., None].repeat(3, axis=2)
    return bw


def np_gamma(a, k):
    gamma = 1.0 - 0.65 * k  # <1 осветляет тёмные области
    return 255 * np.power(a / 255, gamma)


def np_contrast(a, k):
    """Автоконтраст: растяжение по 4–96 процентилям + S-кривая (smoothstep) для среднего контраста."""
    lo, hi = np.percentile(a, [4, 96])
    if hi - lo < 1:
        return a
    x = np.clip((a - lo) / (hi - lo), 0, 1)
    x = _blend(x, x * x * (3 - 2 * x), 0.6)  # тени темнее, света ярче
    return _blend(a, x * 255, k)


def np_saturate(a, k):
    """Насыщенность: отталкиваем каналы от серого."""
    g = (a @ LUMA)[..., None]
    return g + (a - g) * (1 + 2.5 * k)  # k=1: цветность ×3.5


def np_vignette(a, k):
    """Затемнение к краям (радиальная маска)."""
    h, w, _ = a.shape
    y, x = np.ogrid[:h, :w]
    r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    mask = 1 - 0.75 * k * np.clip(r - 0.35, 0, 1) ** 1.5
    return a * mask[..., None]


# ---------- PyTorch-свёртки ----------
KERNELS = {
    "emboss": torch.tensor([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=torch.float32),
}
SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
SOBEL_Y = SOBEL_X.T.contiguous()


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    """H x W x 3 -> 1 x C x H x W (float, 0..1)."""
    t = torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE)
    return t.permute(2, 0, 1).unsqueeze(0) / 255.0


def _from_tensor(t: torch.Tensor) -> np.ndarray:
    return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).cpu().numpy()


def _conv(t: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise conv2d: одно и то же ядро для каждого из каналов C."""
    c = t.shape[1]
    w = kernel.to(t.device).expand(c, 1, *kernel.shape).contiguous()
    pad = kernel.shape[0] // 2
    t = F.pad(t, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(t, w, groups=c)


def _mix(orig, new, k):
    return orig * (1 - k) + new * k


def _scale(a: np.ndarray) -> float:
    """Размеры эффектов привязаны к разрешению: на Full HD они крупнее, чем на 640 px."""
    return max(1.0, max(a.shape[:2]) / 640)


def _gauss_blur(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Гауссово размытие: два разделимых прохода conv2d (по строкам и столбцам)."""
    if sigma < 0.3:
        return t
    r = max(1, int(np.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=t.device)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = t.shape[1]
    wh = k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous()
    wv = k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous()
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), wh, groups=c)
    return F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), wv, groups=c)


def th_blur(a, k):
    t = _to_tensor(a)
    return _from_tensor(_gauss_blur(t, 4 * k * _scale(a)))


def th_sharpen(a, k):
    """Unsharp mask: исходник + A·(исходник − размытие)."""
    t = _to_tensor(a)
    detail = t - _gauss_blur(t, 1.5 * _scale(a))
    return _from_tensor(t + (3.0 * k) * detail)


def th_emboss(a, k):
    t = _to_tensor(a)
    emb = 0.5 + 2.0 * _conv(_gauss_blur(t, 0.8 * _scale(a)), KERNELS["emboss"])
    return _from_tensor(_mix(t, emb, k))


def _edge_map(a: np.ndarray, t: torch.Tensor) -> torch.Tensor:
    """Контуры Собеля 0..1: сглаживание → 2 ядра conv2d → нормировка → утолщение линий."""
    s = _scale(a)
    g = _gauss_blur(t.mean(dim=1, keepdim=True), 0.7 * s)
    mag = torch.sqrt(_conv(g, SOBEL_X) ** 2 + _conv(g, SOBEL_Y) ** 2)
    ref = torch.quantile(mag.flatten()[::97], 0.97) + 1e-6
    mag = (mag / ref).clamp(0, 1)
    r = int(s / 1.5)
    if r >= 1:  # утолщаем линии пропорционально разрешению
        mag = F.max_pool2d(mag, 2 * r + 1, stride=1, padding=r)
    return mag


def th_edges(a, k):
    """Контуры Собеля: тёмные линии на белом фоне, как в карандашном наброске."""
    t = _to_tensor(a)
    mag = (_edge_map(a, t) * (0.8 + 1.2 * k)).clamp(0, 1)
    return _from_tensor((1 - mag).expand(-1, 3, -1, -1).contiguous())


def th_ink(a, k):
    """Накладывает контуры Собеля чёрными линиями поверх изображения."""
    t = _to_tensor(a)
    mag = (_edge_map(a, t) * 1.5).clamp(0, 1)
    return _from_tensor(t * (1 - mag * k))


# ---------- реестр фильтров и пресетов ----------
FILTERS = {
    "grayscale": (np_grayscale, "numpy", "Оттенки серого", "Взвешенная сумма каналов R,G,B."),
    "invert": (np_invert, "numpy", "Негатив", "255 − значение пикселя."),
    "sepia": (np_sepia, "numpy", "Сепия", "Матричное преобразование каналов в тёплые тона."),
    "posterize": (np_posterize, "numpy", "Постеризация", "Квантование до 3–8 уровней на канал."),
    "pixelate": (np_pixelate, "numpy", "Пикселизация", "Усреднение квадратных блоков (до 1.6% стороны)."),
    "threshold": (np_threshold, "numpy", "Порог", "Чёрно-белое изображение по порогу яркости."),
    "saturate": (np_saturate, "numpy", "Насыщенность", "Усиление цветности относительно серого (до ×3.5)."),
    "vignette": (np_vignette, "numpy", "Виньетка", "Радиальное затемнение к краям кадра."),
    "gamma": (np_gamma, "numpy", "Гамма-коррекция", "Подъём теней: x^γ."),
    "auto_contrast": (np_contrast, "numpy", "Автоконтраст", "Растяжение по 4–96 процентилям + S-кривая: сильнее контраст."),
    "blur": (th_blur, "torch", "Размытие", "Гауссово размытие, два разделимых conv2d; радиус растёт с разрешением."),
    "sharpen": (th_sharpen, "torch", "Резкость", "Unsharp mask: детали (исходник − размытие, conv2d) усиливаются в 1–4 раза."),
    "emboss": (th_emboss, "torch", "Рельеф", "Сглаживание + ядро emboss 3×3 (conv2d) — эффект тиснения."),
    "edges": (th_edges, "torch", "Контуры", "Оператор Собеля (2 ядра conv2d), линии утолщаются с разрешением — набросок."),
    "ink": (th_ink, "torch", "Чернила", "Контуры Собеля поверх исходника."),
}

# пресет = цепочка (фильтр, множитель интенсивности)
PRESETS = {
    "product_boost": ("Карточка товара", "Автоконтраст + подъём теней + насыщенность + резкость: чище и ярче для каталога.",
                      [("auto_contrast", 1.0), ("gamma", 0.6), ("saturate", 0.8), ("sharpen", 0.6)]),
    "comic": ("Комикс", "Автоконтраст + постеризация + насыщенные цвета + чёрные контуры Собеля.",
              [("auto_contrast", 1.0), ("gamma", 0.7), ("posterize", 0.6), ("saturate", 0.9), ("ink", 1.0)]),
    "retro": ("Ретро", "Сепия + мягкость + тиснение + виньетка: выцветшая плёнка.",
              [("sepia", 1.0), ("blur", 0.2), ("emboss", 0.3), ("vignette", 1.0)]),
    "pixel_art": ("Пиксель-арт", "Мелкая пикселизация + мягкая постеризация палитры.",
                  [("pixelate", 0.7), ("posterize", 0.3), ("saturate", 0.3)]),
    "sketch": ("Набросок", "Только контуры на белом фоне.",
               [("edges", 1.0)]),
}


def list_filters() -> list[FilterInfo]:
    out = [FilterInfo(name=n, title=t, engine=e, description=d)
           for n, (_, e, t, d) in FILTERS.items()]
    out += [FilterInfo(name=n, title=t, engine="preset", description=d)
            for n, (t, d, _) in PRESETS.items()]
    return out


def resolve_chain(name: str, intensity: float) -> list[tuple[str, float]]:
    if name in PRESETS:
        return [(f, round(intensity * m, 3)) for f, m in PRESETS[name][2]]
    if name in FILTERS:
        return [(name, intensity)]
    raise KeyError(name)


def apply_chain(arr: np.ndarray, chain: list[tuple[str, float]]) -> np.ndarray:
    a = arr.astype(np.float32)
    for name, k in chain:
        a = np.clip(FILTERS[name][0](a, k), 0, 255).astype(np.float32)
    return np.round(a).astype(np.uint8)


def describe_chain(chain) -> list[AppliedFilter]:
    return [AppliedFilter(name=n, engine=FILTERS[n][1], intensity=k) for n, k in chain]
