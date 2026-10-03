"""Анализ (NumPy) и фильтры: NumPy-операции над массивами и PyTorch conv2d."""
import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .schemas import AppliedFilter, FilterInfo, ImageStats

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
HIST_BINS = 16


# ---------- загрузка / кодирование ----------
def decode_rgb(data: bytes) -> np.ndarray:
    """bytes -> RGB uint8 массив H x W x 3 (бросает исключение для битых файлов)."""
    img = Image.open(io.BytesIO(data))
    img.load()
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


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
    levels = int(round(8 - 6 * k))  # k=0 -> 8 уровней, k=1 -> 2 уровня
    levels = max(2, levels)
    step = 255 / (levels - 1)
    return np.round(a / step) * step


def np_pixelate(a, k):
    h, w, _ = a.shape
    block = max(1, int(round(2 + 14 * k)))
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
    gamma = 1.0 - 0.5 * k  # <1 осветляет тёмные области
    return 255 * np.power(a / 255, gamma)


def np_contrast(a, k):
    """Растяжение гистограммы: 1-й..99-й процентили -> 0..255."""
    lo, hi = np.percentile(a, [1, 99])
    if hi - lo < 1:
        return a
    stretched = (a - lo) * (255 / (hi - lo))
    return _blend(a, stretched, k)


# ---------- PyTorch-свёртки ----------
KERNELS = {
    "blur": torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16,
    "sharpen": torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32),
    "emboss": torch.tensor([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=torch.float32),
}
SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
SOBEL_Y = SOBEL_X.T.contiguous()


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    """H x W x 3 -> 1 x C x H x W (float, 0..1)."""
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0) / 255.0


def _from_tensor(t: torch.Tensor) -> np.ndarray:
    return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).numpy()


def _conv(t: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise conv2d: одно и то же ядро для каждого из каналов C."""
    c = t.shape[1]
    w = kernel.expand(c, 1, *kernel.shape).contiguous()
    pad = kernel.shape[0] // 2
    t = F.pad(t, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(t, w, groups=c)


def _repeat_conv(t, kernel, times):
    for _ in range(times):
        t = _conv(t, kernel)
    return t


def _mix(orig, new, k):
    return orig * (1 - k) + new * k


def th_blur(a, k):
    t = _to_tensor(a)
    return _from_tensor(_repeat_conv(t, KERNELS["blur"], int(round(5 * k))))


def th_sharpen(a, k):
    t = _to_tensor(a)
    return _from_tensor(_mix(t, _conv(t, KERNELS["sharpen"]), k))


def th_emboss(a, k):
    t = _to_tensor(a)
    emb = _conv(t, KERNELS["emboss"]) + 0.5
    return _from_tensor(_mix(t, emb, k))


def _sobel_magnitude(t: torch.Tensor) -> torch.Tensor:
    g = t.mean(dim=1, keepdim=True)  # 1 x 1 x H x W
    gx, gy = _conv(g, SOBEL_X), _conv(g, SOBEL_Y)
    return torch.sqrt(gx ** 2 + gy ** 2)


def th_edges(a, k):
    """Контуры Собеля: тёмные линии на белом фоне, как в карандашном наброске."""
    t = _to_tensor(a)
    mag = _sobel_magnitude(t)
    mag = (mag / (mag.amax() + 1e-6) * (2 + 4 * k)).clamp(0, 1)
    return _from_tensor((1 - mag).expand(-1, 3, -1, -1).contiguous())


def th_ink(a, k):
    """Накладывает контуры Собеля чёрными линиями поверх изображения."""
    t = _to_tensor(a)
    mag = _sobel_magnitude(t)
    mag = (mag / (mag.amax() + 1e-6) * 4).clamp(0, 1)
    return _from_tensor(t * (1 - mag * k))


# ---------- реестр фильтров и пресетов ----------
FILTERS = {
    "grayscale": (np_grayscale, "numpy", "Оттенки серого", "Взвешенная сумма каналов R,G,B."),
    "invert": (np_invert, "numpy", "Негатив", "255 − значение пикселя."),
    "sepia": (np_sepia, "numpy", "Сепия", "Матричное преобразование каналов в тёплые тона."),
    "posterize": (np_posterize, "numpy", "Постеризация", "Квантование до 2–8 уровней на канал."),
    "pixelate": (np_pixelate, "numpy", "Пикселизация", "Усреднение блоков 2–16 px."),
    "threshold": (np_threshold, "numpy", "Порог", "Чёрно-белое изображение по порогу яркости."),
    "gamma": (np_gamma, "numpy", "Гамма-коррекция", "Подъём теней: x^γ."),
    "auto_contrast": (np_contrast, "numpy", "Автоконтраст", "Растяжение гистограммы по 1–99 процентилям."),
    "blur": (th_blur, "torch", "Размытие", "Гауссово ядро 3×3, conv2d, до 5 проходов."),
    "sharpen": (th_sharpen, "torch", "Резкость", "Ядро повышения резкости 3×3 через conv2d."),
    "emboss": (th_emboss, "torch", "Рельеф", "Ядро emboss — эффект тиснения."),
    "edges": (th_edges, "torch", "Контуры", "Оператор Собеля (2 ядра conv2d) — набросок карандашом."),
    "ink": (th_ink, "torch", "Чернила", "Контуры Собеля поверх исходника."),
}

# пресет = цепочка (фильтр, множитель интенсивности)
PRESETS = {
    "product_boost": ("Карточка товара", "Автоконтраст + подъём теней + резкость: чище и ярче для каталога.",
                      [("auto_contrast", 1.0), ("gamma", 0.6), ("sharpen", 1.0)]),
    "comic": ("Комикс", "Постеризация цвета + чёрные контуры Собеля.",
              [("posterize", 0.6), ("ink", 1.0)]),
    "retro": ("Ретро", "Сепия + лёгкое размытие + тиснение для выцветшей плёнки.",
              [("sepia", 1.0), ("blur", 0.25), ("emboss", 0.15)]),
    "pixel_art": ("Пиксель-арт", "Крупная пикселизация + постеризация палитры.",
                  [("pixelate", 1.0), ("posterize", 0.5)]),
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
