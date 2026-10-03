"""Генерирует docs/ImageScope.pptx (python docs/make_deck.py)."""
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
IMG = HERE / "img"
INK, PAPER, AMBER, SLATE, MUTED, CARD = (RGBColor.from_string(c) for c in
    ("14213D", "FFFFFF", "F5A623", "2B3A5C", "5B6785", "EEF1F7"))

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    f = slide.background.fill
    f.solid(); f.fore_color.rgb = color


def text(slide, x, y, w, h, s, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        r = p.add_run(); r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, font
    return tb


def box(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill; s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def title(slide, s):
    text(slide, 0.6, 0.45, 12, 0.8, s, size=36, bold=True, font="Cambria")


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


# 1 ---------------------------------------------------------------- титул
s = prs.slides.add_slide(BLANK); bg(s, INK)
text(s, 0.8, 1.5, 7, 1.2, "ImageScope", size=66, bold=True, color=PAPER, font="Cambria")
text(s, 0.8, 2.9, 6.6, 1.4, "Собираем картинки каталога, измеряем и превращаем в стиль — с «до/после» за один клик",
     size=22, color=RGBColor.from_string("CADCFC"))
for i, (n, l) in enumerate([("120", "обоев с wallpaperscraft.com"), ("13", "фильтров NumPy и PyTorch"), ("5", "творческих пресетов")]):
    text(s, 0.8 + i * 2.5, 5.2, 2.3, 0.9, n, size=48, bold=True, color=AMBER, font="Cambria")
    text(s, 0.8 + i * 2.5, 6.1, 2.2, 0.7, l, size=14, color=PAPER)
s.shapes.add_picture(str(IMG / "orig1.jpg"), Inches(8.2), Inches(1.0), width=Inches(4.5))
s.shapes.add_picture(str(IMG / "comic1.jpg"), Inches(8.2), Inches(4.1), width=Inches(4.5))
text(s, 8.2, 3.55, 2.4, 0.5, "до", size=14, color=AMBER)
text(s, 8.2, 6.75, 3, 0.5, "после: «комикс»", size=14, color=AMBER)
notes(s, "Задача: превратить сырые картинки каталога в единый, чистый визуальный ряд и дать измеримый результат. ImageScope сам собирает данные, анализирует и показывает до/после.")

# 2 ----------------------------------------------------- задача и архитектура
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
title(s, "Задача и архитектура")
text(s, 0.6, 1.4, 12, 0.9,
     "Картинки каталога разные: тёмные, малоконтрастные, неровные. ImageScope находит их на реальном сайте (WallpapersCraft, раздел Art), "
     "считает метрики и применяет фильтры — пользователь сразу видит эффект.", size=18, color=MUTED)
steps = [("Интерфейс", "HTML/JS: фильтр, интенсивность, галерея"), ("FastAPI", "POST /api/analyze, Pydantic"),
         ("Парсинг", "httpx + BeautifulSoup, тайм-ауты, пропуск битых"), ("NumPy / PyTorch", "статистика, фильтры, conv2d"),
         ("Отчёт", "JSON: метрики, до/после, ошибки")]
w, gap = 2.25, 0.2
for i, (h, d) in enumerate(steps):
    x = 0.6 + i * (w + gap)
    box(s, x, 3.0, w, 2.4, INK if i != 3 else AMBER)
    c = PAPER if i != 3 else INK
    text(s, x + 0.2, 3.25, w - 0.4, 0.6, h, size=20, bold=True, color=c, font="Cambria")
    text(s, x + 0.2, 4.0, w - 0.4, 1.3, d, size=14, color=c)
    if i < 4:
        text(s, x + w - 0.02, 3.95, 0.25, 0.4, "›", size=28, bold=True, color=MUTED)
text(s, 0.6, 5.9, 12, 0.9, "Устойчивость: тайм-ауты 10 с, лимит 3 МБ на файл, ≤ 20 изображений за запрос. Один плохой файл не ломает отчёт — "
     "он попадает в список ошибок.", size=16)
notes(s, "Цепочка единая: интерфейс ходит только в API. Парсер собирает 5 страниц каталога параллельно, берёт равномерную выборку и качает картинки с лимитами. Ошибки возвращаются в JSON.")

# 3 ------------------------------------------------------ методы NumPy/PyTorch
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
title(s, "Методы: NumPy и PyTorch")
cols = [("NumPy: измеряем и правим массивы", INK,
         ["Размер, соотношение сторон", "Яркость и контраст (std яркости)", "Средний цвет, доли тёмных/светлых пикселей",
          "16-бинная гистограмма", "Фильтры: сепия, постеризация, пикселизация, гамма, автоконтраст, негатив, порог, grayscale"]),
        ("PyTorch: свёртки conv2d", AMBER,
         ["Тензор 1×C×H×W, depthwise-ядра", "Размытие (Гаусс 3×3), резкость, рельеф", "Собель: два ядра → величина градиента → контуры",
          "Результат — готовая картинка, не число", "Пресеты — цепочки фильтров с общей интенсивностью"])]
for i, (h, col, items) in enumerate(cols):
    x = 0.6 + i * 6.2
    box(s, x, 1.5, 5.9, 4.4, CARD)
    box(s, x + 0.3, 1.8, 0.5, 0.5, col, MSO_SHAPE.OVAL)
    text(s, x + 1.0, 1.8, 4.7, 0.6, h, size=20, bold=True, anchor=MSO_ANCHOR.MIDDLE, font="Cambria")
    text(s, x + 0.35, 2.7, 5.2, 3.8, ["• " + t for t in items], size=16)
notes(s, "NumPy-часть — статистика и фильтры чистыми операциями над массивами. PyTorch-часть — настоящий conv2d с depthwise-ядрами; Собель даёт контуры для комикса и наброска.")

# 4 ------------------------------------------------- концепция и до/после
s = prs.slides.add_slide(BLANK); bg(s, INK)
text(s, 0.6, 0.45, 12, 0.8, "Один каталог — пять стилей", size=36, bold=True, color=PAPER, font="Cambria")
row = [("Оригинал", "orig3.jpg", "в среднем: яркость 83, контраст 50"), ("Карточка", "product_boost3.jpg", "в среднем: яркость 109, контраст 62"),
       ("Комикс", "comic3.jpg", "постеризация + контуры Собеля"), ("Ретро", "retro3.jpg", "сепия + размытие + тиснение"),
       ("Пиксель-арт", "pixel_art3.jpg", "пикселизация + палитра")]
for i, (h, f, cap) in enumerate(row):
    x = 0.6 + i * 2.5
    box(s, x, 1.6, 2.3, 4.4, SLATE)
    pic = s.shapes.add_picture(str(IMG / f), Inches(x + 0.15), Inches(2.5), width=Inches(2.0))
    text(s, x + 0.15, 4.7, 2.0, 0.4, h, size=18, bold=True, color=AMBER, font="Cambria")
    text(s, x + 0.15, 5.15, 2.0, 0.8, cap, size=14, color=PAPER)
text(s, 0.6, 6.3, 12, 0.7, "Средние по 6 обоям: «Карточка товара» поднимает яркость 83 → 109 и контраст 50 → 62 — каталог выглядит чище и единообразнее.",
     size=16, color=RGBColor.from_string("CADCFC"))
notes(s, "Одни и те же обои пропущена через разные пресеты. Главный практический пресет — «Карточка товара»: автоконтраст, подъём теней, резкость.")

# 5 ---------------------------------------------------- вклад и выводы
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
title(s, "Вклад команды и выводы")
rows = [("Имя 1", "FastAPI, Pydantic-модели, тесты", "/api/analyze, валидация, JSON-отчёт"),
        ("Имя 2", "Парсинг и загрузка", "httpx + BeautifulSoup, ссылки, тайм-ауты, ошибки"),
        ("Имя 3", "NumPy/PyTorch и интерфейс", "13 фильтров, 5 пресетов, галерея до/после")]
tbl = s.shapes.add_table(4, 3, Inches(0.6), Inches(1.5), Inches(12.1), Inches(2.6)).table
for j, (h, w_) in enumerate(zip(["Участник", "Зона ответственности", "Результат"], [2.2, 4.4, 5.5])):
    tbl.columns[j].width = Inches(w_)
    c = tbl.cell(0, j); c.text = h
for i, r in enumerate(rows, 1):
    for j, v in enumerate(r):
        tbl.cell(i, j).text = v
for i in range(4):
    for j in range(3):
        c = tbl.cell(i, j); c.fill.solid()
        c.fill.fore_color.rgb = INK if i == 0 else (CARD if i % 2 else PAPER)
        for p in c.text_frame.paragraphs:
            for run in p.runs:
                run.font.size = Pt(16); run.font.name = "Calibri"; run.font.bold = i == 0
                run.font.color.rgb = PAPER if i == 0 else INK
text(s, 0.6, 4.5, 12, 0.5, "Выводы", size=22, bold=True, font="Cambria")
text(s, 0.6, 5.1, 12, 1.8, ["• Рабочая цепочка от сбора до «до/после» на реальном сайте, 120 изображений",
                            "• NumPy и PyTorch реально участвуют в результате: метрики и визуальные фильтры",
                            "• Дальше: свои ядра, сохранение пресетов, пакетная выгрузка обработанных карточек"], size=16)
notes(s, "Таблицу вклада заменить реальными именами команды перед защитой. Каждый участник кратко называет свой вклад.")

prs.save(HERE / "ImageScope.pptx")
print("saved")
