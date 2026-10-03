"""Настройки приложения: источник данных и лимиты."""

SOURCE_NAME = "Books to Scrape (демо-магазин для парсинга)"
SOURCE_PAGES = [
    "https://books.toscrape.com/catalogue/page-1.html",
    "https://books.toscrape.com/catalogue/page-2.html",
    "https://books.toscrape.com/catalogue/page-3.html",
    "https://books.toscrape.com/catalogue/page-4.html",
    "https://books.toscrape.com/catalogue/page-5.html",
]
SOURCE_URL = "https://books.toscrape.com/"

HTML_TIMEOUT = 10.0          # секунд на загрузку страницы
IMAGE_TIMEOUT = 10.0         # секунд на загрузку одного изображения
MAX_IMAGE_BYTES = 3_000_000  # больше — пропускаем
MAX_IMAGES = 20              # жёсткий потолок обрабатываемых изображений
DEFAULT_IMAGES = 12
PREVIEW_SIDE = 320           # максимальная сторона превью в ответе, px
CONCURRENCY = 8
USER_AGENT = "ImageScope-Hackathon/1.0"
