"""Настройки приложения: источник данных и лимиты."""

SOURCE_NAME = "WallpapersCraft — каталог «Art», 1920x1080"
SOURCE_URL = "https://wallpaperscraft.com/catalog/art/1920x1080"
# страница 1 — без суффикса, дальше /page2, /page3 ... (≈15 превью на страницу)
SOURCE_PAGES = [SOURCE_URL] + [f"{SOURCE_URL}/page{n}" for n in range(2, 9)]
IMAGE_SELECTOR = "img.wallpapers__image"  # только превью обоев, без логотипа и иконок
CRAWL_DELAY = 1.0  # секунд между страницами (Crawl-delay в robots.txt источника)

HTML_TIMEOUT = 10.0          # секунд на загрузку страницы
IMAGE_TIMEOUT = 10.0         # секунд на загрузку одного изображения
MAX_IMAGE_BYTES = 3_000_000  # больше — пропускаем
MAX_IMAGES = 20              # жёсткий потолок обрабатываемых изображений
DEFAULT_IMAGES = 12
PREVIEW_SIDE = 320           # максимальная сторона превью в ответе, px
CONCURRENCY = 8
USER_AGENT = "ImageScope-Hackathon/1.0"
