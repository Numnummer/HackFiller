const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let filters = [];
let shown = [];
let cur = -1;

function showStatus(text, kind) {
  const el = $("status");
  el.hidden = !text;
  el.className = "status " + (kind || "");
  el.textContent = text || "";
}

function updateDesc() {
  const f = filters.find((x) => x.name === $("filter").value);
  $("filterDesc").textContent = f ? `[${f.engine}] ${f.description}` : "";
}

async function init() {
  try {
    const [src, fl] = await Promise.all([
      fetch("/api/source").then((r) => r.json()),
      fetch("/api/filters").then((r) => r.json()),
    ]);
    $("srcLink").textContent = `${src.name} — ${src.url}`;
    $("srcLink").href = src.url;
    filters = fl;
    const groups = { preset: "Пресеты", numpy: "NumPy-фильтры", torch: "PyTorch-свёртки" };
    $("filter").innerHTML = Object.entries(groups).map(([eng, label]) =>
      `<optgroup label="${label}">` +
      fl.filter((f) => f.engine === eng).map((f) => `<option value="${f.name}">${esc(f.title)}</option>`).join("") +
      "</optgroup>").join("");
    updateDesc();
  } catch (e) {
    showStatus("Не удалось загрузить настройки: " + e.message, "error");
  }
}

function histChart(before, after) {
  const max = Math.max(...before, ...after, 0.001);
  $("hist").innerHTML = before.map((v, i) =>
    `<div class="col" title="яркость ${i * 16}–${i * 16 + 15}: до ${(v * 100).toFixed(1)}%, после ${(after[i] * 100).toFixed(1)}%">` +
    `<i class="a" style="height:${(v / max) * 100}%"></i><i class="b" style="height:${(after[i] / max) * 100}%"></i></div>`).join("");
}

function render(d) {
  const s = d.summary;
  const card = (v, l) => `<div class="card"><b>${v}</b><span>${l}</span></div>`;
  $("cards").innerHTML =
    card(d.images_found, "найдено изображений") + card(s.processed, "обработано") + card(s.failed, "пропущено") +
    card(`${s.avg_brightness} → ${s.avg_result_brightness}`, "средняя яркость до → после") +
    card(`${s.avg_contrast} → ${s.avg_result_contrast}`, "средний контраст до → после") +
    card(d.elapsed_sec + " c", "время");
  histChart(s.histogram, s.result_histogram);

  $("errPanel").hidden = !d.errors.length;
  $("errors").innerHTML = d.errors.map((e) =>
    `<li>[${e.stage}] ${esc(e.url || "")} — ${esc(e.message)}</li>`).join("");

  shown = d.images;
  $("gallery").innerHTML = d.images.map((im, idx) => {
    const a = im.stats, b = im.result_stats;
    const chain = im.filters.map((f) => `<span class="tag">${f.name} (${f.engine}) ×${f.intensity}</span>`).join("");
    const flags = a.flags.map((f) => `<span class="tag warn">${esc(f)}</span>`).join("");
    return `<div class="item"><h3 title="${esc(im.title)}">${esc(im.title || im.url)}</h3>
      <div class="pair">
        <figure><img src="${im.original}" alt="до" data-i="${idx}"><figcaption>до</figcaption></figure>
        <figure><img src="${im.processed}" alt="после" data-i="${idx}"><figcaption>после</figcaption></figure>
      </div>
      <div class="meta">
        <span>Размер: ${a.width}×${a.height} (${a.aspect_ratio})</span>
        <span>Ср. цвет: <i style="display:inline-block;width:12px;height:12px;background:rgb(${a.mean_color});border:1px solid #888;vertical-align:-1px"></i> rgb(${a.mean_color})</span>
        <span>Яркость: ${a.brightness} → ${b.brightness}</span>
        <span>Контраст: ${a.contrast} → ${b.contrast}</span>
      </div>
      <div>${chain}${flags}</div></div>`;
  }).join("");
  $("report").hidden = false;
}

async function run() {
  $("run").disabled = true;
  $("report").hidden = true;
  showStatus("Собираем страницы, загружаем и обрабатываем изображения…", "loading");
  try {
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filter: $("filter").value,
        intensity: parseFloat($("intensity").value),
        limit: parseInt($("limit").value, 10),
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    render(data);
    showStatus(data.errors.length ? `Готово, но часть пропущена (${data.errors.length}) — см. список ошибок.` : "", "");
  } catch (e) {
    showStatus("Ошибка: " + e.message, "error");
  } finally {
    $("run").disabled = false;
  }
}

$("filter").addEventListener("change", updateDesc);
$("intensity").addEventListener("input", (e) => ($("intVal").textContent = (+e.target.value).toFixed(2)));
$("limit").addEventListener("input", (e) => ($("limVal").textContent = e.target.value));
$("run").addEventListener("click", run);
init();

function openModal(i) {
  if (!shown.length) return;
  cur = (i + shown.length) % shown.length;
  const im = shown[cur], a = im.stats, b = im.result_stats;
  $("mTitle").textContent = `${im.title || im.url} (${cur + 1}/${shown.length})`;
  $("mBefore").src = im.original_full;  // сначала грузится полноразмерный JPEG
  $("mAfter").src = im.processed_full;
  $("mMeta").textContent = `${a.width}×${a.height} · яркость ${a.brightness} → ${b.brightness} · контраст ${a.contrast} → ${b.contrast} · ` +
    im.filters.map((f) => `${f.name} (${f.engine}) ×${f.intensity}`).join(", ");
  $("modal").hidden = false;
}
function closeModal() { $("modal").hidden = true; cur = -1; }

$("gallery").addEventListener("click", (e) => {
  const i = e.target.dataset && e.target.dataset.i;
  if (e.target.tagName === "IMG" && i !== undefined) openModal(+i);
});
$("mClose").addEventListener("click", closeModal);
$("mPrev").addEventListener("click", () => openModal(cur - 1));
$("mNext").addEventListener("click", () => openModal(cur + 1));
$("modal").addEventListener("click", (e) => { if (e.target === $("modal")) closeModal(); });
document.addEventListener("keydown", (e) => {
  if ($("modal").hidden) return;
  if (e.key === "Escape") closeModal();
  if (e.key === "ArrowLeft") openModal(cur - 1);
  if (e.key === "ArrowRight") openModal(cur + 1);
});
