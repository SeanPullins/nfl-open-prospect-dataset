let rows = [];

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = parseLine(lines[0]);

  return lines.slice(1).map(line => {
    const values = parseLine(line);
    const obj = {};
    headers.forEach((h, i) => obj[h] = values[i] ?? "");
    return obj;
  });
}

function parseLine(line) {
  const result = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    const next = line[i + 1];

    if (ch === '"' && inQuotes && next === '"') {
      cur += '"';
      i++;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      result.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }

  result.push(cur);
  return result;
}

function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function pct(x) {
  const n = num(x);
  if (n === null) return "—";
  return `${Math.round(n * 100)}%`;
}

function score(x) {
  const n = num(x);
  if (n === null) return "—";
  return n.toFixed(1);
}

function populateFilters() {
  const years = [...new Set(rows.map(r => r.draft_year).filter(Boolean))].sort((a, b) => Number(b) - Number(a));
  const positions = [...new Set(rows.map(r => r.position_group).filter(Boolean))].sort();

  const yearFilter = document.getElementById("yearFilter");
  const positionFilter = document.getElementById("positionFilter");

  years.forEach(y => {
    const opt = document.createElement("option");
    opt.value = y;
    opt.textContent = y;
    yearFilter.appendChild(opt);
  });

  positions.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p;
    positionFilter.appendChild(opt);
  });
}

function render() {
  const q = document.getElementById("searchInput").value.toLowerCase().trim();
  const year = document.getElementById("yearFilter").value;
  const pos = document.getElementById("positionFilter").value;
  const sort = document.getElementById("sortFilter").value;

  let filtered = rows.filter(r => {
    const hay = `${r.player} ${r.college} ${r.position} ${r.position_group} ${r.outcome_tier}`.toLowerCase();
    if (q && !hay.includes(q)) return false;
    if (year && r.draft_year !== year) return false;
    if (pos && r.position_group !== pos) return false;
    return true;
  });

  filtered.sort((a, b) => {
    const av = num(a[sort]);
    const bv = num(b[sort]);

    if (sort.includes("rank")) {
      return (av ?? 999999) - (bv ?? 999999);
    }

    return (bv ?? -999999) - (av ?? -999999);
  });

  const count = document.getElementById("count");
  count.textContent = `${filtered.length.toLocaleString()} players`;

  const cards = document.getElementById("cards");
  cards.innerHTML = "";

  filtered.slice(0, 250).forEach(r => {
    const card = document.createElement("article");
    card.className = "card";

    card.innerHTML = `
      <div class="card-top">
        <div>
          <div class="name">${escapeHtml(r.player)}</div>
          <div class="meta">${escapeHtml(r.draft_year)} · ${escapeHtml(r.position)} · ${escapeHtml(r.college)} · Pick ${escapeHtml(r.pick)}</div>
        </div>
        <div>
          <div class="grade">${score(r.outcome_grade_pff_powered)}</div>
          <div class="tier">${escapeHtml(r.outcome_tier)}</div>
        </div>
      </div>

      <div class="badges">
        <span class="badge">Class #${escapeHtml(r.overall_rank_in_class || "—")}</span>
        <span class="badge">${escapeHtml(r.position_group)} #${escapeHtml(r.position_rank_in_class || "—")}</span>
        <span class="badge">${escapeHtml(r.draft_value_vs_grade || "—")}</span>
        <span class="badge">${escapeHtml(r.confidence_label || "—")}</span>
      </div>

      <div class="summary">${escapeHtml(r.player_card_summary || "")}</div>

      <div class="probs">
        <div class="prob">Starter <strong>${pct(r.starter_probability)}</strong></div>
        <div class="prob">Elite <strong>${pct(r.elite_probability)}</strong></div>
        <div class="prob">Bust <strong>${pct(r.bust_probability)}</strong></div>
      </div>

      <div class="comps">
        <strong>Comps:</strong> ${escapeHtml(r.historical_position_comps || "—")}
      </div>
    `;

    cards.appendChild(card);
  });
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function init() {
  const response = await fetch("data/player_cards_v8.csv");
  const text = await response.text();
  rows = parseCSV(text);

  populateFilters();
  render();

  document.getElementById("searchInput").addEventListener("input", render);
  document.getElementById("yearFilter").addEventListener("change", render);
  document.getElementById("positionFilter").addEventListener("change", render);
  document.getElementById("sortFilter").addEventListener("change", render);
}

init();
