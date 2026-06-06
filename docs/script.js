let historicalRows = [];
let projectionRows2026 = [];
let futureRows = [];
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

function normalizeHistorical(r) {
  return {
    dataset: "Historical",
    player: r.player,
    draft_year: r.draft_year,
    position: r.position,
    position_group: r.position_group,
    college: r.college,
    pick: r.pick,
    grade: r.outcome_grade_pff_powered,
    tier: r.outcome_tier,
    overall_rank: r.overall_rank_in_class,
    position_rank: r.position_rank_in_class,
    draft_value: r.draft_value_vs_grade,
    confidence: r.confidence_label,
    starter_probability: r.starter_probability,
    elite_probability: r.elite_probability,
    bust_probability: r.bust_probability,
    summary: r.player_card_summary,
    comps: r.historical_position_comps
  };
}

function normalizeFuture(r) {
  return {
    dataset: "Future Class",
    player: r.player,
    draft_year: r.draft_year,
    position: r.position,
    position_group: r.position_group,
    college: r.college,
    pick: r.projected_pick,
    grade: r.projection_score,
    tier: r.projection_tier,
    overall_rank: r.watchlist_rank,
    position_rank: r.position_rank || "",
    draft_value: r.projected_pick ? `Projected pick ${r.projected_pick}` : "Future watchlist",
    confidence: r.class_status || "Future",
    starter_probability: r.starter_probability,
    elite_probability: r.elite_probability,
    bust_probability: r.bust_probability,
    summary: `${r.player} (${r.draft_year}, ${r.position}, ${r.college}) is listed as a future-class player. ${r.projection_explanation || ""}`,
    comps: `College stats: ${r.college_stats_status || "pending"} · PFF: ${r.pff_status || "pending"} · All-22: ${r.all22_status || "pending"}`
  };
}

function normalize2026(r) {
  return {
    dataset: "2026 Projection",
    player: r.player,
    draft_year: r.draft_year || "2026",
    position: r.position,
    position_group: r.position_group,
    college: r.college || r.school,
    pick: r.projected_pick,
    grade: r.projection_score,
    tier: r.projection_tier,
    overall_rank: r.overall_rank_2026,
    position_rank: r.position_rank_2026,
    draft_value: `Projected pick ${r.projected_pick || "—"}`,
    confidence: "Projection",
    starter_probability: r.starter_probability,
    elite_probability: r.elite_probability,
    bust_probability: r.bust_probability,
    summary: `${r.player} (${r.draft_year || "2026"}, ${r.position}, ${r.college || r.school || ""}) has a projection score of ${score(r.projection_score)}. ${r.projection_explanation || ""}`,
    comps: "Projection-only card"
  };
}

function setDataset() {
  const dataset = document.getElementById("datasetFilter").value;

  if (dataset === "projections2026") {
    rows = projectionRows2026.map(normalize2026);
    document.getElementById("downloadLink").href = "data/prospect_projections_2026_v1.csv";
  } else if (dataset === "future") {
    rows = futureRows.map(normalizeFuture);
    document.getElementById("downloadLink").href = "data/future_prospects.csv";
  } else {
    rows = historicalRows.map(normalizeHistorical);
    document.getElementById("downloadLink").href = "data/player_cards_v8.csv";
  }

  populateFilters();
  render();
}

function populateFilters() {
  const yearFilter = document.getElementById("yearFilter");
  const positionFilter = document.getElementById("positionFilter");

  yearFilter.innerHTML = '<option value="">All Years</option>';
  positionFilter.innerHTML = '<option value="">All Positions</option>';

  const years = [...new Set(rows.map(r => r.draft_year).filter(Boolean))].sort((a, b) => Number(b) - Number(a));
  const positions = [...new Set(rows.map(r => r.position_group).filter(Boolean))].sort();

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
    const hay = `${r.player} ${r.college} ${r.position} ${r.position_group} ${r.tier}`.toLowerCase();
    if (q && !hay.includes(q)) return false;
    if (year && String(r.draft_year) !== String(year)) return false;
    if (pos && r.position_group !== pos) return false;
    return true;
  });

  filtered.sort((a, b) => {
    if (sort === "draft_year") return Number(b.draft_year || 0) - Number(a.draft_year || 0);
    if (sort === "overall_rank") return Number(a.overall_rank || 999999) - Number(b.overall_rank || 999999);
    if (sort === "position_rank") return Number(a.position_rank || 999999) - Number(b.position_rank || 999999);
    if (sort === "elite_probability") return Number(b.elite_probability || 0) - Number(a.elite_probability || 0);
    if (sort === "bust_probability") return Number(a.bust_probability || 999) - Number(b.bust_probability || 999);
    return Number(b.grade || 0) - Number(a.grade || 0);
  });

  document.getElementById("count").textContent = `${filtered.length.toLocaleString()} players`;

  const cards = document.getElementById("cards");
  cards.innerHTML = "";

  filtered.slice(0, 250).forEach(r => {
    const card = document.createElement("article");
    card.className = "card";

    card.innerHTML = `
      <div class="card-top">
        <div>
          <div class="name">${escapeHtml(r.player)}</div>
          <div class="meta">${escapeHtml(r.dataset)} · ${escapeHtml(r.draft_year)} · ${escapeHtml(r.position)} · ${escapeHtml(r.college)} · Pick ${escapeHtml(r.pick || "—")}</div>
        </div>
        <div>
          <div class="grade">${score(r.grade)}</div>
          <div class="tier">${escapeHtml(r.tier || "")}</div>
        </div>
      </div>

      <div class="badges">
        <span class="badge">Class #${escapeHtml(r.overall_rank || "—")}</span>
        <span class="badge">${escapeHtml(r.position_group)} #${escapeHtml(r.position_rank || "—")}</span>
        <span class="badge">${escapeHtml(r.draft_value || "—")}</span>
        <span class="badge">${escapeHtml(r.confidence || "—")}</span>
      </div>

      <div class="summary">${escapeHtml(r.summary || "")}</div>

      <div class="probs">
        <div class="prob">Starter <strong>${pct(r.starter_probability)}</strong></div>
        <div class="prob">Elite <strong>${pct(r.elite_probability)}</strong></div>
        <div class="prob">Bust <strong>${pct(r.bust_probability)}</strong></div>
      </div>

      <div class="comps">
        <strong>Comps:</strong> ${escapeHtml(r.comps || "—")}
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

async function loadCSV(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return parseCSV(await response.text());
}

async function init() {
  historicalRows = await loadCSV("data/player_cards_v8.csv");
  projectionRows2026 = await loadCSV("data/prospect_projections_2026_v1.csv");
  futureRows = await loadCSV("data/future_prospects.csv");

  setDataset();

  document.getElementById("datasetFilter").addEventListener("change", setDataset);
  document.getElementById("searchInput").addEventListener("input", render);
  document.getElementById("yearFilter").addEventListener("change", render);
  document.getElementById("positionFilter").addEventListener("change", render);
  document.getElementById("sortFilter").addEventListener("change", render);
}

init();
