let rows = [];

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = parseLine(lines[0]);

  return lines.slice(1).filter(Boolean).map(line => {
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

function cleanInt(x) {
  const n = Number(x);
  if (Number.isFinite(n)) return String(Math.round(n));
  return x ? String(x) : "—";
}


function missTypeLabel(r) {
  if (r.value_flag === "good_miss") return "Good Miss / Value";
  if (r.miss_flag === "bad_miss") return "Bad Miss";
  return "Neutral / Expected";
}

function pickLabel(r) {
  const pick = num(r.pick);
  if (pick === null) return "UDFA/—";
  return `Pick ${Math.round(pick)}`;
}

function roundLabel(r) {
  const round = num(r.round);
  if (round === null) return "";
  return `R${Math.round(round)}`;
}


function cleanYear(x) {
  const n = Number(x);
  if (Number.isFinite(n)) return String(Math.round(n));
  return x ? String(x).replace(/\.0$/, "") : "";
}


function tagHtml(label, className = "tag") {
  if (!label || label === "nan" || label === "undefined") return "";
  return `<span class="${className}">${escapeHtml(label)}</span>`;
}

function populateFilters() {
  const years = [...new Set(rows.map(r => cleanYear(r.draft_year)).filter(Boolean))].sort((a, b) => Number(b) - Number(a));
  const positions = [...new Set(rows.map(r => r.position_group).filter(Boolean))].sort();

  const yearFilter = document.getElementById("yearFilter");
  const positionFilter = document.getElementById("positionFilter");

  yearFilter.innerHTML = '<option value="">All Years</option>';
  positionFilter.innerHTML = '<option value="">All Positions</option>';

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
  const outcome = document.getElementById("outcomeFilter") ? document.getElementById("outcomeFilter").value : "";
  const sort = document.getElementById("sortFilter").value;

  let filtered = rows.filter(r => {
    const hay = `${r.player} ${r.college} ${r.position} ${r.position_group} ${r.outcome_tier} ${r.actual_outcome_flag}`.toLowerCase();

    if (q && !hay.includes(q)) return false;
    if (year && cleanYear(r.draft_year) !== String(year)) return false;
    if (pos && r.position_group !== pos) return false;

    if (outcome === "good_miss" && r.value_flag !== "good_miss") return false;
    if (outcome === "bad_miss" && r.miss_flag !== "bad_miss") return false;
    if (outcome === "neutral" && (r.value_flag === "good_miss" || r.miss_flag === "bad_miss")) return false;

    return true;
  });

  filtered.sort((a, b) => {
    if (sort === "pick_asc") {
      const ay = num(a.draft_year) ?? 999999;
      const by = num(b.draft_year) ?? 999999;
      const ap = num(a.pick) ?? 999999;
      const bp = num(b.pick) ?? 999999;

      if (ay !== by) return by - ay;
      return ap - bp;
    }

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
  cards.className = "player-list";

  filtered.slice(0, 500).forEach((r, idx) => {
    const projectionBadgeHtml = tagHtml(r.projection_badge, "tag projection-tag");
    const careerProjectionBadgeHtml = tagHtml(r.career_projection_label, "tag career-projection-tag");

    const typeLabel = missTypeLabel(r);

    const item = document.createElement("article");
    item.className = `card player-row ${r.outcome_card_class || ""}`;

    const detailsId = `details-${idx}`;

    item.innerHTML = `
      <button class="row-summary" type="button" aria-expanded="false" aria-controls="${detailsId}">
        <div class="row-pick">
          <strong>${escapeHtml(pickLabel(r))}</strong>
          <span>${escapeHtml(roundLabel(r))}</span>
        </div>

        <div class="row-main">
          <div class="row-name">${escapeHtml(r.player)}</div>
          <div class="row-meta">${escapeHtml(cleanYear(r.draft_year))} · ${escapeHtml(r.position)} · ${escapeHtml(r.college)}</div>
          <div class="tag-row">${projectionBadgeHtml}${careerProjectionBadgeHtml}</div>
        </div>

        <div class="row-regrade">
          <span>Should Have Gone</span>
          <strong>${escapeHtml(r.should_have_been_drafted || "—")}</strong>
        </div>

        <div class="row-outcome">
          <span>${escapeHtml(r.actual_outcome_flag || "—")}</span>
          <strong>${escapeHtml(typeLabel)}</strong>
        </div>

        <div class="row-score">
          <span>${escapeHtml(r.display_grade_label || "Grade")}</span>
          <strong>${score(r.display_grade || r.outcome_grade_pff_powered)}</strong>
        </div>

        <div class="row-chevron">▾</div>
      </button>

      <div class="row-details" id="${detailsId}" hidden>
        <div class="badges">
          <span class="badge">Class #${escapeHtml(r.overall_rank_in_class || "—")}</span>
          <span class="badge">${escapeHtml(r.position_group)} #${escapeHtml(r.position_rank_in_class || "—")}</span>
          <span class="badge">${escapeHtml(r.draft_value_vs_grade || "—")}</span>
          <span class="badge outcome-badge">${escapeHtml(r.actual_outcome_flag || "—")}</span>
          <span class="badge outcome-badge">${escapeHtml(typeLabel)}</span>
          <span class="badge">${escapeHtml(r.should_have_been_drafted || "—")}</span>
          <span class="badge">${escapeHtml(r.draft_slot_regrade || "—")}</span>
          <span class="badge">${escapeHtml(r.confidence_label || "—")}</span>
        </div>

        <div class="draft-regrade">
          <div class="draft-regrade-title">Draft Slot Regrade</div>
          <div class="draft-regrade-grid">
            <div>
              <span>Drafted</span>
              <strong>${escapeHtml(pickLabel(r))}</strong>
            </div>
            <div>
              <span>Should Have Gone</span>
              <strong>${escapeHtml(r.should_have_been_drafted || "—")}</strong>
            </div>
            <div>
              <span>Regrade</span>
              <strong>${escapeHtml(r.draft_slot_regrade || "—")}</strong>
            </div>
          </div>
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
      </div>
    `;

    const button = item.querySelector(".row-summary");
    const details = item.querySelector(".row-details");

    button.addEventListener("click", () => {
      const isOpen = !details.hidden;
      details.hidden = isOpen;
      button.setAttribute("aria-expanded", String(!isOpen));
      item.classList.toggle("is-expanded", !isOpen);
    });

    cards.appendChild(item);
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
  const response = await fetch("data/player_cards_v10.csv", { cache: "no-store" });
  const text = await response.text();
  rows = parseCSV(text);

  populateFilters();
  render();

  document.getElementById("searchInput").addEventListener("input", render);
  document.getElementById("yearFilter").addEventListener("change", render);
  document.getElementById("positionFilter").addEventListener("change", render);
  if (document.getElementById("outcomeFilter")) {
    document.getElementById("outcomeFilter").addEventListener("change", render);
  }
  document.getElementById("sortFilter").addEventListener("change", render);
}

init();
