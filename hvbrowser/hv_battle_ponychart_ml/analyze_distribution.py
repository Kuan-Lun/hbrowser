# ruff: noqa: E501
"""原圖角色分布分析 — Flask Web UI.

啟動本地 Flask server，以互動式圖表呈現原圖的角色分布。
每張圖都可切換「僅原圖」/「全部（含 crop）」以觀察分布變化。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.analyze_distribution
  # 瀏覽器開啟 http://localhost:5000
"""

from __future__ import annotations

import json
import os
import webbrowser
from collections import Counter
from itertools import combinations
from typing import Any

from flask import Flask, render_template_string

from .common import CLASS_NAMES, LABELS_FILE, NUM_CLASSES
from .common.data import is_original

app = Flask(__name__)

# ── Pony character colours (consistent across all charts) ──
CHAR_COLORS = [
    "#9B59B6",  # Twilight Sparkle – purple
    "#ECF0F1",  # Rarity – white/light
    "#F1C40F",  # Fluttershy – yellow
    "#E74C3C",  # Rainbow Dash – red (primary rainbow)
    "#FF69B4",  # Pinkie Pie – pink
    "#E67E22",  # Applejack – orange
]
SHORT_NAMES = ["TS", "RA", "FS", "RD", "PP", "AJ"]


def _compute_stats_for(
    samples: dict[str, list[int]],
) -> dict[str, Any]:
    """Compute distribution statistics for a given set of samples."""
    total = len(samples)

    # 1. Label count distribution
    cnt_dist = Counter(len(v) for v in samples.values())
    label_counts = {n: cnt_dist.get(n, 0) for n in (1, 2, 3)}

    # 2. Single-label character distribution
    single = [v[0] for v in samples.values() if len(v) == 1]
    single_cnt = Counter(single)
    single_chars = [single_cnt.get(i + 1, 0) for i in range(NUM_CLASSES)]

    # 3. Double-label combinations
    double = [tuple(sorted(v)) for v in samples.values() if len(v) == 2]
    double_cnt = Counter(double)
    all_combos_2 = list(combinations(range(1, NUM_CLASSES + 1), 2))
    double_data: list[dict[str, Any]] = [
        {
            "label": f"{SHORT_NAMES[c[0]-1]}+{SHORT_NAMES[c[1]-1]}",
            "full": f"{CLASS_NAMES[c[0]-1]} + {CLASS_NAMES[c[1]-1]}",
            "count": double_cnt.get(c, 0),
        }
        for c in all_combos_2
    ]
    double_data.sort(key=lambda x: x["count"], reverse=True)

    # 4. Triple-label combinations
    triple = [tuple(sorted(v)) for v in samples.values() if len(v) == 3]
    triple_cnt = Counter(triple)
    all_combos_3 = list(combinations(range(1, NUM_CLASSES + 1), 3))
    triple_data: list[dict[str, Any]] = [
        {
            "label": "+".join(SHORT_NAMES[x - 1] for x in c),
            "full": " + ".join(CLASS_NAMES[x - 1] for x in c),
            "count": triple_cnt.get(c, 0),
        }
        for c in all_combos_3
    ]
    triple_data.sort(key=lambda x: x["count"], reverse=True)

    # 5. Overall occurrence rate
    overall = [0] * NUM_CLASSES
    for v in samples.values():
        for lbl in v:
            overall[lbl - 1] += 1

    # 6. Co-occurrence matrix & conditional rates
    cooc = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for v in samples.values():
        for a in v:
            for b in v:
                cooc[a - 1][b - 1] += 1

    cond_rates = [[0.0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and overall[i] > 0:
                cond_rates[i][j] = round(cooc[i][j] / overall[i] * 100, 1)

    # Pre-compute max values for heatmap colouring
    flat_cooc = [cooc[i][j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)]
    cooc_max = max(flat_cooc) if flat_cooc else 1
    flat_cond = [
        cond_rates[i][j]
        for i in range(NUM_CLASSES)
        for j in range(NUM_CLASSES)
        if i != j
    ]
    cond_max = max(flat_cond) if flat_cond else 1

    return {
        "total": total,
        "label_counts": {str(n): label_counts[n] for n in (1, 2, 3)},
        "single_chars": single_chars,
        "double_data": double_data,
        "triple_data": triple_data,
        "overall": overall,
        "cooc": cooc,
        "cond_rates": cond_rates,
        "cooc_max": cooc_max,
        "cond_max": cond_max,
    }


def _load_all_stats() -> dict[str, Any]:
    """Load labels and compute stats for both orig-only and all samples."""
    with open(LABELS_FILE, encoding="utf-8") as f:
        raw: dict[str, list[int]] = json.load(f)

    orig = {
        k: v for k, v in raw.items() if is_original(os.path.basename(k))
    }

    return {
        "orig": _compute_stats_for(orig),
        "all": _compute_stats_for(raw),
        "class_names": CLASS_NAMES,
        "short_names": SHORT_NAMES,
        "char_colors": CHAR_COLORS,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PonyChart 角色分布分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { --bg: #1a1a2e; --card: #16213e; --text: #e0e0e0; --border: #0f3460; --accent: #e74c3c; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }
  h1 { text-align: center; margin-bottom: 8px; font-size: 1.6em; }
  .summary { text-align: center; margin-bottom: 24px; opacity: 0.8; font-size: 0.95em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .card-header h2 { font-size: 1em; opacity: 0.9; margin: 0; }
  canvas { width: 100% !important; }
  /* Toggle switch */
  .toggle { display: flex; align-items: center; gap: 6px; font-size: 0.8em; white-space: nowrap; }
  .toggle-label { opacity: 0.6; transition: opacity 0.2s; }
  .toggle-label.active { opacity: 1; font-weight: 600; }
  .toggle-switch { position: relative; width: 36px; height: 20px; cursor: pointer; }
  .toggle-switch input { opacity: 0; width: 0; height: 0; }
  .toggle-slider { position: absolute; inset: 0; background: #3498db; border-radius: 10px; transition: background 0.3s; }
  .toggle-slider::before { content: ''; position: absolute; height: 14px; width: 14px; left: 3px; bottom: 3px; background: white; border-radius: 50%; transition: transform 0.3s; }
  .toggle-switch input:checked + .toggle-slider { background: var(--accent); }
  .toggle-switch input:checked + .toggle-slider::before { transform: translateX(16px); }
  /* Heatmap table */
  .heatmap-container { overflow-x: auto; }
  .heatmap { border-collapse: collapse; width: 100%; font-size: 0.85em; }
  .heatmap th, .heatmap td { padding: 8px 6px; text-align: center; border: 1px solid var(--border); min-width: 60px; }
  .heatmap th { background: var(--bg); font-weight: 600; }
  .heatmap .row-header { text-align: right; font-weight: 600; background: var(--bg); }
  .tabs { display: flex; gap: 8px; margin-bottom: 12px; }
  .tabs button { background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85em; }
  .tabs button.active { background: var(--border); }
</style>
</head>
<body>

<h1>PonyChart 角色分布分析</h1>
<div class="summary" id="summary"></div>

<div class="grid">
  <!-- 1. Label count distribution -->
  <div class="card">
    <div class="card-header">
      <h2>1. 標籤數量分布</h2>
      <div class="toggle">
        <span class="toggle-label active" data-side="orig">原圖</span>
        <label class="toggle-switch"><input type="checkbox" onchange="toggleChart('labelCount', this.checked)"><span class="toggle-slider"></span></label>
        <span class="toggle-label" data-side="all">全部</span>
      </div>
    </div>
    <canvas id="chart-label-count"></canvas>
  </div>

  <!-- 2. Single-label character distribution -->
  <div class="card">
    <div class="card-header">
      <h2 id="title-single">2. 單標籤角色分布</h2>
      <div class="toggle">
        <span class="toggle-label active" data-side="orig">原圖</span>
        <label class="toggle-switch"><input type="checkbox" onchange="toggleChart('single', this.checked)"><span class="toggle-slider"></span></label>
        <span class="toggle-label" data-side="all">全部</span>
      </div>
    </div>
    <canvas id="chart-single"></canvas>
  </div>

  <!-- 3. Double-label combos -->
  <div class="card">
    <div class="card-header">
      <h2 id="title-double">3. 雙標籤組合分布</h2>
      <div class="toggle">
        <span class="toggle-label active" data-side="orig">原圖</span>
        <label class="toggle-switch"><input type="checkbox" onchange="toggleChart('double', this.checked)"><span class="toggle-slider"></span></label>
        <span class="toggle-label" data-side="all">全部</span>
      </div>
    </div>
    <canvas id="chart-double" height="280"></canvas>
  </div>

  <!-- 4. Triple-label combos -->
  <div class="card">
    <div class="card-header">
      <h2 id="title-triple">4. 三標籤組合分布</h2>
      <div class="toggle">
        <span class="toggle-label active" data-side="orig">原圖</span>
        <label class="toggle-switch"><input type="checkbox" onchange="toggleChart('triple', this.checked)"><span class="toggle-slider"></span></label>
        <span class="toggle-label" data-side="all">全部</span>
      </div>
    </div>
    <canvas id="chart-triple" height="360"></canvas>
  </div>

  <!-- 5. Overall occurrence -->
  <div class="card">
    <div class="card-header">
      <h2>5. 各角色整體出現率</h2>
      <div class="toggle">
        <span class="toggle-label active" data-side="orig">原圖</span>
        <label class="toggle-switch"><input type="checkbox" onchange="toggleChart('overall', this.checked)"><span class="toggle-slider"></span></label>
        <span class="toggle-label" data-side="all">全部</span>
      </div>
    </div>
    <canvas id="chart-overall"></canvas>
  </div>

  <!-- 6. Co-occurrence heatmap -->
  <div class="card">
    <div class="card-header">
      <h2>6. 角色共現矩陣</h2>
      <div class="toggle">
        <span class="toggle-label active" data-side="orig">原圖</span>
        <label class="toggle-switch"><input type="checkbox" onchange="toggleHeatmap(this.checked)"><span class="toggle-slider"></span></label>
        <span class="toggle-label" data-side="all">全部</span>
      </div>
    </div>
    <div class="tabs">
      <button class="active" onclick="showHeatmapMode('count', this)">共現次數</button>
      <button onclick="showHeatmapMode('rate', this)">條件機率 P(col|row)</button>
    </div>
    <div class="heatmap-container" id="heatmap-container"></div>
  </div>
</div>

<script>
Chart.defaults.color = '#e0e0e0';
Chart.defaults.borderColor = 'rgba(255,255,255,0.08)';

const NAMES = {{ data.class_names | tojson }};
const COLORS = {{ data.char_colors | tojson }};
const SHORT = {{ data.short_names | tojson }};
const D = {
  orig: {{ data.orig | tojson }},
  all: {{ data.all | tojson }},
};

// Track current source per chart
const chartSrc = { labelCount: 'orig', single: 'orig', double: 'orig', triple: 'orig', overall: 'orig' };
let heatmapSrc = 'orig';
let heatmapMode = 'count';

// ── Summary ──
function updateSummary() {
  // Show summary for both modes for quick reference
  const o = D.orig, a = D.all;
  document.getElementById('summary').innerHTML =
    `原圖: <strong>${o.total}</strong> 張 &nbsp;|&nbsp; 全部（含 crop）: <strong>${a.total}</strong> 張`;
}
updateSummary();

// ── Helper: update toggle label styling ──
function updateToggleLabels(checkbox, showAll) {
  const container = checkbox.closest('.toggle');
  container.querySelector('[data-side="orig"]').classList.toggle('active', !showAll);
  container.querySelector('[data-side="all"]').classList.toggle('active', showAll);
}

// ── 1. Label count (doughnut) ──
const chartLabelCount = new Chart(document.getElementById('chart-label-count'), {
  type: 'doughnut',
  data: {
    labels: ['1 個標籤', '2 個標籤', '3 個標籤'],
    datasets: [{
      data: [D.orig.label_counts['1'], D.orig.label_counts['2'], D.orig.label_counts['3']],
      backgroundColor: ['#3498db', '#2ecc71', '#e74c3c'],
      borderWidth: 0,
    }]
  },
  options: {
    plugins: {
      legend: { position: 'bottom' },
      tooltip: { callbacks: { label: ctx => {
        const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
        return `${ctx.label}: ${ctx.raw} (${(ctx.raw / total * 100).toFixed(1)}%)`;
      }}}
    }
  }
});

// ── 2. Single-label (bar) ──
const chartSingle = new Chart(document.getElementById('chart-single'), {
  type: 'bar',
  data: {
    labels: NAMES,
    datasets: [{ data: D.orig.single_chars.slice(), backgroundColor: COLORS, borderWidth: 0 }]
  },
  options: {
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => {
      const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
      return `${ctx.raw} (${(ctx.raw / total * 100).toFixed(1)}%)`;
    }}}},
    scales: { y: { beginAtZero: true } }
  }
});

// ── 3. Double-label combos (horizontal bar) ──
const chartDouble = new Chart(document.getElementById('chart-double'), {
  type: 'bar',
  data: {
    labels: D.orig.double_data.map(d => d.label),
    datasets: [{ data: D.orig.double_data.map(d => d.count), backgroundColor: '#3498db', borderWidth: 0 }]
  },
  options: {
    indexAxis: 'y',
    plugins: { legend: { display: false }, tooltip: { callbacks: {
      title: ctx => D[chartSrc.double].double_data[ctx[0].dataIndex].full,
      label: ctx => {
        const total = D[chartSrc.double].double_data.reduce((a, d) => a + d.count, 0);
        return `${ctx.raw} (${(ctx.raw / total * 100).toFixed(1)}%)`;
      }
    }}},
    scales: { x: { beginAtZero: true } }
  }
});

// ── 4. Triple-label combos (horizontal bar) ──
const chartTriple = new Chart(document.getElementById('chart-triple'), {
  type: 'bar',
  data: {
    labels: D.orig.triple_data.map(d => d.label),
    datasets: [{ data: D.orig.triple_data.map(d => d.count), backgroundColor: '#2ecc71', borderWidth: 0 }]
  },
  options: {
    indexAxis: 'y',
    plugins: { legend: { display: false }, tooltip: { callbacks: {
      title: ctx => D[chartSrc.triple].triple_data[ctx[0].dataIndex].full,
      label: ctx => {
        const total = D[chartSrc.triple].triple_data.reduce((a, d) => a + d.count, 0);
        return `${ctx.raw} (${(ctx.raw / total * 100).toFixed(1)}%)`;
      }
    }}},
    scales: { x: { beginAtZero: true } }
  }
});

// ── 5. Overall occurrence (bar) ──
const chartOverall = new Chart(document.getElementById('chart-overall'), {
  type: 'bar',
  data: {
    labels: NAMES,
    datasets: [{ data: D.orig.overall.slice(), backgroundColor: COLORS, borderWidth: 0 }]
  },
  options: {
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => {
      const total = D[chartSrc.overall].total;
      return `${ctx.raw} / ${total} (${(ctx.raw / total * 100).toFixed(1)}%)`;
    }}}},
    scales: { y: { beginAtZero: true } }
  }
});

// ── Chart toggle handler ──
function toggleChart(chartId, showAll) {
  const src = showAll ? 'all' : 'orig';
  chartSrc[chartId] = src;
  const s = D[src];

  // Update toggle label styling
  const checkbox = event.target;
  updateToggleLabels(checkbox, showAll);

  switch (chartId) {
    case 'labelCount':
      chartLabelCount.data.datasets[0].data = [s.label_counts['1'], s.label_counts['2'], s.label_counts['3']];
      chartLabelCount.update();
      break;
    case 'single':
      chartSingle.data.datasets[0].data = s.single_chars.slice();
      chartSingle.update();
      document.getElementById('title-single').textContent = `2. 單標籤角色分布 (${s.label_counts['1']} 張)`;
      break;
    case 'double':
      chartDouble.data.labels = s.double_data.map(d => d.label);
      chartDouble.data.datasets[0].data = s.double_data.map(d => d.count);
      chartDouble.update();
      document.getElementById('title-double').textContent = `3. 雙標籤組合分布 (${s.label_counts['2']} 張)`;
      break;
    case 'triple':
      chartTriple.data.labels = s.triple_data.map(d => d.label);
      chartTriple.data.datasets[0].data = s.triple_data.map(d => d.count);
      chartTriple.update();
      document.getElementById('title-triple').textContent = `4. 三標籤組合分布 (${s.label_counts['3']} 張)`;
      break;
    case 'overall':
      chartOverall.data.datasets[0].data = s.overall.slice();
      chartOverall.update();
      break;
  }
}

// ── 6. Heatmap rendering ──
function renderHeatmap() {
  const s = D[heatmapSrc];
  const container = document.getElementById('heatmap-container');
  let html = '<table class="heatmap"><tr><th></th>';
  SHORT.forEach(n => html += `<th>${n}</th>`);
  html += '</tr>';
  for (let i = 0; i < 6; i++) {
    html += `<tr><td class="row-header">${SHORT[i]}</td>`;
    for (let j = 0; j < 6; j++) {
      if (heatmapMode === 'count') {
        const v = s.cooc[i][j];
        const alpha = s.cooc_max > 0 ? v / s.cooc_max : 0;
        html += `<td style="background:rgba(52,152,219,${alpha.toFixed(3)})">${v}</td>`;
      } else {
        if (i === j) {
          html += '<td style="background:rgba(52,152,219,0.15)">—</td>';
        } else {
          const v = s.cond_rates[i][j];
          const alpha = s.cond_max > 0 ? v / s.cond_max : 0;
          html += `<td style="background:rgba(52,152,219,${alpha.toFixed(3)})">${v}%</td>`;
        }
      }
    }
    html += '</tr>';
  }
  html += '</table>';
  container.innerHTML = html;
}
renderHeatmap();

function toggleHeatmap(showAll) {
  heatmapSrc = showAll ? 'all' : 'orig';
  updateToggleLabels(event.target, showAll);
  renderHeatmap();
}

function showHeatmapMode(mode, btn) {
  heatmapMode = mode;
  btn.parentElement.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderHeatmap();
}
</script>

</body>
</html>"""


@app.route("/")  # type: ignore[untyped-decorator]
def index() -> str:
    data = _load_all_stats()
    result: str = render_template_string(HTML_TEMPLATE, data=data)
    return result


def main() -> None:
    webbrowser.open("http://localhost:5000")
    app.run(debug=False, port=5000)


if __name__ == "__main__":
    main()
