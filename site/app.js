// Demo site: loads the CSVs from data/ and draws charts with Observable Plot.
// Edit freely: each chart is one function below.

const TYPES = ["RECOVERY", "STEADY", "INTENSIVE"];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const typeColors = () => TYPES.map((_, i) => css(`--series-${i + 1}`));
const fmt = d3.format(",");
const fmt1 = d3.format(",.1f");

const state = { workouts: [], users: [], series: new Map(), active: new Set(TYPES), workoutId: null };

// ---------- load ----------
// Parse numbers automatically, but keep "YYYY-MM-DD" date columns as plain strings.
const parseRow = (d) => {
  const row = d3.autoType({ ...d });
  for (const k of Object.keys(d)) if (k.endsWith("_date")) row[k] = d[k];
  return row;
};

Promise.all([
  d3.csv("data/workouts.csv", parseRow),
  d3.csv("data/users.csv", parseRow),
  d3.csv("data/timeseries.csv", d3.autoType),
]).then(([workouts, users, ts]) => {
  state.workouts = workouts;
  state.users = users;
  state.series = d3.group(ts, (d) => d.workout_id);
  state.byId = new Map(workouts.map((w) => [w.workout_id, w]));
  // Fixed random sample for the scatter, so it doesn't jump when filtering.
  state.scatterSample = d3.shuffler(d3.randomLcg(42))(workouts.slice()).slice(0, 2500);

  buildTypeFilter();
  buildWorkoutPicker();
  renderUsersTable();
  renderAll();
}).catch((err) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin",
    `<p class="card">Could not load data (${err.message}). If you opened index.html directly, ` +
    `serve the folder instead: <code>python -m http.server -d site</code></p>`);
});

// ---------- controls ----------
function buildTypeFilter() {
  const box = document.getElementById("type-filter");
  TYPES.forEach((t, i) => {
    const label = document.createElement("label");
    label.className = "chip";
    label.innerHTML = `<input type="checkbox" checked value="${t}">
      <span class="swatch" style="background:var(--series-${i + 1})"></span>${t.toLowerCase()}`;
    label.querySelector("input").addEventListener("change", (e) => {
      e.target.checked ? state.active.add(t) : state.active.delete(t);
      renderAll();
    });
    box.append(label);
  });
}

function buildWorkoutPicker() {
  const select = document.getElementById("workout-select");
  const ids = [...state.series.keys()]
    .map((id) => state.byId.get(id))
    .sort((a, b) => d3.ascending(TYPES.indexOf(a.workout_type), TYPES.indexOf(b.workout_type)) || d3.ascending(a.start_date, b.start_date));
  for (const w of ids) {
    const opt = new Option(`${w.workout_type.toLowerCase()} · ${w.start_date} · ${fmt1(w.distance_km)} km`, w.workout_id);
    select.add(opt);
  }
  state.workoutId = ids[0].workout_id;
  select.addEventListener("change", () => { state.workoutId = +select.value; renderWorkout(); });
  document.getElementById("random-workout").addEventListener("click", () => {
    select.selectedIndex = Math.floor(Math.random() * select.options.length);
    select.dispatchEvent(new Event("change"));
  });
}

// ---------- render ----------
function renderAll() {
  const rows = state.workouts.filter((d) => state.active.has(d.workout_type));
  renderKpis(rows);
  renderLegend("legend-hist");
  renderLegend("legend-scatter");
  renderHistogram(rows);
  renderScatter();
  renderTimeline(rows);
  renderWorkout();
}

function renderKpis(rows) {
  const kpis = [
    ["Workouts", fmt(rows.length)],
    ["Athletes", fmt(new Set(rows.map((d) => d.user_id)).size)],
    ["Distance run", `${fmt(Math.round(d3.sum(rows, (d) => d.distance_km)))} km`],
    ["Avg heart rate", rows.length ? `${Math.round(d3.mean(rows, (d) => d.hr_mean))} BPM` : "–"],
  ];
  document.getElementById("kpis").innerHTML = kpis
    .map(([label, value]) => `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div></div>`)
    .join("");
}

function renderLegend(id) {
  document.getElementById(id).innerHTML = TYPES.filter((t) => state.active.has(t))
    .map((t) => `<span><span class="swatch" style="background:var(--series-${TYPES.indexOf(t) + 1})"></span>${t.toLowerCase()}</span>`)
    .join("");
}

const colorScale = () => ({ domain: TYPES, range: typeColors() });
const tipStyle = () => ({ fill: css("--surface"), stroke: css("--axis") });
const widthOf = (id) => document.getElementById(id).clientWidth;
const mount = (id, chart) => document.getElementById(id).replaceChildren(chart);

function renderHistogram(rows) {
  const bin = d3.bin().domain([120, 192]).thresholds(d3.range(120, 192, 2)).value((d) => d.hr_mean);
  const data = TYPES.filter((t) => state.active.has(t)).flatMap((t) => {
    const ofType = rows.filter((d) => d.workout_type === t);
    return bin(ofType).map((b) => ({ type: t, hr: (b.x0 + b.x1) / 2, share: b.length / ofType.length, n: b.length }));
  });
  mount("chart-hist", Plot.plot({
    width: widthOf("chart-hist"), height: 300, marginLeft: 44, marginBottom: 44,
    x: { label: "Average heart rate (BPM)", labelOffset: 40 },
    y: { label: "Share of workouts", tickFormat: "%", grid: true },
    color: colorScale(),
    marks: [
      Plot.ruleY([0], { stroke: css("--axis") }),
      Plot.lineY(data, {
        x: "hr", y: "share", stroke: "type", strokeWidth: 2, curve: "monotone-x",
        tip: { ...tipStyle(), format: { stroke: true, x: (v) => `${v - 1}–${v + 1}`, y: ".1%" } },
        channels: { Workouts: "n" },
      }),
    ],
  }));
}

function renderScatter() {
  // Draw the smallest group last so it sits on top.
  const data = state.scatterSample
    .filter((d) => state.active.has(d.workout_type))
    .sort((a, b) => d3.ascending(TYPES.indexOf(a.workout_type), TYPES.indexOf(b.workout_type)));
  mount("chart-scatter", Plot.plot({
    width: widthOf("chart-scatter"), height: 360, marginLeft: 44, marginBottom: 44,
    x: { label: "Average speed (km/h)", labelOffset: 40, domain: [4, 20], clamp: true },
    y: { label: "Average heart rate (BPM)", grid: true },
    color: colorScale(),
    marks: [
      Plot.dot(data, {
        x: "speed_mean", y: "hr_mean", fill: "workout_type", r: 3, fillOpacity: 0.7,
        stroke: css("--surface"), strokeWidth: 0.5,
        tip: { ...tipStyle(), format: { fill: true, x: ".1f", y: ".0f" } },
        channels: { Date: "start_date", "Distance (km)": "distance_km" },
      }),
    ],
  }));
}

function renderTimeline(rows) {
  const data = rows.filter((d) => d.start_date >= "2011-01-01" && d.start_date < "2016-01-01")
    .map((d) => ({ date: new Date(d.start_date) }));
  mount("chart-time", Plot.plot({
    width: widthOf("chart-time"), height: 240, marginLeft: 44,
    x: { label: null },
    y: { label: "Workouts", grid: true },
    marks: [
      Plot.rectY(data, Plot.binX({ y: "count" }, {
        x: "date", interval: "month", fill: css("--series-1"), insetLeft: 1, insetRight: 1,
        tip: { ...tipStyle(), format: { x: "%B %Y" } },
      })),
      Plot.ruleY([0], { stroke: css("--axis") }),
    ],
  }));
}

function renderWorkout() {
  const w = state.byId.get(state.workoutId);
  const ts = state.series.get(state.workoutId) ?? [];
  document.getElementById("workout-meta").textContent =
    `${w.workout_type.toLowerCase()} run on ${w.start_date} · ${fmt1(w.distance_km)} km in ${Math.round(w.duration_min)} min · ` +
    `avg ${Math.round(w.hr_mean)} BPM, max ${Math.round(w.hr_max)} BPM`;

  // One small chart per measure, sharing the time axis (never two y-scales on one chart).
  const color = css(`--series-${TYPES.indexOf(w.workout_type) + 1}`);
  const xDomain = d3.extent(ts, (d) => d.t_min);
  const panels = [
    ["chart-hr", "heart_rate", "Heart rate (BPM)"],
    ["chart-speed", "speed", "Speed (km/h)"],
    ["chart-alt", "altitude", "Altitude (m)"],
  ];
  panels.forEach(([id, key, label], i) => {
    const last = i === panels.length - 1;
    mount(id, Plot.plot({
      width: widthOf(id), height: last ? 150 : 130, marginLeft: 44, marginBottom: last ? 36 : 12,
      x: { domain: xDomain, label: last ? "Minutes since start" : null, axis: last ? "bottom" : null },
      y: { label, grid: true, nice: true, zero: false },
      marks: [
        Plot.lineY(ts, { x: "t_min", y: key, stroke: color, strokeWidth: 2 }),
        Plot.ruleX(ts, Plot.pointerX({ x: "t_min", stroke: css("--axis") })),
        Plot.dot(ts, Plot.pointerX({ x: "t_min", y: key, fill: color, stroke: css("--surface"), strokeWidth: 2, r: 4 })),
        Plot.tip(ts, Plot.pointerX({ x: "t_min", y: key, ...tipStyle(), format: { x: ".1f", y: ".1f" } })),
      ],
    }));
  });
}

function renderUsersTable() {
  const cols = [
    ["Athlete", (d) => d.user_id],
    ["Workouts", (d) => fmt(d.workouts)],
    ["Total km", (d) => fmt(Math.round(d.total_km))],
    ["Hours", (d) => fmt(Math.round(d.total_hours))],
    ["Avg HR", (d) => Math.round(d.hr_mean)],
    ["Avg km/h", (d) => fmt1(d.speed_mean)],
    ["Active", (d) => `${d.first_date.slice(0, 4)}–${d.last_date.slice(0, 4)}`],
  ];
  const top = state.users.slice(0, 10);
  document.getElementById("users-table").innerHTML =
    `<thead><tr>${cols.map(([h]) => `<th>${h}</th>`).join("")}</tr></thead>` +
    `<tbody>${top.map((d) => `<tr>${cols.map(([, f]) => `<td>${f(d)}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

// Re-render on resize and when the OS switches light/dark.
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => state.workouts.length && renderAll(), 150); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.workouts.length && renderAll());
