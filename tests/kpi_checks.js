// KPI checks for the running dashboard. Run from the repo root:  node tests/kpi_checks.js
// Part A: pure KPI maths of site/summary.js against values derived independently from REAL CSV rows
//         of runner 897592 (13 runs, 2013-12-22 .. 2014-05-06, one run in timeseries.csv).
// Part B: the real inline script of site/index.html run in Node with a DOM/localStorage/fetch stub.
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const ROOT = path.resolve(__dirname, "..");
const RS = require(path.join(ROOT, "site", "summary.js"));

let passed = 0, failed = 0;
function check(name, fn) {
  try { fn(); passed++; console.log("ok   - " + name); }
  catch (e) { failed++; console.log("FAIL - " + name + "\n       " + e.message); }
}
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, (msg || "") + " expected " + b + " got " + a);

// ---------- real data ----------
const wText = fs.readFileSync(path.join(ROOT, "site/data/workouts.csv"), "utf8");
const tText = fs.readFileSync(path.join(ROOT, "site/data/timeseries.csv"), "utf8");
const rows = RS.parseCsv(wText);
const tsMap = RS.parseTimeseries(tText);
const UID = "897592";
const ctx = RS.buildKpiContext(rows, UID, tsMap);
const ctxNoTs = RS.buildKpiContext(rows, UID, null);

// Literal copy of the 13 real rows (id, type, date, min, hr_mean, hr_std, hr_min, hr_max, speed_mean, ascent, km)
const REAL = [
  [280025879, "RECOVERY", "2013-12-22", 77.57, 143.7, 8.33, 82.0, 152.71, 10.71, 86.55, 13.84],
  [281031981, "RECOVERY", "2013-12-30", 91.65, 143.79, 7.37, 90.25, 166.71, 10.34, 108.0, 15.79],
  [284886858, "STEADY", "2014-01-07", 59.83, 149.71, 9.03, 98.25, 166.86, 11.02, 108.65, 10.99],
  [288116327, "RECOVERY", "2014-01-19", 112.08, 143.85, 6.92, 95.5, 172.86, 10.77, 155.49, 20.13],
  [321979692, "RECOVERY", "2014-01-26", 114.0, 144.28, 6.93, 102.25, 155.25, 10.28, 123.17, 19.53],
  [321979610, "STEADY", "2014-02-09", 137.82, 148.18, 5.13, 110.75, 169.14, 9.38, 116.4, 21.54],
  [321979519, "RECOVERY", "2014-02-27", 73.17, 144.74, 17.29, 105.75, 171.14, 8.45, 107.94, 10.3],
  [314498134, "STEADY", "2014-03-19", 65.73, 153.85, 10.39, 83.75, 169.43, 11.33, 103.46, 12.41],
  [321977351, "RECOVERY", "2014-04-04", 90.05, 142.29, 12.3, 79.5, 161.14, 10.73, 153.15, 16.11],
  [326064710, "RECOVERY", "2014-04-14", 63.97, 138.24, 5.71, 97.5, 147.0, 10.75, 91.75, 11.46],
  [326064701, "RECOVERY", "2014-04-16", 123.95, 142.82, 8.08, 97.0, 154.0, 10.53, 130.19, 21.74],
  [331689943, "RECOVERY", "2014-04-29", 86.25, 135.67, 8.66, 88.5, 151.86, 10.15, 137.46, 14.6],
  [340870700, "RECOVERY", "2014-05-06", 113.83, 135.58, 5.88, 93.5, 145.75, 10.29, 97.48, 19.53]
].map(a => ({ id: String(a[0]), type: a[1], date: a[2], min: a[3], hr: a[4], sd: a[5], hmin: a[6], hmax: a[7], spd: a[8], up: a[9], km: a[10] }));

// ---------- independent helpers (written separately from summary.js) ----------
function isoLabel(dateStr) {
  const p = dateStr.split("-").map(Number);
  const d = new Date(Date.UTC(p[0], p[1] - 1, p[2]));
  const dayNum = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const y = d.getUTCFullYear();
  const wk = Math.ceil(((d - Date.UTC(y, 0, 1)) / 864e5 + 1) / 7);
  return y + "-W" + (wk < 10 ? "0" : "") + wk;
}
function phi(x) { // standard normal CDF by Simpson integration of the pdf from -10
  const n = 4000, a = -10, h = (x - a) / n;
  const f = t => Math.exp(-t * t / 2) / Math.sqrt(2 * Math.PI);
  let s = f(a) + f(x);
  for (let i = 1; i < n; i++) s += f(a + i * h) * (i % 2 ? 4 : 2);
  return s * h / 3;
}
function estZones(r, ref) { // truncated normal on [hmin, hmax]
  const F = x => phi((Math.min(Math.max(x, r.hmin), r.hmax) - r.hr) / r.sd);
  const tot = F(r.hmax) - F(r.hmin), c1 = 0.8 * ref, c2 = 0.9 * ref;
  return { low: (F(c1) - F(r.hmin)) / tot, mod: (F(c2) - F(c1)) / tot, high: (F(r.hmax) - F(c2)) / tot };
}
function seriesFor(id) {
  return tText.trim().split(/\r?\n/).slice(1).map(l => l.split(",")).filter(c => c[0] === id)
    .map(c => [+c[1], +c[2], +c[3], +c[4]]).sort((a, b) => a[0] - b[0]);
}
function rawSeriesZones(s, ref) {
  const z = [0, 0, 0]; let tot = 0;
  for (let i = 0; i < s.length - 1; i++) {
    const w = s[i + 1][0] - s[i][0];
    z[s[i][1] < 0.8 * ref ? 0 : s[i][1] < 0.9 * ref ? 1 : 2] += w; tot += w;
  }
  return { low: z[0] / tot, mod: z[1] / tot, high: z[2] / tot };
}
function corr(x, y) {
  const n = x.length, mx = x.reduce((a, b) => a + b) / n, my = y.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
}
function ols(x, y) {
  const n = x.length, mx = x.reduce((a, b) => a + b) / n, my = y.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  return { b: sxy / sxx, a: my - (sxy / sxx) * mx };
}

const REF = 172.86; // max hr_max of this runner (> 170 floor)
const SERIES_ID = "321977351";
const seriesRow = REAL.find(r => r.id === SERIES_ID);
const rawZones = {}; // per run, independent
REAL.forEach(r => { rawZones[r.id] = r.id === SERIES_ID ? rawSeriesZones(seriesFor(r.id), REF) : estZones(r, REF); });
const loadOf = r => r.min * (rawZones[r.id].low + 2 * rawZones[r.id].mod + 3 * rawZones[r.id].high);

// ============================ Part A ============================
check("context: 13 real runs for runner 897592, matches the literal CSV rows", () => {
  assert.strictEqual(ctx.runs.length, 13);
  ctx.runs.forEach((run, i) => {
    const r = REAL[i];
    assert.strictEqual(run.id, r.id); assert.strictEqual(run.date, r.date);
    near(run.minutes, r.min, 1e-9); near(run.hrMean, r.hr, 1e-9); near(run.speed, r.spd, 1e-9); near(run.km, r.km, 1e-9);
  });
  near(ctx.ref, REF, 1e-9);
});

check("ISO week labels: module == independent ISO algorithm for all 13 dates", () => {
  REAL.forEach(r => assert.strictEqual(RS.weekLabel(RS.weekIndex(RS.parseDay(r.date))), isoLabel(r.date), r.date));
  assert.strictEqual(isoLabel("2013-12-30"), "2014-W01"); // year-boundary sanity of the independent helper
});

check("KPI1 weekly consistency: sessions/minutes/streaks", () => {
  const c = RS.consistencyKpi(ctx.runs);
  assert.strictEqual(c.sessions, 13);
  near(c.minutes, 77.57 + 91.65 + 59.83 + 112.08 + 114.0 + 137.82 + 73.17 + 65.73 + 90.05 + 63.97 + 123.95 + 86.25 + 113.83, 1e-6);
  assert.strictEqual(c.activeWeeks, 12);   // W51(2013), W01-04, W06, W09, W12, W14, W16, W18, W19 (W16 has 2 runs)
  assert.strictEqual(c.spanWeeks, 21);     // 2013-W51 .. 2014-W19
  assert.strictEqual(c.longest, 4);        // 2014-W01..W04
  assert.strictEqual(c.current, 2);        // W18, W19 (gap at W17)
  assert.deepStrictEqual(c.weeks.map(w => w.label),
    ["2014-W12", "2014-W13", "2014-W14", "2014-W15", "2014-W16", "2014-W17", "2014-W18", "2014-W19"]);
  assert.deepStrictEqual(c.weeks.map(w => w.sessions), [1, 0, 1, 0, 2, 0, 1, 1]);
  near(c.weeks[4].minutes, 63.97 + 123.95, 1e-6);
});

check("KPI2 simple session summaries: line + totals from distance/duration/speed", () => {
  const last = ctx.runs[12];
  assert.strictEqual(RS.buildSessionLine(RS.sessionTriples(last)),
    "On 2014-05-06 you ran 19.5 km in 114 minutes. Your average pace was 5:50 minutes per kilometre.");
  assert.strictEqual(RS.fmtPace(60 / 10.29), "5:50");
  const card = RS.kpiCards(ctx, "Beginner").find(c => c.id === "simple_session_summaries");
  assert.strictEqual(card.lines.length, 6);
  assert.strictEqual(card.lines[5], "Together these 5 runs: 83.4 km in 478 minutes, an average pace of 5:44 minutes per kilometre.");
});

check("KPI3 zones: estimated run == independent truncated-normal integration", () => {
  const r = REAL.find(x => x.id === "326064710"); // hr_mean 138.24, hr_max 147 -> no high zone
  const m = ctx.metrics.find(x => x.id === r.id), e = rawZones[r.id];
  assert.strictEqual(m.zoneSource, "SummaryEstimatedZones");
  near(m.zones.low, e.low, 0.003, "low"); near(m.zones.mod, e.mod, 0.003, "mod"); near(m.zones.high, e.high, 0.003, "high");
  near(m.zones.mod, 0.462, 0.005); near(m.zones.high, 0, 1e-6);
  near(m.zones.low + m.zones.mod + m.zones.high, 1, 1e-9);
});

check("KPI3 zones: time-series run uses the series (time-weighted) and is labelled so", () => {
  const m = ctx.metrics.find(x => x.id === SERIES_ID), e = rawZones[SERIES_ID];
  assert.strictEqual(m.zoneSource, "TimeSeriesZones");
  near(m.zones.low, e.low, 1e-9); near(m.zones.mod, e.mod, 1e-9); near(m.zones.high, e.high, 1e-9);
  const noTs = ctxNoTs.metrics.find(x => x.id === SERIES_ID);
  assert.strictEqual(noTs.zoneSource, "SummaryEstimatedZones"); // same run falls back when no series is loaded
});

check("KPI3 TID: last-10-run aggregate + per-source split", () => {
  const t = RS.tidKpi(ctx.metrics);
  assert.strictEqual(t.all.runs, 10); assert.strictEqual(t.from, "2014-01-19"); assert.strictEqual(t.to, "2014-05-06");
  assert.strictEqual(t.series.runs, 1); assert.strictEqual(t.estimated.runs, 9);
  const last10 = REAL.slice(3);
  const tot = last10.reduce((a, r) => a + r.min, 0);
  ["low", "mod", "high"].forEach(z => near(t.all[z], last10.reduce((a, r) => a + r.min * rawZones[r.id][z], 0) / tot, 0.003, z));
  near(t.series.minutes, seriesRow.min, 1e-9);
  near(t.series.low, rawZones[SERIES_ID].low, 1e-9);
  const est = last10.filter(r => r.id !== SERIES_ID), et = est.reduce((a, r) => a + r.min, 0);
  near(t.estimated.low, est.reduce((a, r) => a + r.min * rawZones[r.id].low, 0) / et, 0.003);
  const card = RS.kpiCards(ctx, "Expert").find(c => c.id === "training_intensity_distribution");
  assert.ok(/Source time-series: 1 run/.test(card.lines[1]) && /Source summary estimate/.test(card.lines[2]));
  const cardNo = RS.kpiCards(ctxNoTs, "Expert").find(c => c.id === "training_intensity_distribution");
  assert.ok(/time-series file not loaded/.test(cardNo.lines[1]));
});

check("KPI4 coupling: beats/km, ascent/km, correlations, within-run r", () => {
  const k = RS.couplingKpi(ctx.metrics), last = ctx.metrics[12];
  near(last.beatsPerKm, 135.58 * 60 / 10.29, 1e-9); near(last.beatsPerKm, 790.55, 0.01);
  near(last.ascentPerKm, 97.48 / 19.53, 1e-9);
  near(k.rHrSpeed, corr(REAL.map(r => r.hr), REAL.map(r => r.spd)), 1e-9);
  const bpk = REAL.map(r => r.hr * 60 / r.spd), apk = REAL.map(r => r.up / r.km);
  near(k.rCostAscent, corr(apk, bpk), 1e-9);
  assert.strictEqual(k.recent.length, 5);
  const s = seriesFor(SERIES_ID), m = ctx.metrics.find(x => x.id === SERIES_ID);
  near(m.seriesR, corr(s.map(p => p[1]), s.map(p => p[2])), 1e-9);
  // the series mean HR/speed agree with the summary columns for this run (basis for using summary means)
  near(s.reduce((a, p) => a + p[1], 0) / s.length, seriesRow.hr, 0.05);
  near(s.reduce((a, p) => a + p[2], 0) / s.length, seriesRow.spd, 0.05);
});

check("KPI5 periodized: weekly volume, acute/chronic per ISO week, trend", () => {
  const p = RS.periodizedKpi(ctx.metrics);
  assert.deepStrictEqual(p.weeks.map(w => +w.km.toFixed(2)), [12.41, 0, 16.11, 0, 33.2, 0, 14.6, 19.53]);
  const L = {}; // independent weekly loads
  REAL.forEach(r => { const k = isoLabel(r.date); L[k] = (L[k] || 0) + loadOf(r); });
  const lab = p.weeks.map(w => w.label), ld = l => L[l] || 0;
  const w19 = p.weeks[7], w18 = "2014-W18", w17 = "2014-W17", w16 = "2014-W16";
  near(w19.acute, ld("2014-W19"), 0.5);
  near(w19.chronic, (ld("2014-W19") + ld(w18) + ld(w17) + ld(w16)) / 4, 0.5);
  near(w19.ratio, w19.acute / w19.chronic, 1e-9);
  assert.strictEqual(w19.status, w19.ratio < 0.8 ? "low" : w19.ratio > 1.3 ? "high" : "balanced");
  const w16row = p.weeks[4];
  near(w16row.acute, ld(w16), 0.5);
  near(w16row.intensity, ld(w16) / (63.97 + 123.95), 0.003);
  near(p.kmPerWeekTrend, ols([0, 1, 2, 3, 4, 5, 6, 7], [12.41, 0, 16.11, 0, 33.2, 0, 14.6, 19.53]).b, 1e-9);
  assert.strictEqual(lab[0], "2014-W12");
});

check("KPI6 strain: expected HR from earlier-run line, residuals, flags, drift", () => {
  const s = RS.strainKpi(ctx.metrics);
  assert.strictEqual(s.rows.length, 6); assert.strictEqual(s.baselineRuns, 7); assert.strictEqual(s.model, "line");
  const base = REAL.slice(0, 7), fit = ols(base.map(r => r.spd), base.map(r => r.hr));
  REAL.slice(7).forEach((r, i) => {
    const e = fit.a + fit.b * r.spd;
    near(s.rows[i].expected, e, 1e-9); near(s.rows[i].resid, r.hr - e, 1e-9);
    assert.strictEqual(s.rows[i].flag, r.hr - e >= 5);
  });
  const row = s.rows.find(r => r.m.id === SERIES_ID);
  const ser = seriesFor(SERIES_ID), mid = (ser[0][0] + ser[ser.length - 1][0]) / 2;
  const half = f => { let h = 0, v = 0, w = 0; for (let i = 0; i < ser.length - 1; i++) if (f(ser[i][0])) { const d = ser[i + 1][0] - ser[i][0]; h += ser[i][1] * d; v += ser[i][2] * d; w += d; } return h / v; };
  near(row.m.drift, (half(t => t >= mid) / half(t => t < mid) - 1) * 100, 1e-6);
  assert.strictEqual(s.rows.filter(r => r.m.drift !== null).length, 1); // only the series run has drift
});

check("Triples: new predicates/individuals all exist in docs/ontology/version1.ttl", () => {
  const ttl = fs.readFileSync(path.join(ROOT, "docs/ontology/version1.ttl"), "utf8");
  const preds = new Set(), objs = new Set();
  ctx.metrics.forEach(m => RS.metricTriples(m).forEach(t => { preds.add(t.p); if (typeof t.o === "string" && t.o.indexOf("endo:") === 0 && !/^endo:(Athlete|Workout)_/.test(t.o)) objs.add(t.o); }));
  ["startDate", "paceMinPerKm", "lowZoneShare", "moderateZoneShare", "highZoneShare", "zoneSource", "trainingLoad", "beatsPerKm", "hrDriftPct", "distanceKm"]
    .forEach(p => { assert.ok(preds.has(p), p + " emitted"); });
  preds.forEach(p => { if (p !== "rdf:type") assert.ok(new RegExp("^endo:" + p + " a ", "m").test(ttl), "ontology has endo:" + p); });
  objs.forEach(o => assert.ok(new RegExp("^" + o + " a ", "m").test(ttl), "ontology has " + o));
  assert.ok(/^endo:ZoneSource a owl:Class/m.test(ttl));
});

check("Wording: Beginner plain vs Expert compact; same runner, different text", () => {
  const b = RS.kpiCards(ctx, "Beginner"), e = RS.kpiCards(ctx, "Expert");
  assert.deepStrictEqual(b.map(c => c.id), ["weekly_consistency_and_persistence", "simple_session_summaries", "not_available_beginner"]);
  assert.deepStrictEqual(e.map(c => c.id), ["training_intensity_distribution", "internal_vs_external_load_coupling",
    "periodized_load_progression", "physiological_strain_and_fatigue_indicators", "not_available_expert"]);
  assert.ok(b[0].lines[0].startsWith("You ran in 12 of the 21 weeks"));
  assert.ok(/^TID /.test(e[0].lines[0]));
  assert.ok(!JSON.stringify(b).includes("beats/km") && !JSON.stringify(b).includes("acute"));
});

check("Visual data: stats/zones/bars match the KPI values", () => {
  const b = RS.kpiCards(ctx, "Beginner"), e = RS.kpiCards(ctx, "Expert");
  assert.deepStrictEqual(b[0].stats.map(s => s.value), ["12/21", "4", "2"]);
  assert.deepStrictEqual(b[0].bars.items.map(i => i.cls), ["good", "mute", "good", "mute", "good", "mute", "good", "good"]);
  assert.strictEqual(b[0].bars.items[4].text, "188 min (2 runs)");
  assert.deepStrictEqual(b[1].stats.map(s => s.value), ["19.5 km", "5:50", "83.4 km"]);
  const z = e[0].zones;
  assert.deepStrictEqual(z.map(r => r.label), ["All 10 runs", "Time-series (1 run)", "Summary estimate (9 runs)"]);
  z.forEach(r => near(r.low + r.mod + r.high, 1, 1e-9));
  near(z[1].low, rawZones[SERIES_ID].low, 1e-9);
  assert.strictEqual(e[1].bars.items.length, 5);
  near(e[1].bars.items[4].value, 135.58 * 60 / 10.29, 1e-9); // bars are chronological: latest run last
  const p = e[2].bars.items;
  assert.deepStrictEqual(p.map(i => i.label), ["W12", "W13", "W14", "W15", "W16", "W17", "W18", "W19"]);
  assert.strictEqual(p[7].cls, "good");  // W19 ratio 1.04 balanced
  assert.strictEqual(p[4].cls, "bad");   // W16 ratio 2.59 high
  assert.strictEqual(p[5].cls, "warn");  // W17 rest, ratio 0 low
  assert.deepStrictEqual(e[3].bars.items.map(i => i.cls).filter(c => c === "bad").length, 1);
  assert.ok(e.slice(0, 4).every(c => c.collapse === 1 && c.stats.length >= 2));
});

check("Chart 4 data: weekly zone shares (minute-weighted) + zone source per ISO week", () => {
  const wz = RS.weeklyZonesKpi(ctx.metrics);
  assert.deepStrictEqual(wz.weeks.map(w => w.label.slice(5)), ["W12", "W13", "W14", "W15", "W16", "W17", "W18", "W19"]);
  assert.deepStrictEqual(wz.weeks.map(w => w.runs), [1, 0, 1, 0, 2, 0, 1, 1]);
  assert.deepStrictEqual(wz.weeks.map(w => w.source), ["estimate", "rest", "time-series", "rest", "estimate", "rest", "estimate", "estimate"]);
  const inW = l => REAL.filter(r => isoLabel(r.date) === l);
  wz.weeks.filter(w => w.runs).forEach(w => {
    const rs = inW(w.label), tot = rs.reduce((a, r) => a + r.min, 0);
    ["low", "mod", "high"].forEach(z => near(w[z], rs.reduce((a, r) => a + r.min * rawZones[r.id][z], 0) / tot, 0.003, w.label + " " + z));
    near(w.low + w.mod + w.high, 1, 1e-9);
  });
  near(wz.target, 0.8, 1e-12);
  const card = RS.kpiCards(ctx, "Expert").find(c => c.id === "training_intensity_distribution");
  assert.ok(card.wide && card.weekZones.weeks.length === 8);
});

check("Chart 6 data: every run a point; latest 6 judged against the earlier-run line", () => {
  const s = RS.strainKpi(ctx.metrics), sc = RS.strainScatter(ctx.metrics, s);
  assert.strictEqual(sc.points.length, 13);
  assert.strictEqual(sc.points.filter(p => p.group === "latest").length, 6);
  assert.strictEqual(sc.points.filter(p => p.flag).length, s.flagged);
  const fit = ols(REAL.slice(0, 7).map(r => r.spd), REAL.slice(0, 7).map(r => r.hr));
  sc.points.forEach((p, i) => { near(p.expected, fit.a + fit.b * REAL[i].spd, 1e-9); near(p.resid, REAL[i].hr - p.expected, 1e-9); });
  assert.deepStrictEqual(sc.points.filter(p => p.flag).map(p => p.date), ["2014-03-19"]);
});

check("Reasoning popups: every card has text + sources; every source key is in site/sources.bib once", () => {
  const bib = fs.readFileSync(path.join(ROOT, "site/sources.bib"), "utf8");
  const keys = [...bib.matchAll(/^@\w+\s*\{\s*([^,\s]+)\s*,/gm)].map(m => m[1]);
  assert.strictEqual(new Set(keys.map(k => k.toLowerCase())).size, keys.length, "no duplicate bib keys (case-insensitive)");
  Object.keys(RS.SOURCES).forEach(k => assert.ok(keys.includes(k), "bib has " + k));
  ["Beginner", "Expert"].forEach(mode => RS.kpiCards(ctx, mode).forEach(c => {
    assert.ok(c.reason && c.reason.why.length >= 1 && c.reason.sources.length >= 1, c.id + " has reasoning");
    c.reason.sources.forEach(k => assert.ok(RS.SOURCES[k], c.id + " source " + k + " is known"));
  }));
  let depth = 0; for (const ch of bib.replace(/^%.*$/gm, "")) { if (ch === "{") depth++; else if (ch === "}") depth--; assert.ok(depth >= 0); }
  assert.strictEqual(depth, 0, "balanced braces in sources.bib");
});

// ============================ Part B: dashboard script with DOM stub ============================
class El {
  constructor(tag, attrs) {
    this.tag = tag; this.attrs = attrs || {}; this.children = []; this.listeners = {};
    this.dataset = {}; this.style = {}; this.title = ""; this._text = ""; this.value = ""; this.className = this.attrs.class || "";
    this.hidden = "hidden" in this.attrs;
    this._cls = new Set((this.attrs.class || "").split(/\s+/).filter(Boolean));
    Object.keys(this.attrs).forEach(k => { if (k.indexOf("data-") === 0) this.dataset[k.slice(5).replace(/-(.)/g, (_, c) => c.toUpperCase())] = this.attrs[k]; });
    const self = this;
    this.classList = {
      add: c => self._cls.add(c), remove: c => self._cls.delete(c), contains: c => self._cls.has(c),
      toggle: (c, on) => { if (on === undefined ? !self._cls.has(c) : on) self._cls.add(c); else self._cls.delete(c); }
    };
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(""); }
  appendChild(c) { this.children.push(c); return c; }
  prepend(c) { this.children.unshift(c); }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  focus() {}
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "class") { this.className = String(v); this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); } }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  closest(sel) {
    if (sel === "button[data-mode]") return this.tag === "button" && this.dataset.mode ? this : null;
    if (sel === "a[data-section]") return this.tag === "a" && this.dataset.section ? this : null;
    return null;
  }
  click() { (this.listeners.click || []).forEach(f => f({ target: this, preventDefault() {} })); }
}
function makeDom(html) {
  const byId = {}, all = [];
  const re = /<(\w+)((?:\s+[\w-]+(?:="[^"]*")?)*)\s*>/g;
  let m;
  while ((m = re.exec(html))) {
    const attrs = {};
    (m[2].match(/[\w-]+(?:="[^"]*")?/g) || []).forEach(a => { const i = a.indexOf("="); attrs[i < 0 ? a : a.slice(0, i)] = i < 0 ? "" : a.slice(i + 2, -1); });
    const el = new El(m[1].toLowerCase(), attrs); all.push(el);
    if (attrs.id) byId[attrs.id] = el;
  }
  const doc = {
    getElementById: id => byId[id],
    createElement: t => new El(t),
    createElementNS: (ns, t) => new El(t),
    createTextNode: s => { const e = new El("#text"); e._text = String(s); return e; },
    querySelectorAll: sel => {
      if (sel === ".role-card") return all.filter(e => e._cls.has("role-card"));
      if (sel === "#runner-mode-toggle button") return all.filter(e => e.tag === "button" && e.dataset.mode);
      if (sel === "#main-nav a") return all.filter(e => e.tag === "a" && e.dataset.section);
      return [];
    }
  };
  return { doc, byId, all };
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function settle(env, ms) { await sleep(ms || 400); }

function bootDashboard(store) {
  const html = fs.readFileSync(path.join(ROOT, "site/index.html"), "utf8");
  const script = html.split(/<script>/)[1].split(/<\/script>/)[0];
  const dom = makeDom(html);
  const ls = { _s: Object.assign({}, store || {}), getItem(k) { return k in this._s ? this._s[k] : null; }, setItem(k, v) { this._s[k] = String(v); }, removeItem(k) { delete this._s[k]; } };
  const fetchLog = [];
  const fetchStub = url => { fetchLog.push(url); const f = path.join(ROOT, "site", url); return fs.existsSync(f) ? Promise.resolve({ ok: true, text: () => Promise.resolve(fs.readFileSync(f, "utf8")) }) : Promise.resolve({ ok: false, status: 404 }); };
  new Function("document", "localStorage", "fetch", "RunSummary", script)(dom.doc, ls, fetchStub, RS);
  return { dom, ls, fetchLog };
}
// The real handler sits on the toggle container and reads e.target (event bubbling), so dispatch there.
function clickMode(env, mode) {
  const btn = env.dom.all.find(e => e.tag === "button" && e.dataset.mode === mode);
  env.dom.byId["runner-mode-toggle"].listeners.click.forEach(f => f({ target: btn }));
}
// All elements below a node (document order) with a given tag / class.
function findAll(node, pred, out) { out = out || []; node.children.forEach(c => { if (pred(c)) out.push(c); findAll(c, pred, out); }); return out; }
const liTexts = card => findAll(card, e => e.tag === "li").map(li => li.textContent);
const cardIds = env => env.dom.byId["kpi-cards"].children.map(c => c.dataset.kpi);

(async function partB() {
  let env = bootDashboard({ runtracker_session: JSON.stringify({ role: "Beginner runner" }) });
  await settle();
  check("dashboard Beginner runner: KPI panel visible, 2 KPI cards + 1 placeholder, no timeseries fetch", () => {
    assert.strictEqual(env.dom.byId["kpi-panel"].hidden, false);
    assert.deepStrictEqual(cardIds(env), ["weekly_consistency_and_persistence", "simple_session_summaries", "not_available_beginner"]);
    assert.ok(!env.fetchLog.some(u => /timeseries/.test(u)));
    const who = env.dom.byId["whoami"].textContent.match(/ID (\d+)/);
    assert.ok(who && env.dom.byId["kpi-meta"].textContent.indexOf("Runner " + who[1] + ":") === 0, "KPI runner == run-history runner");
    assert.ok(/real start dates/.test(env.dom.byId["kpi-meta"].textContent));
  });

  env.dom.byId["kpi-runner-input"].value = UID;
  env.dom.byId["kpi-runner-show"].click();
  await settle();
  check("runner box selects runner 897592: meta + real first/last dates", () => {
    assert.ok(env.dom.byId["kpi-meta"].textContent.indexOf("Runner 897592: 13 runs from 2013-12-22 to 2014-05-06") === 0);
    const w = liTexts(env.dom.byId["kpi-cards"].children[0]);
    assert.strictEqual(w[0], "You ran in 12 of the 21 weeks between 2013-12-22 and 2014-05-06.");
  });

  clickMode(env, "Expert");
  await settle(600);
  check("toggle to Expert re-renders: 4 expert KPI cards + placeholder, timeseries fetched, same runner", () => {
    assert.deepStrictEqual(cardIds(env), ["training_intensity_distribution", "internal_vs_external_load_coupling",
      "periodized_load_progression", "physiological_strain_and_fatigue_indicators", "not_available_expert"]);
    assert.ok(env.fetchLog.some(u => /timeseries/.test(u)));
    assert.ok(env.dom.byId["kpi-meta"].textContent.indexOf("Runner 897592") === 0);
    const tid = liTexts(env.dom.byId["kpi-cards"].children[0]);
    assert.ok(/Source time-series: 1 run/.test(tid[1]), tid[1]);
  });

  check("rendered Expert cards: stat chips, 3-segment zone bars, bar rows with widths, details block", () => {
    const cards = env.dom.byId["kpi-cards"].children, has = (n, c) => findAll(n, e => (e.className || "").split(" ").indexOf(c) >= 0);
    assert.strictEqual(has(cards[0], "kpi-stat").length, 3);
    const bars = has(cards[0], "zonebar");
    assert.strictEqual(bars.length, 3);
    bars.forEach(b => { assert.strictEqual(b.children.length, 3); near(b.children.reduce((a, s) => a + parseFloat(s.style.width), 0), 100, 1e-6); });
    assert.strictEqual(has(cards[1], "bar-row").length, 5);
    has(cards[2], "bar-fill").forEach(f => assert.ok(parseFloat(f.style.width) >= 0 && parseFloat(f.style.width) <= 100));
    assert.strictEqual(findAll(cards[3], e => e.tag === "details" && e.className !== "chart-table").length, 1); // + the chart's own "Table view"
    assert.strictEqual(has(cards[3], "badge").length >= 1, true); // run 2014-03-19 is flagged
    assert.strictEqual(findAll(cards[3], e => e.tag === "li").some(li => /FLAG/.test(li.textContent)), true);
  });

  check("rendered charts: weekly zone columns + 80 % line, scatter with one dot per run, tooltips, table views", () => {
    const cards = env.dom.byId["kpi-cards"].children, cls = (n, c) => findAll(n, e => (e.getAttribute && (e.getAttribute("class") || "").split(" ").indexOf(c) >= 0) || (e.className || "").split(" ").indexOf(c) >= 0);
    const tid = cards[0], strain = cards[3];
    assert.ok(/kpi-card--wide/.test(tid.className) && /kpi-card--wide/.test(strain.className));
    assert.strictEqual(findAll(tid, e => e.tag === "svg").length, 1);
    assert.strictEqual(cls(tid, "ref").length, 1, "one 80 % reference line");
    assert.strictEqual(cls(tid, "hit").length, 5, "one hover target per active week");
    assert.ok(cls(tid, "seg-low").length >= 5);
    assert.strictEqual(cls(strain, "dot").filter(e => e.tag === "circle").length, 13, "one dot per run");
    assert.strictEqual(cls(strain, "dot--flag").filter(e => e.tag === "circle").length, 1);
    assert.strictEqual(cls(strain, "trend").length, 1);
    const hit = cls(strain, "hit")[0];
    hit.listeners.focus[0]();
    const tip = cls(strain, "chart-tip")[0];
    assert.strictEqual(tip.hidden, false); assert.ok(/HR \d+ bpm at/.test(tip.textContent));
    hit.listeners.blur[0]();
    assert.strictEqual(tip.hidden, true);
    assert.strictEqual(findAll(tid, e => e.tag === "table").length, 1);
    assert.strictEqual(findAll(strain, e => e.tag === "tr").length, 14, "header + 13 rows");
  });

  check("'Why this?' button on every card opens the reasoning dialog with sources", () => {
    const cards = env.dom.byId["kpi-cards"].children;
    cards.forEach(c => assert.strictEqual(findAll(c, e => e.className === "btn-why").length, 1, c.dataset.kpi));
    findAll(cards[0], e => e.className === "btn-why")[0].click();
    const dlg = env.dom.byId["why-dialog"], body = env.dom.byId["why-body"];
    assert.strictEqual(dlg.getAttribute("open"), "");
    assert.ok(/Seiler \(2010\)/.test(body.textContent) && /Kenneally/.test(body.textContent));
    assert.ok(findAll(body, e => e.tag === "a").some(a => a.href === "sources.html#seiler2010"));
    env.dom.byId["why-close"].click();
    assert.strictEqual(dlg.getAttribute("open"), null);
  });

  clickMode(env, "Beginner");
  await settle();
  check("toggle back to Beginner restores beginner cards", () => {
    assert.strictEqual(cardIds(env)[0], "weekly_consistency_and_persistence");
  });

  env.dom.byId["kpi-runner-input"].value = "000";
  env.dom.byId["kpi-runner-show"].click();
  await settle();
  check("unknown runner id shows a clear message and no cards", () => {
    assert.strictEqual(env.dom.byId["kpi-meta"].textContent, "No usable runs found for runner 000.");
    assert.strictEqual(cardIds(env).length, 0);
  });

  const navTo = section => {
    const a = env.dom.all.find(e => e.tag === "a" && e.dataset.section === section);
    env.dom.byId["main-nav"].listeners.click.forEach(f => f({ target: a, preventDefault() {} }));
  };
  navTo("history");
  check("sidebar 'Run history' section: history shown, last run + KPIs hidden", () => {
    assert.strictEqual(env.dom.byId["run-history"].hidden, false);
    assert.strictEqual(env.dom.byId["kpi-panel"].hidden, true);
    assert.strictEqual(env.dom.byId["random-run-summary"].hidden, true);
  });
  navTo("dashboard");
  check("sidebar 'Dashboard' section: last run + KPIs back, history hidden", () => {
    assert.strictEqual(env.dom.byId["run-history"].hidden, true);
    assert.strictEqual(env.dom.byId["kpi-panel"].hidden, false);
    assert.strictEqual(env.dom.byId["random-run-summary"].hidden, false);
  });

  env.dom.byId["logout-link"].click();
  check("logout hides the KPI panel", () => assert.strictEqual(env.dom.byId["kpi-panel"].hidden, true));

  const coach = bootDashboard({ runtracker_session: JSON.stringify({ role: "Coach" }) });
  await settle(200);
  check("Coach: KPI panel stays hidden and no data is fetched", () => {
    assert.strictEqual(coach.dom.byId["kpi-panel"].hidden, true);
    assert.strictEqual(coach.fetchLog.length, 0);
  });

  const adv = bootDashboard({ runtracker_session: JSON.stringify({ role: "Advanced runner" }) });
  await settle(900);
  check("Advanced runner starts in Expert mode: expert cards for the history runner", () => {
    assert.deepStrictEqual(cardIds(adv), ["training_intensity_distribution", "internal_vs_external_load_coupling",
      "periodized_load_progression", "physiological_strain_and_fatigue_indicators", "not_available_expert"]);
  });

  console.log("\n" + passed + " passed, " + failed + " failed");
  process.exit(failed ? 1 : 0);
})();
