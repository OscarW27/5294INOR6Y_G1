// Random run summary: maps a workouts.csv row to ontology-style triples
// (docs/ontology/version1.ttl, prefix endo:) and builds one sentence from them.
// Plain JS, no libraries. Works in the browser (globals) and in Node (module.exports).
(function (root) {
  "use strict";

  // CSV workout_type -> ontology class (+ label from rdfs:label)
  var TYPE_TO_CLASS = {
    RECOVERY:  { cls: "RecoveryRun",  label: "recovery run",  opener: "an easy recovery run",  effort: "you were taking it easy" },
    STEADY:    { cls: "SteadyRun",    label: "steady run",    opener: "a steady run",          effort: "you were working at a steady pace" },
    INTENSIVE: { cls: "IntensiveRun", label: "intensive run", opener: "a hard run",            effort: "you were working hard" }
  };

  // CSV column -> ontology datatype property
  var COLUMN_TO_PREDICATE = {
    duration_min:  "durationMinutes",
    hr_mean:       "hrMean",
    hr_std:        "hrStd",
    hr_min:        "hrMin",
    hr_max:        "hrMax",
    speed_mean:    "speedMean",
    speed_max:     "speedMax",
    altitude_gain: "altitudeGain",
    // Not in version1.ttl; kept as an extension predicate so distance can be shown.
    distance_km:   "distanceKm"
  };

  // Run focus: per-run labels from the k-means clustering of run shapes
  // (analysis/run_types, k=5; clustering output served as data/run_types.csv).
  // The label is ML-found data keyed by workout_id, not computed from summary
  // metrics; runs without a label (727 of 40,186) get no focus sentence.
  var RUN_TYPE_TO_KEY = {
    "Easy / short":       "EasyShort",
    "Long & steady":      "LongSteady",
    "Tempo / hard":       "TempoHard",
    "Fartlek / variable": "FartlekVariable",
    "Stop-and-go":        "StopAndGo"
  };
  var RUN_TYPE_BY_ID = {};
  function setRunTypes(rows) {
    RUN_TYPE_BY_ID = {};
    (rows || []).forEach(function (row) {
      var key = RUN_TYPE_TO_KEY[row.run_type];
      if (key) RUN_TYPE_BY_ID[Number(row.workout_id)] = key;
    });
  }

  var FOCUS_TEXT = {
    EasyShort: {
      short: "easy / short",
      plain: "Overall it was an easy, short run: a relaxed pace and a modest distance, the kind of session that keeps you moving between harder days."
    },
    LongSteady: {
      short: "long & steady",
      plain: "Overall it was a long, steady run: you settled into an even rhythm and covered a good distance."
    },
    TempoHard: {
      short: "tempo / hard",
      plain: "Overall it was a tempo run: you pushed harder than your usual pace and kept it going, which really asks of the heart."
    },
    FartlekVariable: {
      short: "fartlek / variable",
      plain: "Overall it was a varied run: your speed went up and down a lot, like a fartlek mixing fast and easy stretches."
    },
    StopAndGo: {
      short: "stop-and-go",
      plain: "Overall it was a stop-and-go run: a low average speed with plenty of slow stretches, like walk breaks or a wandering route."
    }
  };

  // Parse workouts.csv (no quoted fields) into an array of row objects.
  function parseCsv(text) {
    var lines = text.split(/\r?\n/);
    var header = lines[0].split(",");
    var rows = [];
    for (var i = 1; i < lines.length; i++) {
      if (!lines[i]) continue;
      var cells = lines[i].split(",");
      var row = {};
      for (var j = 0; j < header.length; j++) row[header[j]] = cells[j];
      rows.push(row);
    }
    return rows;
  }

  // Row -> array of {s, p, o} plain-object triples.
  function rowToTriples(row) {
    var cls = TYPE_TO_CLASS[row.workout_type];
    if (!cls) return null;
    var s = "endo:Workout_" + row.workout_id;
    var triples = [
      { s: s, p: "rdf:type", o: "endo:" + cls.cls },
      { s: s, p: "workoutId", o: Number(row.workout_id) },
      { s: s, p: "performedBy", o: "endo:Athlete_" + row.user_id }
    ];
    Object.keys(COLUMN_TO_PREDICATE).forEach(function (col) {
      var v = parseFloat(row[col]);
      if (!isNaN(v)) triples.push({ s: s, p: COLUMN_TO_PREDICATE[col], o: v });
    });
    return triples;
  }

  function obj(triples, pred) {
    for (var i = 0; i < triples.length; i++) if (triples[i].p === pred) return triples[i].o;
    return undefined;
  }

  function classInfo(triples) {
    var name = String(obj(triples, "rdf:type")).replace("endo:", "");
    for (var k in TYPE_TO_CLASS) if (TYPE_TO_CLASS[k].cls === name) return TYPE_TO_CLASS[k];
    return { cls: name, label: name, opener: "a run", effort: "you were making an effort" };
  }

  function round(n, d) { var f = Math.pow(10, d); return Math.round(n * f) / f; }

  // mode: "Beginner" | "Expert"
  function buildSummary(triples, mode) {
    var info = classInfo(triples);
    var dur = obj(triples, "durationMinutes");
    var dist = obj(triples, "distanceKm");
    var hrMean = obj(triples, "hrMean");
    var hrStd = obj(triples, "hrStd");
    var spd = obj(triples, "speedMean");
    var up = obj(triples, "altitudeGain");
    var focusName = RUN_TYPE_BY_ID[obj(triples, "workoutId")];
    var focus = focusName ? FOCUS_TEXT[focusName] : undefined;

    if (mode === "Expert") {
      var parts = [];
      if (hrMean !== undefined) {
        parts.push("HR mean " + Math.round(hrMean) + (hrStd !== undefined ? " (sd " + round(hrStd, 1) + ")" : ""));
      }
      if (spd !== undefined) parts.push("speed mean " + round(spd, 1) + " km/h");
      if (up !== undefined) parts.push("ascent " + Math.round(up) + " m");
      if (focus) parts.push("focus: " + focus.short);
      var head = info.cls + (dist !== undefined ? " " + round(dist, 1) + " km" : "") +
        (dur !== undefined ? " / " + Math.round(dur) + " min" : "");
      return head + (parts.length ? "; " + parts.join(", ") : "") + ".";
    }

    // Beginner: plain words, explain terms, flowing sentences.
    var s = "That was " + info.opener;
    if (dist !== undefined && dur !== undefined) {
      s += ": " + round(dist, 1) + " km in " + Math.round(dur) + " minutes. ";
    } else if (dur !== undefined) {
      s += ", and it took " + Math.round(dur) + " minutes. ";
    } else {
      s += ". ";
    }
    if (hrMean !== undefined) {
      s += "Your heart rate stayed around " + Math.round(hrMean) + " beats per minute, which tells us " + info.effort + ". ";
    }
    if (up !== undefined && up >= 1) {
      s += "Along the way you climbed about " + Math.round(up) + " metres uphill. ";
    }
    if (focus) s += focus.plain;
    return s.trim();
  }

  // ---- run history / search helpers (all derived from triples) ----
  var CLASS_TO_SHORT = { RecoveryRun: "recovery", SteadyRun: "steady", IntensiveRun: "intense" };

  // Triples -> { id, athlete, label, minutes, km } (minutes/km may be null).
  function runInfo(triples) {
    var cls = String(obj(triples, "rdf:type")).replace("endo:", "");
    var dur = obj(triples, "durationMinutes");
    var dist = obj(triples, "distanceKm");
    return {
      id: obj(triples, "workoutId"),
      athlete: String(obj(triples, "performedBy")).replace("endo:Athlete_", ""),
      cls: cls,
      label: CLASS_TO_SHORT[cls] || cls,
      minutes: dur === undefined ? null : Math.round(dur),
      km: dist === undefined ? null : round(dist, 1)
    };
  }

  function formatRun(info) {
    return info.label + " — " + (info.minutes === null ? "? min" : info.minutes + " min") +
      " — " + (info.km === null ? "? km" : info.km.toFixed(1) + " km");
  }

  // Map every row through rowToTriples once; each entry gets a lowercase search string.
  function buildIndex(rows) {
    var out = [];
    for (var i = 0; i < rows.length; i++) {
      var t = rowToTriples(rows[i]);
      if (!t) continue;
      var info = runInfo(t);
      info.hay = ("runner " + info.athlete + " athlete " + info.athlete + " #" + info.id + " " +
        info.label + " " + info.cls + " " + formatRun(info)).toLowerCase();
      out.push(info);
    }
    return out;
  }

  // All whitespace-separated terms must appear. Returns at most `limit` matches plus the total.
  function searchIndex(index, query, limit) {
    var terms = String(query).toLowerCase().split(/\s+/).filter(Boolean);
    var hits = [], total = 0;
    for (var i = 0; i < index.length; i++) {
      var ok = true;
      for (var k = 0; k < terms.length; k++) if (index[i].hay.indexOf(terms[k]) === -1) { ok = false; break; }
      if (!ok) continue;
      total++;
      if (hits.length < limit) hits.push(index[i]);
    }
    return { hits: hits, total: total };
  }

  // =====================================================================
  // Running dashboard KPIs (docs/running_dashboard_kpis.json), one runner at a time.
  // All maths uses the REAL start_date / duration / HR columns of workouts.csv (and
  // timeseries.csv when the workout is in it) - never the fabricated run-history timestamps.
  //
  // Documented choices (my own cutoffs; the KPI file gives none):
  //  - Reference max HR (age is unknown, so no 220-age): the runner's highest hr_max over all
  //    their runs, floored at HRMAX_FLOOR (170) so a runner who never ran hard is not scored
  //    against a too-low ceiling.
  //  - HR zones, as a fraction of that reference: low < 80 %, moderate 80-90 %, high >= 90 %
  //    (three-zone model in the spirit of seiler2010: mostly-low "polarized/pyramidal" target ~80 % low).
  //  - Zone source per run: time-series (time-weighted by the gap to the next sample) when the
  //    workout is in timeseries.csv; otherwise ESTIMATED from the summary: HR is taken as a normal
  //    curve with mean hr_mean, sd hr_std (or (hr_max-hr_mean)/2 if hr_std is 0), cut off at
  //    hr_min / hr_max and renormalised; the zone shares are the curve's mass in each zone.
  //  - Load (training impulse style, Edwards-like): minutes x (1 x low + 2 x moderate + 3 x high share).
  //    "Intensity score" = load / minutes (1 = all low ... 3 = all high).
  //  - Acute load = load of the ISO week (7 days); chronic = mean weekly load of that week and the
  //    3 before it (28 days). Ratio acute/chronic: < 0.8 low, 0.8-1.3 balanced, > 1.3 high.
  //  - Strain: expected HR for a run's speed = least-squares line HR~speed over the runner's
  //    earlier runs (all but the latest 6; all runs if fewer than 5 earlier ones; plain mean HR
  //    if the speeds do not vary). A run is flagged when HR is >= 5 bpm above expected.
  //  - TID window = the runner's latest 10 runs; week tables = latest 8 ISO weeks up to the week
  //    of the runner's last run (the data is historical, so "now" = that last run).
  // =====================================================================
  var DAY_MS = 86400000;
  var HRMAX_FLOOR = 170;
  var ZONE_LOW_MAX = 0.80;
  var ZONE_MOD_MAX = 0.90;
  var TID_RECENT_RUNS = 10;
  var WEEKS_SHOWN = 8;
  var ACWR_LOW = 0.8, ACWR_HIGH = 1.3;
  var STRAIN_RECENT = 6;
  var STRAIN_BASELINE_MIN = 5;
  var STRAIN_FLAG_BPM = 5;
  var DRIFT_MIN_MINUTES = 20;
  var SESSIONS_SHOWN = 5;
  var COUPLING_SHOWN = 5;

  function pad2k(n) { return n < 10 ? "0" + n : "" + n; }
  function plural(n, word) { return n + " " + word + (n === 1 ? "" : "s"); }
  function pct(x) { return Math.round(x * 100); }
  function signed(n, d) { var v = round(n, d); return (v > 0 ? "+" : "") + v; }

  // Calendar day number (days since 1970-01-01, UTC) of "YYYY-MM-DD"; NaN if not a date.
  function parseDay(s) { return Math.floor(Date.parse(s + "T00:00:00Z") / DAY_MS); }
  // ISO week index: weeks start on Monday; index * 7 is that week's Thursday (day 0 = Thursday).
  function weekIndex(day) { return Math.floor((day + 3) / 7); }
  function weekLabel(wi) {
    var thu = wi * 7;
    var y = new Date(thu * DAY_MS).getUTCFullYear();
    var jan1 = Math.floor(Date.UTC(y, 0, 1) / DAY_MS);
    return y + "-W" + pad2k(Math.floor((thu - jan1) / 7) + 1);
  }
  function fmtPace(p) {
    var m = Math.floor(p), s = Math.round((p - m) * 60);
    if (s === 60) { m++; s = 0; }
    return m + ":" + pad2k(s);
  }

  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : NaN; }
  function median(a) {
    var b = a.slice().sort(function (x, y) { return x - y; }), n = b.length;
    return n ? (n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2) : NaN;
  }
  // Least squares y = a + b x; null if x does not vary or fewer than 2 points.
  function linfit(xs, ys) {
    var n = xs.length;
    if (n < 2) return null;
    var mx = mean(xs), my = mean(ys), sxx = 0, sxy = 0;
    for (var i = 0; i < n; i++) { sxx += (xs[i] - mx) * (xs[i] - mx); sxy += (xs[i] - mx) * (ys[i] - my); }
    if (sxx < 1e-12) return null;
    return { a: my - (sxy / sxx) * mx, b: sxy / sxx };
  }
  function pearson(xs, ys) {
    var n = xs.length;
    if (n < 3) return null;
    var mx = mean(xs), my = mean(ys), sxx = 0, syy = 0, sxy = 0;
    for (var i = 0; i < n; i++) {
      var dx = xs[i] - mx, dy = ys[i] - my;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    if (sxx < 1e-12 || syy < 1e-12) return null;
    return sxy / Math.sqrt(sxx * syy);
  }

  // All usable runs of one runner from the parsed CSV rows, oldest first.
  // A row needs a valid date, duration > 0, speed > 0, hr_mean > 0 and a distance.
  function userRuns(rows, uid) {
    var id = String(uid), out = [];
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (r.user_id !== id) continue;
      var day = parseDay(r.start_date), minutes = parseFloat(r.duration_min);
      var speed = parseFloat(r.speed_mean), hrMean = parseFloat(r.hr_mean), km = parseFloat(r.distance_km);
      if (isNaN(day) || !(minutes > 0) || !(speed > 0) || !(hrMean > 0) || isNaN(km)) continue;
      var up = parseFloat(r.altitude_gain);
      out.push({
        id: r.workout_id, athlete: id, day: day, date: r.start_date, minutes: minutes, km: km,
        speed: speed, hrMean: hrMean, hrStd: parseFloat(r.hr_std), hrMin: parseFloat(r.hr_min),
        hrMax: parseFloat(r.hr_max), ascent: isNaN(up) ? 0 : up, row: r
      });
    }
    out.sort(function (a, b) { return a.day - b.day || Number(a.id) - Number(b.id); });
    return out;
  }

  // timeseries.csv text -> { workout_id: [[t_min, hr, speed, altitude], ...] } sorted by time.
  function parseTimeseries(text) {
    var lines = text.split(/\r?\n/), head = lines[0].split(","), map = {};
    var iw = head.indexOf("workout_id"), it = head.indexOf("t_min"), ih = head.indexOf("heart_rate"),
        is = head.indexOf("speed"), ia = head.indexOf("altitude");
    for (var i = 1; i < lines.length; i++) {
      if (!lines[i]) continue;
      var c = lines[i].split(",");
      var t = parseFloat(c[it]), hr = parseFloat(c[ih]), sp = parseFloat(c[is]), al = parseFloat(c[ia]);
      if (isNaN(t) || isNaN(hr) || isNaN(sp)) continue;
      (map[c[iw]] = map[c[iw]] || []).push([t, hr, sp, isNaN(al) ? 0 : al]);
    }
    for (var k in map) map[k].sort(function (a, b) { return a[0] - b[0]; });
    return map;
  }

  // Each sample counts for the time until the next one (last sample: 0). Equal weights if all 0.
  function sampleWeights(s) {
    var w = [], sum = 0, i;
    for (i = 0; i < s.length; i++) { w.push(i < s.length - 1 ? s[i + 1][0] - s[i][0] : 0); sum += w[i]; }
    if (sum <= 0) for (i = 0; i < w.length; i++) w[i] = 1;
    return w;
  }

  function zoneOf(hr, ref) { return hr < ZONE_LOW_MAX * ref ? 0 : hr < ZONE_MOD_MAX * ref ? 1 : 2; }

  function zonesFromSeries(s, ref) {
    var w = sampleWeights(s), z = [0, 0, 0], tot = 0;
    for (var i = 0; i < s.length; i++) { z[zoneOf(s[i][1], ref)] += w[i]; tot += w[i]; }
    return { low: z[0] / tot, mod: z[1] / tot, high: z[2] / tot };
  }

  function normCdf(x) { // Abramowitz-Stegun 7.1.26 erf approximation
    var sign = x < 0 ? -1 : 1, a = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * a);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return 0.5 * (1 + sign * y);
  }

  function zonesFromSummary(run, ref) {
    var mu = run.hrMean;
    var sd = run.hrStd > 0 ? run.hrStd : Math.max((run.hrMax - mu) / 2, 1);
    if (isNaN(sd)) sd = 1;
    var hi = run.hrMax > mu ? run.hrMax : mu + 3 * sd;
    var lo = run.hrMin < mu ? run.hrMin : mu - 3 * sd;
    function F(x) { return normCdf((Math.min(Math.max(x, lo), hi) - mu) / sd); }
    var total = F(hi) - F(lo);
    var c1 = ZONE_LOW_MAX * ref, c2 = ZONE_MOD_MAX * ref;
    if (!(total > 1e-6)) { var z = [0, 0, 0]; z[zoneOf(mu, ref)] = 1; return { low: z[0], mod: z[1], high: z[2] }; }
    return { low: (F(c1) - F(lo)) / total, mod: (F(c2) - F(c1)) / total, high: (F(hi) - F(c2)) / total };
  }

  // First half vs second half (by time) of the run: change of the HR/speed ratio, in percent.
  function seriesDrift(s, minutes) {
    if (minutes < DRIFT_MIN_MINUTES || s.length < 10) return null;
    var w = sampleWeights(s), mid = (s[0][0] + s[s.length - 1][0]) / 2;
    var a = { h: 0, v: 0, w: 0 }, b = { h: 0, v: 0, w: 0 };
    for (var i = 0; i < s.length; i++) {
      var g = s[i][0] < mid ? a : b;
      g.h += s[i][1] * w[i]; g.v += s[i][2] * w[i]; g.w += w[i];
    }
    if (a.w <= 0 || b.w <= 0) return null;
    var h1 = a.h / a.w, v1 = a.v / a.w, h2 = b.h / b.w, v2 = b.v / b.w;
    if (v1 <= 1 || v2 <= 1) return null;
    return ((h2 / v2) / (h1 / v1) - 1) * 100;
  }

  function refHrMax(runs) {
    var m = HRMAX_FLOOR;
    for (var i = 0; i < runs.length; i++) if (runs[i].hrMax > m) m = runs[i].hrMax;
    return m;
  }

  // Per-run derived metrics. series = timeseries rows of this workout or undefined.
  function runMetrics(run, ref, series) {
    var hasSeries = !!series && series.length >= 2;
    var zones = hasSeries ? zonesFromSeries(series, ref) : zonesFromSummary(run, ref);
    var score = zones.low + 2 * zones.mod + 3 * zones.high;
    var pace = 60 / run.speed, m = {
      id: run.id, athlete: run.athlete, day: run.day, date: run.date, minutes: run.minutes, km: run.km,
      speed: run.speed, pace: pace, hrMean: run.hrMean, hrMax: run.hrMax, ascent: run.ascent,
      ascentPerKm: run.km > 0 ? run.ascent / run.km : null,
      beatsPerKm: run.hrMean * pace,
      zones: zones, zoneSource: hasSeries ? "TimeSeriesZones" : "SummaryEstimatedZones",
      score: score, load: run.minutes * score,
      seriesR: null, drift: null, run: run
    };
    if (hasSeries) {
      m.seriesR = pearson(series.map(function (p) { return p[1]; }), series.map(function (p) { return p[2]; }));
      m.drift = seriesDrift(series, run.minutes);
    }
    return m;
  }

  // Triples for the beginner session summary (existing mapping + start date and pace).
  function sessionTriples(run) {
    var t = rowToTriples(run.row) || [{ s: "endo:Workout_" + run.id, p: "rdf:type", o: "endo:RunningWorkout" }];
    var s = "endo:Workout_" + run.id;
    t.push({ s: s, p: "startDate", o: run.date });
    t.push({ s: s, p: "paceMinPerKm", o: 60 / run.speed });
    return t;
  }

  // Triples for the expert metrics: session triples + zones, zone source, load, cardiac cost, drift.
  function metricTriples(m) {
    var t = sessionTriples(m.run), s = "endo:Workout_" + m.id;
    t.push({ s: s, p: "lowZoneShare", o: m.zones.low });
    t.push({ s: s, p: "moderateZoneShare", o: m.zones.mod });
    t.push({ s: s, p: "highZoneShare", o: m.zones.high });
    t.push({ s: s, p: "zoneSource", o: "endo:" + m.zoneSource });
    t.push({ s: s, p: "trainingLoad", o: m.load });
    t.push({ s: s, p: "beatsPerKm", o: m.beatsPerKm });
    if (m.drift !== null) t.push({ s: s, p: "hrDriftPct", o: m.drift });
    return t;
  }

  // Build everything the cards need for one runner. tsMap may be null (no time series loaded).
  function buildKpiContext(rows, uid, tsMap) {
    var runs = userRuns(rows, uid);
    if (!runs.length) return null;
    var ref = refHrMax(runs);
    var metrics = runs.map(function (r) { return runMetrics(r, ref, tsMap ? tsMap[r.id] : undefined); });
    return { athlete: String(uid), runs: runs, metrics: metrics, ref: ref, tsLoaded: !!tsMap };
  }

  // ---- KPI 1: weekly consistency and persistence ----
  function consistencyKpi(runs) {
    var byWeek = {};
    runs.forEach(function (r) {
      var w = weekIndex(r.day), b = byWeek[w] || (byWeek[w] = { sessions: 0, minutes: 0 });
      b.sessions++; b.minutes += r.minutes;
    });
    var keys = Object.keys(byWeek).map(Number).sort(function (a, b) { return a - b; });
    var first = keys[0], last = keys[keys.length - 1], longest = 0, cur = 0, prev = null, sessions = 0, minutes = 0;
    keys.forEach(function (k) {
      cur = prev !== null && k === prev + 1 ? cur + 1 : 1;
      if (cur > longest) longest = cur;
      prev = k; sessions += byWeek[k].sessions; minutes += byWeek[k].minutes;
    });
    var weeks = [];
    for (var w = last - WEEKS_SHOWN + 1; w <= last; w++) {
      var b = byWeek[w];
      weeks.push({ label: weekLabel(w), sessions: b ? b.sessions : 0, minutes: b ? b.minutes : 0 });
    }
    return {
      firstDate: runs[0].date, lastDate: runs[runs.length - 1].date,
      spanWeeks: last - first + 1, activeWeeks: keys.length, sessions: sessions, minutes: minutes,
      longest: longest, current: cur, weeks: weeks
    };
  }

  // ---- KPI 2: simple session summaries (from triples) ----
  function buildSessionLine(triples) {
    var date = obj(triples, "startDate"), dist = obj(triples, "distanceKm"),
        dur = obj(triples, "durationMinutes"), pace = obj(triples, "paceMinPerKm");
    var s = "On " + date + " you ran " + (dist !== undefined ? round(dist, 1) + " km" : "a run") +
      (dur !== undefined ? " in " + plural(Math.round(dur), "minute") : "") + ".";
    if (pace !== undefined) s += " Your average pace was " + fmtPace(pace) + " minutes per kilometre.";
    return s;
  }

  // ---- KPI 3: training intensity distribution ----
  function aggregateZones(list) {
    if (!list.length) return null;
    var min = 0, z = { low: 0, mod: 0, high: 0 };
    list.forEach(function (m) {
      min += m.minutes; z.low += m.zones.low * m.minutes; z.mod += m.zones.mod * m.minutes; z.high += m.zones.high * m.minutes;
    });
    return { runs: list.length, minutes: min, low: z.low / min, mod: z.mod / min, high: z.high / min };
  }
  function tidKpi(ms) {
    var recent = ms.slice(-TID_RECENT_RUNS);
    return {
      from: recent[0].date, to: recent[recent.length - 1].date,
      all: aggregateZones(recent),
      series: aggregateZones(recent.filter(function (m) { return m.zoneSource === "TimeSeriesZones"; })),
      estimated: aggregateZones(recent.filter(function (m) { return m.zoneSource !== "TimeSeriesZones"; }))
    };
  }

  // ---- expert one-line run description (from triples) ----
  function buildExpertRunLine(triples) {
    var cls = String(obj(triples, "rdf:type")).replace("endo:", "");
    var dist = obj(triples, "distanceKm"), dur = obj(triples, "durationMinutes"), spd = obj(triples, "speedMean"),
        hr = obj(triples, "hrMean"), pace = obj(triples, "paceMinPerKm"), bpk = obj(triples, "beatsPerKm"),
        up = obj(triples, "altitudeGain"), lo = obj(triples, "lowZoneShare"), mo = obj(triples, "moderateZoneShare"),
        hi = obj(triples, "highZoneShare"), load = obj(triples, "trainingLoad"),
        src = String(obj(triples, "zoneSource")).replace("endo:", "") === "TimeSeriesZones" ? "time-series" : "summary estimate";
    var parts = [];
    if (spd !== undefined) parts.push("speed " + round(spd, 1) + " km/h" + (pace !== undefined ? " (" + fmtPace(pace) + "/km)" : ""));
    if (hr !== undefined) parts.push("HR " + Math.round(hr) + (bpk !== undefined ? " (" + Math.round(bpk) + " beats/km)" : ""));
    if (up !== undefined) {
      parts.push("ascent " + Math.round(up) + " m" + (dist > 0 ? " (" + round(up / dist, 1) + " m/km)" : ""));
    }
    if (lo !== undefined) parts.push("zones L/M/H " + pct(lo) + "/" + pct(mo) + "/" + pct(hi) + " % [" + src + "]");
    if (load !== undefined) parts.push("load " + Math.round(load));
    return obj(triples, "startDate") + " " + cls + (dist !== undefined ? " " + round(dist, 1) + " km" : "") +
      (dur !== undefined ? " / " + Math.round(dur) + " min" : "") + (parts.length ? "; " + parts.join(", ") : "") + ".";
  }

  // ---- KPI 4: internal vs external load coupling ----
  function couplingKpi(ms) {
    var up = ms.filter(function (m) { return m.ascentPerKm !== null; });
    var med = median(ms.map(function (m) { return m.beatsPerKm; }));
    return {
      runs: ms.length, medianBeatsPerKm: med,
      rHrSpeed: pearson(ms.map(function (m) { return m.hrMean; }), ms.map(function (m) { return m.speed; })),
      rCostAscent: pearson(up.map(function (m) { return m.ascentPerKm; }), up.map(function (m) { return m.beatsPerKm; })),
      recent: ms.slice(-COUPLING_SHOWN)
    };
  }

  // ---- KPI 5: periodized load progression ----
  function periodizedKpi(ms) {
    var byWeek = {};
    ms.forEach(function (m) {
      var w = weekIndex(m.day), b = byWeek[w] || (byWeek[w] = { km: 0, minutes: 0, load: 0, sessions: 0 });
      b.km += m.km; b.minutes += m.minutes; b.load += m.load; b.sessions++;
    });
    function loadOf(w) { return byWeek[w] ? byWeek[w].load : 0; }
    var last = weekIndex(ms[ms.length - 1].day), weeks = [];
    for (var w = last - WEEKS_SHOWN + 1; w <= last; w++) {
      var b = byWeek[w], chronic = (loadOf(w) + loadOf(w - 1) + loadOf(w - 2) + loadOf(w - 3)) / 4;
      var ratio = chronic > 0 ? loadOf(w) / chronic : null;
      weeks.push({
        label: weekLabel(w), sessions: b ? b.sessions : 0, km: b ? b.km : 0, minutes: b ? b.minutes : 0,
        intensity: b ? b.load / b.minutes : null, acute: loadOf(w), chronic: chronic, ratio: ratio,
        status: ratio === null ? "n/a" : ratio < ACWR_LOW ? "low" : ratio > ACWR_HIGH ? "high" : "balanced"
      });
    }
    var idx = weeks.map(function (_, i) { return i; });
    var kmFit = linfit(idx, weeks.map(function (x) { return x.km; }));
    var act = weeks.map(function (x, i) { return { i: i, v: x.intensity }; }).filter(function (x) { return x.v !== null; });
    var inFit = linfit(act.map(function (x) { return x.i; }), act.map(function (x) { return x.v; }));
    return { weeks: weeks, kmPerWeekTrend: kmFit ? kmFit.b : null, intensityTrend: inFit ? inFit.b : null };
  }

  // ---- KPI 6: physiological strain and fatigue ----
  function strainKpi(ms) {
    var recent = ms.slice(-STRAIN_RECENT), base = ms.slice(0, ms.length - recent.length), usedAll = false;
    if (base.length < STRAIN_BASELINE_MIN) { base = ms; usedAll = true; }
    var fit = base.length >= STRAIN_BASELINE_MIN
      ? linfit(base.map(function (m) { return m.speed; }), base.map(function (m) { return m.hrMean; })) : null;
    var baseHr = mean(base.map(function (m) { return m.hrMean; }));
    var rows = recent.map(function (m) {
      var expected = fit ? fit.a + fit.b * m.speed : baseHr;
      var resid = m.hrMean - expected;
      return { m: m, expected: expected, resid: resid, flag: resid >= STRAIN_FLAG_BPM };
    });
    return {
      rows: rows, baselineRuns: base.length, usedAll: usedAll, model: fit ? "line" : "mean",
      meanResid: mean(rows.map(function (r) { return r.resid; })),
      flagged: rows.filter(function (r) { return r.flag; }).length
    };
  }

  function zoneText(a) { return "low " + pct(a.low) + " / mod " + pct(a.mod) + " / high " + pct(a.high) + " %"; }

  // Cards for the chosen mode: [{ id, title, lines: [string], placeholder? }]
  function kpiCards(ctx, mode) {
    var runs = ctx.runs, ms = ctx.metrics, cards = [], lines;

    if (mode !== "Expert") {
      // Beginner: plain words, explain terms.
      var c = consistencyKpi(runs);
      lines = [
        "You ran in " + c.activeWeeks + " of the " + c.spanWeeks + " weeks between " + c.firstDate + " and " + c.lastDate + ".",
        "Your longest streak is " + plural(c.longest, "week") + " in a row (a streak is weeks in a row with at least one run).",
        "Your latest streak is " + plural(c.current, "week") + ", counted up to your last run on " + c.lastDate + ".",
        "In the weeks you ran, you did about " + round(c.sessions / c.activeWeeks, 1) + " runs and " +
          Math.round(c.minutes / c.activeWeeks) + " active minutes per week."
      ];
      cards.push({
        id: "weekly_consistency_and_persistence", title: "Weekly consistency", lines: lines,
        stats: [
          { value: c.activeWeeks + "/" + c.spanWeeks, label: "weeks with a run" },
          { value: String(c.longest), label: "longest streak (weeks)" },
          { value: String(c.current), label: "latest streak (weeks)" }
        ],
        bars: {
          title: "Active minutes in your last " + WEEKS_SHOWN + " weeks",
          items: c.weeks.map(function (w) {
            return { label: w.label.slice(5), value: w.minutes, cls: w.sessions ? "good" : "mute",
              text: w.sessions ? Math.round(w.minutes) + " min (" + plural(w.sessions, "run") + ")" : "no runs" };
          })
        }
      });

      var recent = runs.slice(-SESSIONS_SHOWN).reverse();
      lines = recent.map(function (r) { return buildSessionLine(sessionTriples(r)); });
      var km = 0, min = 0;
      recent.forEach(function (r) { km += r.km; min += r.minutes; });
      lines.push("Together these " + recent.length + " runs: " + round(km, 1) + " km in " + Math.round(min) + " minutes" +
        (km > 0 ? ", an average pace of " + fmtPace(min / km) + " minutes per kilometre." : "."));
      cards.push({
        id: "simple_session_summaries", title: "Your latest runs", lines: lines,
        stats: [
          { value: round(recent[0].km, 1) + " km", label: "last run, " + Math.round(recent[0].minutes) + " min" },
          { value: fmtPace(60 / recent[0].speed), label: "last run pace, min/km" },
          { value: round(km, 1) + " km", label: "last " + recent.length + " runs together" }
        ]
      });

      cards.push({
        id: "not_available_beginner", placeholder: true, title: "Not available yet",
        lines: ["Run tags (casual or goal run) and park or street routes need information this dataset does not have, so they are left out."]
      });
      return cards;
    }

    // Expert: compact.
    var t = tidKpi(ms);
    lines = ["TID " + t.from + " to " + t.to + " (" + plural(t.all.runs, "run") + ", " + Math.round(t.all.minutes) + " min): " + zoneText(t.all) +
      "; ref HRmax " + Math.round(ctx.ref) + ", cuts <" + pct(ZONE_LOW_MAX) + " / " + pct(ZONE_LOW_MAX) + "-" + pct(ZONE_MOD_MAX) + " / >=" + pct(ZONE_MOD_MAX) + " %."];
    lines.push("Source time-series: " + (t.series ? plural(t.series.runs, "run") + ", " + Math.round(t.series.minutes) + " min: " + zoneText(t.series) : "no run in timeseries.csv") +
      (ctx.tsLoaded ? "." : " (time-series file not loaded)."));
    lines.push("Source summary estimate (hr_mean/hr_std/hr_min/hr_max): " +
      (t.estimated ? plural(t.estimated.runs, "run") + ", " + Math.round(t.estimated.minutes) + " min: " + zoneText(t.estimated) : "none") + ".");
    lines.push("Low share " + pct(t.all.low) + " % vs ~80 % polarized/pyramidal target.");
    var zoneRows = [{ label: "All " + plural(t.all.runs, "run"), z: t.all }];
    if (t.series) zoneRows.push({ label: "Time-series (" + plural(t.series.runs, "run") + ")", z: t.series });
    if (t.estimated) zoneRows.push({ label: "Summary estimate (" + plural(t.estimated.runs, "run") + ")", z: t.estimated });
    cards.push({
      id: "training_intensity_distribution", title: "Training intensity distribution", lines: lines, collapse: 1,
      stats: [
        { value: pct(t.all.low) + " %", label: "low (target ~80 %)" },
        { value: pct(t.all.mod) + " %", label: "moderate" },
        { value: pct(t.all.high) + " %", label: "high" }
      ],
      zones: zoneRows.map(function (r) { return { label: r.label, low: r.z.low, mod: r.z.mod, high: r.z.high }; })
    });

    var k = couplingKpi(ms);
    lines = ["All " + k.runs + " runs: r(HR, speed) " + (k.rHrSpeed === null ? "n/a" : round(k.rHrSpeed, 2)) +
      ", r(beats/km, ascent m/km) " + (k.rCostAscent === null ? "n/a" : round(k.rCostAscent, 2)) +
      ", median " + Math.round(k.medianBeatsPerKm) + " beats/km."];
    k.recent.slice().reverse().forEach(function (m) {
      var extra = "; beats/km " + signed((m.beatsPerKm / k.medianBeatsPerKm - 1) * 100, 0) + " % vs median";
      if (m.seriesR !== null) extra += "; within-run r(HR, speed) " + round(m.seriesR, 2) + " [time-series]";
      lines.push(buildExpertRunLine(metricTriples(m)).replace(/\.$/, "") + extra + ".");
    });
    cards.push({
      id: "internal_vs_external_load_coupling", title: "Internal vs external load", lines: lines, collapse: 1,
      stats: [
        { value: k.rHrSpeed === null ? "n/a" : String(round(k.rHrSpeed, 2)), label: "r (HR, speed)" },
        { value: String(Math.round(k.medianBeatsPerKm)), label: "median beats/km" }
      ],
      bars: {
        title: "Beats per km, latest " + k.recent.length + " runs (lower = cheaper)",
        items: k.recent.map(function (m) {
          return { label: m.date.slice(5), value: m.beatsPerKm, cls: m.beatsPerKm > k.medianBeatsPerKm * 1.05 ? "warn" : "good",
            text: Math.round(m.beatsPerKm) + (m.zoneSource === "TimeSeriesZones" ? " (series)" : "") };
        })
      }
    });

    var p = periodizedKpi(ms);
    lines = ["Trend, last " + WEEKS_SHOWN + " ISO weeks: volume " + (p.kmPerWeekTrend === null ? "n/a" : signed(p.kmPerWeekTrend, 1) + " km/wk per wk") +
      ", intensity score " + (p.intensityTrend === null ? "n/a" : signed(p.intensityTrend, 2) + " per wk") + " (1 low ... 3 high)."];
    p.weeks.slice().reverse().forEach(function (w) {
      lines.push(w.label + ": " + (w.sessions ? w.sessions + "x, " + round(w.km, 1) + " km, " + Math.round(w.minutes) + " min, intensity " + round(w.intensity, 2) : "rest") +
        "; acute " + Math.round(w.acute) + " / chronic " + Math.round(w.chronic) +
        (w.ratio === null ? " = n/a" : " = " + round(w.ratio, 2) + " " + w.status) + ".");
    });
    var lastW = p.weeks[p.weeks.length - 1];
    cards.push({
      id: "periodized_load_progression", title: "Periodized load progression", lines: lines, collapse: 1,
      stats: [
        { value: p.kmPerWeekTrend === null ? "n/a" : signed(p.kmPerWeekTrend, 1), label: "km/week trend per week" },
        { value: lastW.ratio === null ? "n/a" : String(round(lastW.ratio, 2)), label: "acute/chronic, latest week (" + lastW.status + ")" }
      ],
      bars: {
        title: "Weekly km (colour = acute/chronic balance)",
        items: p.weeks.map(function (w) {
          return { label: w.label.slice(5), value: w.km, cls: w.status === "balanced" ? "good" : w.status === "n/a" ? "mute" : w.status === "high" ? "bad" : "warn",
            text: w.sessions ? round(w.km, 1) + " km, " + (w.ratio === null ? "n/a" : round(w.ratio, 2)) : "rest" };
        })
      }
    });

    var s = strainKpi(ms);
    lines = ["HR vs expected for pace (" + (s.model === "line" ? "line HR~speed" : "mean HR") + " over " + s.baselineRuns + " " +
      (s.usedAll ? "runs incl. recent" : "earlier runs") + "): mean " + signed(s.meanResid, 1) + " bpm over " + plural(s.rows.length, "run") +
      "; " + s.flagged + " flagged (>= +" + STRAIN_FLAG_BPM + " bpm)."];
    s.rows.slice().reverse().forEach(function (r) {
      var m = r.m;
      lines.push(m.date + ": pace " + fmtPace(m.pace) + "/km, HR " + Math.round(m.hrMean) + " (expected " + Math.round(r.expected) + ", " +
        signed(r.resid, 1) + " bpm)" + (r.flag ? " FLAG" : "") +
        (m.drift !== null ? "; drift " + signed(m.drift, 1) + " % [time-series]" : "") + ".");
    });
    cards.push({
      id: "physiological_strain_and_fatigue_indicators", title: "Physiological strain and fatigue", lines: lines, collapse: 1,
      stats: [
        { value: signed(s.meanResid, 1) + " bpm", label: "mean HR vs expected" },
        { value: s.flagged + "/" + s.rows.length, label: "runs flagged (>= +" + STRAIN_FLAG_BPM + " bpm)" }
      ],
      bars: {
        title: "HR above expected for pace, latest " + s.rows.length + " runs (bpm; 0 = as usual)",
        items: s.rows.map(function (r) {
          return { label: r.m.date.slice(5), value: Math.max(r.resid, 0), cls: r.flag ? "bad" : "good",
            text: signed(r.resid, 1) + " bpm" + (r.flag ? " flag" : "") };
        })
      }
    });

    cards.push({
      id: "not_available_expert", placeholder: true, title: "Not available in this dataset",
      lines: ["Plan adherence needs user-supplied training plan metadata or workout tags; not in the data."]
    });
    return cards;
  }

  var api = {
    // KPIs
    parseDay: parseDay, weekIndex: weekIndex, weekLabel: weekLabel, fmtPace: fmtPace,
    userRuns: userRuns, parseTimeseries: parseTimeseries, refHrMax: refHrMax,
    zonesFromSeries: zonesFromSeries, zonesFromSummary: zonesFromSummary, runMetrics: runMetrics,
    sessionTriples: sessionTriples, metricTriples: metricTriples, buildKpiContext: buildKpiContext,
    consistencyKpi: consistencyKpi, buildSessionLine: buildSessionLine, tidKpi: tidKpi,
    couplingKpi: couplingKpi, periodizedKpi: periodizedKpi, strainKpi: strainKpi,
    buildExpertRunLine: buildExpertRunLine, kpiCards: kpiCards,
    parseCsv: parseCsv,
    rowToTriples: rowToTriples,
    setRunTypes: setRunTypes,
    buildSummary: buildSummary,
    runInfo: runInfo,
    formatRun: formatRun,
    buildIndex: buildIndex,
    searchIndex: searchIndex
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RunSummary = api;
})(typeof window !== "undefined" ? window : this);
