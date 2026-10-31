// ServeTell front end: fills the story from data/findings.json and runs the matchup model in the browser.
const AIMS = ['wide', 'body', 'T'];
const AIM_LABEL = { wide: 'Wide', body: 'Body', T: 'T' };
const SIDES = ['deuce', 'ad'];
const SURFACES = ['Hard', 'Clay', 'Grass', 'Other'];
const SHRINK_K = 100;  // same pseudo-count the analysis used for "what comes next" mixes
const Z90 = 1.645;
const DEFAULTS = { atp: ['Jannik Sinner', 'Carlos Alcaraz'], wta: ['Aryna Sabalenka', 'Iga Swiatek'] };

const state = { findings: null, models: {}, tour: 'atp' };
const $ = (s, el = document) => el.querySelector(s);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (x, d = 1) => (x == null || !isFinite(x)) ? '–' : (x * 100).toFixed(d) + '%';
const pts = (x, d = 1) => (x >= 0 ? '+' : '−') + Math.abs(x * 100).toFixed(d);
const sigm = x => 1 / (1 + Math.exp(-x));
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const sum = a => a.reduce((x, y) => x + y, 0);

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function model(tour) {
  if (!state.models[tour]) state.models[tour] = await getJSON(`data/model-${tour}.json`);
  return state.models[tour];
}

// ------------------------------------------------------------------ story numbers

function bindStory(F) {
  const byTour = (list, t) => list.find(x => x.tour === t);
  const val = F.validation;
  const rel = t => val[t].direction_check.raw_slope;
  const relText = t => rel(t) <= 0.005 ? '0%' : pct(rel(t), 0);
  const ser = t => byTour(F.serial.by_tour, t);
  const ant = t => byTour(F.anticipation, t);
  const prs = t => byTour(F.pressure, t);
  const sec = t => byTour(F.two_first, t);
  const mm = F.minimax.by_tour;
  const cells = F.baselines.cells.filter(c => c.serve === 1);
  const bodyGaps = [];
  for (const t of ['ATP', 'WTA']) for (const sd of SIDES) {
    const g = a => cells.find(c => c.tour === t && c.side === sd && c.aim === a).won;
    bodyGaps.push((g('T') + g('wide')) / 2 - g('body'));
  }
  const drops = [];
  for (const t of ['ATP', 'WTA']) for (const a of ['wide', 'T']) drops.push(prs(t).won_normal[a] - prs(t).won_bp[a]);
  const secPlayers = sum(F.two_first.map(x => x.players));
  const secGain = sum(F.two_first.map(x => Math.round(x.share_gain * x.players)));
  const sigNames = F.two_first.flatMap(x => x.top.map(p => p.player));
  const readGain = t => ant(t).acc_tell - ant(t).acc_usual;
  const k = {
    serves: (F.meta.serves / 1e6).toFixed(2) + ' million',
    tile_reliab: relText('ATP'),
    atp_reliab: relText('ATP'), wta_reliab: relText('WTA'),
    tile_switch: pct(ser('WTA').share_sig_switch, 0),
    atp_sig_switch: pct(ser('ATP').share_sig_switch, 0), wta_sig_switch: pct(ser('WTA').share_sig_switch, 0),
    tile_read: pts(readGain('ATP')) + ' pts', wta_read: pts(readGain('WTA')),
    tile_second: pct(1 - secGain / secPlayers, 0),
    minimax_tests: sum(mm.map(x => x.tests)).toLocaleString(),
    minimax_p05: pct(sum(mm.map(x => x.share_p05 * x.tests)) / sum(mm.map(x => x.tests)), 0),
    body_gap: (sum(bodyGaps) / bodyGaps.length * 100).toFixed(1),
    atp_rep: pct(ser('ATP').obs), atp_rep_exp: pct(ser('ATP').exp), wta_rep: pct(ser('WTA').obs), wta_rep_exp: pct(ser('WTA').exp),
    atp_acc_usual: pct(ant('ATP').acc_usual), atp_acc_tell: pct(ant('ATP').acc_tell),
    wta_acc_usual: pct(ant('WTA').acc_usual), wta_acc_tell: pct(ant('WTA').acc_tell),
    atp_fav_obs: pct(prs('ATP').fav_obs), atp_fav_exp: pct(prs('ATP').fav_exp),
    wta_fav_obs: pct(prs('WTA').fav_obs), wta_fav_exp: pct(prs('WTA').fav_exp),
    bp_drop: (sum(drops) / drops.length * 100).toFixed(1) + ' points per 100',
    atp_second_med: (-sec('ATP').median_diff * 100).toFixed(1) + ' points per 100 second serves',
    wta_second_med: (-sec('WTA').median_diff * 100).toFixed(1),
    second_gain: String(secGain), second_players: String(secPlayers),
    second_sig: sigNames.length === 0 ? 'none' : sigNames.length === 1 ? `one player (${sigNames[0]})` : `${sigNames.length} (${sigNames.join(', ')})`,
    matches: F.meta.matches.toLocaleString(), atp_matches: F.meta.by_tour.ATP.matches.toLocaleString(),
    wta_matches: F.meta.by_tour.WTA.matches.toLocaleString(),
    range: `${F.meta.first.slice(0, 4)} to ${F.meta.last.slice(0, 4)}`,
    lam_v: val.ATP.lam[2].toLocaleString(),
    atp_test_matches: val.ATP.test_matches.toLocaleString(), wta_test_matches: val.WTA.test_matches.toLocaleString(),
  };
  setKeys(k);

  const rows = [['Coin flip (50%)', null], ['Tour baseline only', 'base'], ['+ server strength', 'server'], ['+ server and returner (ServeTell)', 'full']];
  const cell = (t, key, m) => key ? val[t][key][m].toFixed(4) : (m === 'logloss' ? Math.LN2.toFixed(4) : '0.2500');
  $('#val-table').innerHTML = `<thead><tr><th>Model, scored on 2025–26</th><th>ATP log-loss</th><th>ATP Brier</th><th>WTA log-loss</th><th>WTA Brier</th></tr></thead><tbody>` +
    rows.map(([name, key]) => `<tr><td>${name}</td>${['ATP', 'WTA'].flatMap(t => ['logloss', 'brier'].map(m => `<td>${cell(t, key, m)}</td>`)).join('')}</tr>`).join('') +
    `</tbody>`;
}

function setKeys(k) {
  document.querySelectorAll('[data-k]').forEach(el => { if (k[el.dataset.k] != null) el.textContent = k[el.dataset.k]; });
}

// ------------------------------------------------------------------ charts

let clipId = 0;
function plot(el, o) {
  const W = o.w || 520, H = o.h || 360, m = Object.assign({ l: 54, r: 18, t: 14, b: 46 }, o.m || {});
  const [x0, x1] = o.x.domain, [y0, y1] = o.y.domain;
  const sx = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const sy = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const id = `clip${clipId++}`;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.label)}"><defs><clipPath id="${id}"><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/></clipPath></defs>`;
  for (const t of o.x.ticks) s += `<g class="grid"><line x1="${sx(t)}" x2="${sx(t)}" y1="${m.t}" y2="${H - m.b}"/></g><text x="${sx(t)}" y="${H - m.b + 17}" text-anchor="middle">${o.x.fmt(t)}</text>`;
  for (const t of o.y.ticks) s += `<g class="grid"><line x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}"/></g><text x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${o.y.fmt(t)}</text>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="lbl">${esc(o.x.label)}</text>`;
  s += `<text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle" class="lbl">${esc(o.y.label)}</text>`;
  s += `<g clip-path="url(#${id})">`;
  for (const L of o.lines || []) {
    const d = L.pts.map(([a, b], i) => `${i ? 'L' : 'M'}${sx(a).toFixed(1)},${sy(b).toFixed(1)}`).join('');
    s += `<path d="${d}" fill="none" stroke="${L.color}" stroke-width="${L.width || 2}" ${L.dash ? `stroke-dasharray="${L.dash}"` : ''} opacity="${L.opacity || 1}"/>`;
  }
  for (const S of o.series || []) {
    for (const p of S.points) {
      if (p.x < x0 || p.x > x1 || p.y < y0 || p.y > y1) continue;
      s += `<circle class="pt" cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="${(p.r || S.r || 4).toFixed(1)}" fill="${S.color}" fill-opacity="${S.opacity || .7}" ${p.tip ? `data-tip="${esc(p.tip)}"` : ''}/>`;
    }
  }
  s += `</g>`;
  for (const L of o.labels || []) s += `<text x="${sx(L.x) + (L.dx || 0)}" y="${sy(L.y) + (L.dy || 0)}" text-anchor="${L.anchor || 'start'}" class="lbl" style="fill:${L.color || 'var(--ink)'}">${esc(L.text)}</text>`;
  s += `</svg>`;
  el.innerHTML = (o.legend ? legend(o.legend) : '') + s;
}

function legend(items) {
  return `<div class="legend">${items.map(i => `<span><i style="background:${i.color}"></i>${esc(i.label)}</span>`).join('')}</div>`;
}

function dots(el, o) {
  const W = o.w || 520, rowH = 44, m = { l: 92, r: 18, t: 18, b: 40 };
  const H = m.t + m.b + rowH * o.rows.length;
  const [x0, x1] = o.x.domain;
  const sx = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.label)}">`;
  for (const t of o.x.ticks) s += `<g class="grid"><line x1="${sx(t)}" x2="${sx(t)}" y1="${m.t - 6}" y2="${H - m.b}"/></g><text x="${sx(t)}" y="${H - m.b + 17}" text-anchor="middle">${o.x.fmt(t)}</text>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" class="lbl">${esc(o.x.label)}</text>`;
  o.rows.forEach((row, i) => {
    const y = m.t + rowH * i + rowH / 2;
    const vs = row.values.map(v => v.v);
    s += `<text x="${m.l - 12}" y="${y + 4}" text-anchor="end" class="lbl">${esc(row.label)}</text>`;
    s += `<line x1="${sx(Math.min(...vs))}" x2="${sx(Math.max(...vs))}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-width="6" stroke-linecap="round"/>`;
    [...row.values].sort((a, b) => a.v - b.v).forEach((v, j) => {
      s += `<circle class="pt" cx="${sx(v.v)}" cy="${y}" r="7" fill="${v.color}" data-tip="${esc(v.tip)}"/>`;
      s += `<text x="${sx(v.v)}" y="${j % 2 ? y + 22 : y - 12}" text-anchor="middle" style="font-size:11px">${(v.v * 100).toFixed(1)}</text>`;
    });
  });
  s += `</svg>`;
  el.innerHTML = (o.legend ? legend(o.legend) : '') + s;
}

const fmtPct = d => v => (v * 100).toFixed(d) + '%';
const fmtPts = v => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v * 100).toFixed(0);
const range = (a, b, step) => { const out = []; for (let v = a; v <= b + 1e-9; v += step) out.push(+v.toFixed(6)); return out; };

function drawCharts(F) {
  const atp = css('--atp'), wta = css('--wta'), ink = css('--ink');
  const tours = [['ATP', atp], ['WTA', wta]];

  // 01 reliability of raw T-vs-wide splits
  const D = 0.3;
  plot($('#chart-reliab'), {
    label: 'Past T minus wide win rate vs the same gap in 2025-26',
    x: { domain: [-D, D], ticks: range(-0.3, 0.3, 0.1), fmt: fmtPts, label: 'T minus wide, through 2024 (points per 100)' },
    y: { domain: [-D, D], ticks: range(-0.3, 0.3, 0.1), fmt: fmtPts, label: 'T minus wide, 2025–26' },
    lines: [{ pts: [[-D, -D], [D, D]], color: ink, dash: '6 5', width: 1.5, opacity: .6 },
      ...tours.map(([t, c]) => { const dc = F.validation[t].direction_check; return { pts: [[-D, dc.raw_intercept - D * dc.raw_slope], [D, dc.raw_intercept + D * dc.raw_slope]], color: c, width: 3 }; })],
    series: tours.map(([t, c]) => ({ color: c, opacity: .55, points: F.validation[t].direction_check.scatter.map(p => ({ x: p.raw, y: p.obs, r: 2.5 + Math.sqrt(p.w) / 7, tip: `${p.player}, ${p.side} court: ${fmtPts(p.raw)} before → ${fmtPts(p.obs)} in 2025–26` })) })),
    labels: [{ x: 0.12, y: 0.27, text: 'if splits were all signal', color: ink, anchor: 'end' }],
    legend: [...tours.map(([t, c]) => ({ label: `${t} (slope ${F.validation[t].direction_check.raw_slope.toFixed(2)})`, color: c }))],
  });

  // 02 baseline by aim
  const cell = (t, sd, a) => F.baselines.cells.find(c => c.tour === t && c.serve === 1 && c.side === sd && c.aim === a);
  dots($('#chart-base'), {
    label: 'First-serve points won by target',
    x: { domain: [0.5, 0.7], ticks: range(0.5, 0.7, 0.05), fmt: fmtPct(0), label: 'Points won by the server' },
    rows: ['ATP', 'WTA'].flatMap(t => SIDES.map(sd => ({
      label: `${t} · ${sd === 'deuce' ? 'Deuce' : 'Ad'}`,
      values: AIMS.map(a => { const c = cell(t, sd, a); return { v: c.won, color: css('--' + a), tip: `${t} ${sd} court, ${AIM_LABEL[a]}: ${pct(c.won)} won over ${c.n.toLocaleString()} first serves` }; }),
    }))),
    legend: AIMS.map(a => ({ label: AIM_LABEL[a], color: css('--' + a) })),
  });

  // 03 switching: each server's z-score against the bell curve a random server would produce
  const zLo = -10, zHi = 5, zStep = 0.5;
  const zBins = range(zLo, zHi - zStep, zStep);
  const zLine = t => {
    const zs = F.serial_players.filter(p => p.tour === t).map(p => Math.max(zLo, Math.min(zHi - 1e-9, p.z)));
    return zBins.flatMap(b => { const d = zs.filter(z => z >= b && z < b + zStep).length / zs.length / zStep; return [[b, d], [b + zStep, d]]; });
  };
  const normal = range(-4, 4, 0.1).map(z => [z, Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI)]);
  plot($('#chart-switch'), {
    label: 'Distribution of server switching scores vs a random server',
    x: { domain: [zLo, zHi], ticks: range(-10, 4, 2), fmt: v => v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0', label: 'Switching score (z): left = switches more than chance' },
    y: { domain: [0, 0.45], ticks: range(0, 0.4, 0.1), fmt: v => v.toFixed(1), label: 'Share of servers (density)' },
    lines: [{ pts: normal, color: ink, dash: '6 5', width: 1.5, opacity: .7 }, ...tours.map(([t, c]) => ({ pts: zLine(t), color: c, width: 2.5 }))],
    labels: [{ x: 0.6, y: 0.41, text: 'random server', color: ink }],
    legend: tours.map(([t, c]) => ({ label: `${t} (${F.serial_players.filter(p => p.tour === t).length} servers)`, color: c })),
  });

  // 04 pressure
  dots($('#chart-pressure'), {
    label: 'Points won on break points vs other points',
    x: { domain: [0.5, 0.7], ticks: range(0.5, 0.7, 0.05), fmt: fmtPct(0), label: 'First-serve points won by the server' },
    rows: F.pressure.flatMap(p => AIMS.map(a => ({
      label: `${p.tour} · ${AIM_LABEL[a]}`,
      values: [{ v: p.won_normal[a], color: css('--atp'), tip: `${p.tour} ${AIM_LABEL[a]}, other points: ${pct(p.won_normal[a])}` },
        { v: p.won_bp[a], color: css('--wta'), tip: `${p.tour} ${AIM_LABEL[a]}, break points: ${pct(p.won_bp[a])}` }],
    }))),
    legend: [{ label: 'Other points', color: css('--atp') }, { label: 'Break points', color: css('--wta') }],
  });

  // 05 second serve
  const best = new Set(F.two_first.flatMap(x => x.top.map(p => p.player)));
  const labels = F.two_first.flatMap(x => x.top.slice(0, 2)).map(p => ({ x: p.safe, y: p.risky, text: p.player, dx: 8, dy: -6 }));
  plot($('#chart-second'), {
    label: 'Second-serve win rate vs hitting another first serve',
    x: { domain: [0.3, 0.62], ticks: range(0.3, 0.6, 0.05), fmt: fmtPct(0), label: 'Points won with their real second serve' },
    y: { domain: [0.3, 0.62], ticks: range(0.3, 0.6, 0.05), fmt: fmtPct(0), label: 'Points won with another first serve' },
    lines: [{ pts: [[0.3, 0.3], [0.62, 0.62]], color: ink, dash: '6 5', width: 1.5, opacity: .6 }],
    series: tours.map(([t, c]) => ({ color: c, opacity: .6, r: 3.6, points: F.two_first.find(x => x.tour === t).points.map(p => ({ x: p.safe, y: p.risky, r: best.has(p.player) ? 6 : 3.6, tip: `${p.player}: second serve ${pct(p.safe)}, another first serve ${pct(p.risky)}` })) })),
    labels: [...labels, { x: 0.6, y: 0.585, text: 'break-even', anchor: 'end', color: ink }],
    legend: tours.map(([t, c]) => ({ label: t, color: c })),
  });

  // calibration
  plot($('#chart-cal'), {
    label: 'Forecast vs actual on 2025-26 matches', h: 320,
    x: { domain: [0.38, 0.72], ticks: range(0.4, 0.7, 0.1), fmt: fmtPct(0), label: 'Forecast chance server wins the point' },
    y: { domain: [0.38, 0.72], ticks: range(0.4, 0.7, 0.1), fmt: fmtPct(0), label: 'Actual' },
    lines: [{ pts: [[0.38, 0.38], [0.72, 0.72]], color: ink, dash: '6 5', width: 1.5, opacity: .6 },
      ...tours.map(([t, c]) => ({ pts: F.validation[t].calibration.map(b => [b.pred, b.obs]), color: c, width: 2 }))],
    series: tours.map(([t, c]) => ({ color: c, opacity: .95, r: 4.5, points: F.validation[t].calibration.map(b => ({ x: b.pred, y: b.obs, tip: `${t}: forecast ${pct(b.pred)}, actual ${pct(b.obs)} (${b.n.toLocaleString()} serves)` })) })),
    legend: tours.map(([t, c]) => ({ label: t, color: c })),
  });
}

// ------------------------------------------------------------------ matchup model

function eta(M, S, R, serve, side, aim, surf) {
  const c = serve * 6 + side * 3 + aim;
  let e = M.base[c * 4 + surf], v = M.base_var[c * 4 + surf];
  for (const X of [S && S.srv, R && R.ret]) {
    if (!X) continue;
    e += X.u + X.s[serve] + X.v[c];
    v += X.uv + X.sv[serve] + X.vv[c];
  }
  return [e, v];
}

function tourMix(tour, serve, side) {
  const cells = state.findings.baselines.cells.filter(c => c.tour === tour.toUpperCase() && c.serve === serve + 1 && c.side === SIDES[side]);
  const n = AIMS.map(a => cells.find(c => c.aim === a).n);
  return n.map(x => x / sum(n));
}

function playerMix(P, serve, side, tour) {
  const n = AIMS.map((_, a) => P.mix[serve * 6 + side * 3 + a]);
  return sum(n) >= 30 ? n.map(x => x / sum(n)) : tourMix(tour, serve, side);
}

function servePoint(M, S, R, serve, surf, mixOf) {
  // Chance the server wins a point on this serve, averaged over both courts with the given mix.
  let p = 0;
  for (let side = 0; side < 2; side++) {
    const mix = mixOf(serve, side);
    for (let a = 0; a < 3; a++) p += 0.5 * mix[a] * sigm(eta(M, S, R, serve, side, a, surf)[0]);
  }
  return p;
}

function hold(p) {
  const q = 1 - p;
  return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * p * p / (1 - 2 * p * q);
}

async function renderReport() {
  const tour = state.tour;
  const M = await model(tour);
  const sName = $('#server').value.trim(), rName = $('#returner').value.trim();
  const S = M.players[sName], R = M.players[rName];
  const err = $('#err');
  if (!S || !R || sName === rName) {
    err.hidden = false;
    err.textContent = !S ? `Pick a server from the list (${Object.keys(M.players).length} ${tour.toUpperCase()} players with 1,000+ charted serves and returns).`
      : !R ? 'Pick a returner from the list.' : 'Pick two different players.';
    return;
  }
  err.hidden = true;
  const surfName = $('#surface').value;
  const surf = SURFACES.indexOf(surfName);
  const params = new URLSearchParams({ tour, server: sName, returner: rName, surface: surfName });
  history.replaceState(null, '', `?${params}${location.hash}`);
  setKeys({ srv_name: sName });

  const mixS = (serve, side) => playerMix(S, serve, side, tour);
  const mixT = (serve, side) => tourMix(tour, serve, side);
  const spw = servePoint(M, S, R, 0, surf, mixS);
  const spwAvgRet = servePoint(M, S, null, 0, surf, mixS);
  const spwTour = servePoint(M, null, null, 0, surf, mixT);
  const sec = servePoint(M, S, R, 1, surf, mixS);
  const secTour = servePoint(M, null, null, 1, surf, mixT);
  const seEta = Math.sqrt(S.srv.uv + S.srv.sv[0] + R.ret.uv + R.ret.sv[0]);
  const band = Z90 * seEta * spw * (1 - spw);
  const kpi = (v, l, d, cls = '') => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div><div class="d ${cls}">${d}</div></div>`;
  const retEffect = spw - spwAvgRet;
  $('#kpis').innerHTML =
    kpi(`${pct(spw)}<small>±${(band * 100).toFixed(1)}</small>`, 'Service points won', `tour average ${pct(spwTour)}`) +
    kpi(pct(hold(spw), 0), 'Service games held', `tour average ${pct(hold(spwTour), 0)}`) +
    kpi(pct(sec), 'Second-serve points won', `tour average ${pct(secTour)}`) +
    kpi(`${pts(retEffect)}<small>pts</small>`, `${esc(rName)}'s return effect`, retEffect < 0 ? 'tougher than an average returner' : 'easier than an average returner', retEffect < 0 ? 'down' : 'up');
  $('#forecast-note').textContent = `Forecast for ${sName} serving to ${rName} on ${surfName.toLowerCase()}, using ${sName}'s own first- and second-serve mix. ` +
    `Based on ${S.n_srv.toLocaleString()} charted serves by ${sName} and ${R.n_ret.toLocaleString()} returns by ${rName}.`;

  renderCourt(M, S, R, surf, surfName, sName);
  renderReads(S, sName);
  renderPressure(S, sName);
  renderSecond(S, sName);
}

function renderCourt(M, S, R, surf, surfName, sName) {
  const lanes = { 0: [0, 1, 2], 1: [2, 1, 0] };  // deuce box: wide..T left to right; ad box: T..wide
  let html = '', verdicts = '';
  for (let side = 0; side < 2; side++) {
    const mix = playerMix(S, 0, side, state.tour);
    const est = AIMS.map((_, a) => { const [e, v] = eta(M, S, R, 0, side, a, surf); const p = sigm(e); return { p, band: Z90 * Math.sqrt(v) * p * (1 - p) }; });
    const raw = AIMS.map((_, a) => { const n = S.mix[side * 3 + a]; return { n, p: n ? S.wins[side * 3 + a] / n : null }; });
    // T vs wide: the directional terms are all that differ, so only their variance counts.
    const cT = side * 3 + 2, cW = side * 3;
    const varDiff = M.base_var[cT * 4 + surf] + M.base_var[cW * 4 + surf] + S.srv.vv[cT] + S.srv.vv[cW] + R.ret.vv[cT] + R.ret.vv[cW];
    const pbar = (est[2].p + est[0].p) / 2;
    const diff = est[2].p - est[0].p;
    const seDiff = Math.sqrt(varDiff) * pbar * (1 - pbar);
    const credible = Math.abs(diff) > Z90 * seDiff;
    const better = diff > 0 ? 2 : 0;
    const rawDiff = raw[2].p != null && raw[0].p != null ? raw[2].p - raw[0].p : null;
    html += `<div class="box"><div class="boxname">${side ? 'Ad court' : 'Deuce court'}</div>`;
    for (const a of lanes[side]) {
      const r = raw[a];
      html += `<div class="lane ${credible && a === better ? 'best' : ''}"><span class="aim">${AIM_LABEL[AIMS[a]]}</span>` +
        `<span class="p">${pct(est[a].p, 0)}</span><span class="pm">±${(est[a].band * 100).toFixed(1)}</span>` +
        `<span class="raw">raw vs all: ${r.p == null ? '–' : pct(r.p, 0)}<br>n ${r.n.toLocaleString()}</span>` +
        `<span class="use"><i style="width:${(mix[a] * 100).toFixed(0)}%"></i></span><span class="uset">${pct(mix[a], 0)} of serves</span></div>`;
    }
    html += `</div>`;
    const perMatch = S.per_match[side];
    let v;
    if (credible) {
      const gain = 0.2 * Math.abs(diff) * perMatch;
      v = `<b>Lean ${AIM_LABEL[AIMS[better]]}: ${pts(Math.abs(diff))} points per 100 first serves</b>Moving 1 in 5 serves from ${AIM_LABEL[AIMS[2 - better]]} to ${AIM_LABEL[AIMS[better]]} is worth about ${gain.toFixed(1)} points a match (${perMatch.toFixed(0)} first serves to this court per match).`;
    } else {
      v = `<b>No reliable T-vs-wide edge (${pts(diff)} ± ${(Z90 * seDiff * 100).toFixed(1)})</b>Keep mixing and make the returner guess.`;
    }
    if (rawDiff != null && Math.abs(rawDiff) > 0.03 && Math.abs(rawDiff) > 2 * Math.abs(diff)) {
      v += ` The raw split says ${rawDiff > 0 ? 'T' : 'wide'} by ${Math.abs(rawDiff * 100).toFixed(1)} points. Most of that is noise.`;
    }
    const bodyGap = (est[0].p + est[2].p) / 2 - est[1].p;
    if (mix[1] > 0.03 && bodyGap > 0.015) v += ` Body serves trail by ${(bodyGap * 100).toFixed(1)}.`;
    verdicts += `<div class="verdict"><span class="small">${side ? 'Ad' : 'Deuce'} court</span>${v}</div>`;
  }
  const court = $('#court');
  court.className = `court ${surfName}`;
  court.innerHTML = html;
  $('#verdicts').innerHTML = verdicts;
}

function renderReads(S, sName) {
  let html = '', bestShift = null;
  for (let side = 0; side < 2; side++) {
    const marg = AIMS.map((_, a) => S.mix[side * 3 + a]);
    const usual = marg.map(x => x / sum(marg));
    let rows = '';
    for (let prev = 0; prev < 3; prev++) {
      const cnt = AIMS.map((_, a) => S.trans[side * 9 + prev * 3 + a]);
      const n = sum(cnt);
      if (n < 30) { rows += `<tr><td>After ${AIM_LABEL[AIMS[prev]]}</td><td colspan="3" class="small">too few (${n})</td></tr>`; continue; }
      const cond = cnt.map((c, a) => (c + SHRINK_K * usual[a]) / (n + SHRINK_K));
      rows += `<tr><td>After ${AIM_LABEL[AIMS[prev]]} <span class="small">(${n})</span></td>` + cond.map((p, a) => {
        const shift = p - usual[a];
        if (!bestShift || Math.abs(shift) > Math.abs(bestShift.shift)) bestShift = { side, prev, a, p, u: usual[a], shift };
        return `<td class="${Math.abs(shift) >= 0.04 ? 'hot' : ''}">${pct(p, 0)}</td>`;
      }).join('') + `</tr>`;
    }
    html += `<table class="read"><caption>${side ? 'Ad' : 'Deuce'} court</caption><thead><tr><th>Last serve</th>${AIMS.map(a => `<th>${AIM_LABEL[a]}</th>`).join('')}</tr></thead><tbody>${rows}` +
      `<tr class="usual"><td>Usual mix</td>${usual.map(p => `<td>${pct(p, 0)}</td>`).join('')}</tr></tbody></table>`;
  }
  $('#reads').innerHTML = html;
  const sw = S.switch;
  let note = '';
  if (bestShift && Math.abs(bestShift.shift) >= 0.02) {
    note += `Biggest read: after a ${AIM_LABEL[AIMS[bestShift.prev]]} serve to the ${SIDES[bestShift.side]} court, ${sName} goes ${AIM_LABEL[AIMS[bestShift.a]]} ${pct(bestShift.p, 0)} of the time (usually ${pct(bestShift.u, 0)}). `;
  }
  if (sw) {
    note += Math.abs(sw.z) >= 1.96
      ? `${sName} repeats the last spot ${pct(sw.obs, 0)} of the time vs ${pct(sw.exp, 0)} for a random server with the same mix (z = ${sw.z}): ${sw.z < 0 ? 'switches' : 'repeats'} more than chance.`
      : `${sName}'s repeat rate (${pct(sw.obs, 0)} vs ${pct(sw.exp, 0)} at random) is within chance: no reliable switching tell.`;
  }
  $('#switch-note').textContent = note;
}

function mixBar(shares) {
  return `<div class="mixbar">${AIMS.map((a, i) => `<span style="width:${(shares[i] * 100).toFixed(1)}%;background:var(--${a})" title="${AIM_LABEL[a]} ${pct(shares[i], 0)}">${shares[i] >= 0.12 ? `${AIM_LABEL[a]} ${pct(shares[i], 0)}` : ''}</span>`).join('')}</div>`;
}

function renderPressure(S, sName) {
  let html = legend(AIMS.map(a => ({ label: AIM_LABEL[a], color: `var(--${a})` })));
  for (let side = 0; side < 2; side++) {
    const bp = AIMS.map((_, a) => S.bp[side * 3 + a]);
    const all = AIMS.map((_, a) => S.mix[side * 3 + a]);
    const normal = all.map((x, a) => x - bp[a]);
    const nb = sum(bp);
    html += `<p class="small" style="margin:8px 0 2px;font-weight:700">${side ? 'Ad' : 'Deuce'} court</p>`;
    html += `<div class="mixrow"><span class="lab">Other points</span>${mixBar(normal.map(x => x / sum(normal)))}</div>`;
    html += nb >= 15
      ? `<div class="mixrow"><span class="lab">Break points (${nb})</span>${mixBar(bp.map(x => x / nb))}</div>`
      : `<div class="mixrow"><span class="lab">Break points</span><span class="small">only ${nb} charted</span></div>`;
  }
  $('#pressure').innerHTML = html;
}

function renderSecond(S, sName) {
  const s = S.second;
  if (!s) { $('#second').innerHTML = `<p class="small">Fewer than 400 charted second serves for ${esc(sName)}.</p>`; return; }
  const d = s.risky - s.safe;
  const sigGain = d > 1.96 * s.se;
  $('#second').innerHTML = `<div class="kpis" style="grid-template-columns:1fr 1fr">` +
    `<div class="kpi"><div class="v">${pct(s.safe)}</div><div class="l">wins with their real second serve</div></div>` +
    `<div class="kpi"><div class="v">${pct(s.risky)}</div><div class="l">would win with another first serve</div></div></div>` +
    `<p class="note">${sigGain ? `Going big would gain about ${(d * 100).toFixed(1)} points per 100 second serves. One of the rare players where it's worth testing.`
      : d > 0 ? `Slightly better on paper (${pts(d)}), but within noise. Keep it safe.` : `Keep it safe: going big would cost about ${(-d * 100).toFixed(1)} points per 100 second serves.`}</p>`;
}

function bindRepeaters(atp, wta) {
  // Do servers who dodge the switching habit serve better? Correlate excess repeat rate with first-serve strength.
  const corr = M => {
    const xs = [], ys = [];
    for (const p of Object.values(M.players)) {
      if (!p.switch || !p.srv || p.switch.pairs < 1000) continue;
      xs.push(p.switch.obs - p.switch.exp);
      ys.push(p.srv.u + p.srv.s[0]);
    }
    const mx = sum(xs) / xs.length, my = sum(ys) / ys.length;
    let sxy = 0, sxx = 0, syy = 0;
    xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; syy += (ys[i] - my) ** 2; });
    return { r: sxy / Math.sqrt(sxx * syy), n: xs.length };
  };
  const a = corr(atp), w = corr(wta);
  const top = state.findings.serial_players.filter(p => p.tour === 'ATP').sort((x, y) => y.z - x.z).slice(0, 4).map(p => p.player);
  setKeys({ repeaters: top.join(', '), atp_rep_r: a.r.toFixed(2), atp_rep_n: String(a.n), wta_rep_r: w.r.toFixed(2) });
}

// ------------------------------------------------------------------ controls

async function setTour(tour, keepNames) {
  state.tour = tour;
  const M = await model(tour);
  const names = Object.entries(M.players).sort((a, b) => b[1].matches - a[1].matches);
  $('#players').innerHTML = names.map(([n, p]) => `<option value="${esc(n)}">${p.matches} matches · last ${p.last.slice(0, 4)}</option>`).join('');
  if (!keepNames) {
    const [a, b] = DEFAULTS[tour].every(n => M.players[n]) ? DEFAULTS[tour] : [names[0][0], names[1][0]];
    $('#server').value = a;
    $('#returner').value = b;
  }
  await renderReport();
}

function setupTooltips() {
  const tip = $('#tip');
  document.addEventListener('pointerover', e => {
    const t = e.target.closest('[data-tip]');
    if (!t) return;
    tip.textContent = t.dataset.tip;
    tip.hidden = false;
  });
  document.addEventListener('pointermove', e => {
    if (tip.hidden) return;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = `${x}px`;
    tip.style.top = `${e.clientY + 14}px`;
  });
  document.addEventListener('pointerout', e => { if (e.target.closest('[data-tip]')) tip.hidden = true; });
}

async function init() {
  setupTooltips();
  state.findings = await getJSON('data/findings.json');
  bindStory(state.findings);
  drawCharts(state.findings);

  const q = new URLSearchParams(location.search);
  const tour = q.get('tour') === 'wta' ? 'wta' : 'atp';
  document.querySelector(`input[name=tour][value=${tour}]`).checked = true;
  if (q.get('surface') && SURFACES.includes(q.get('surface'))) $('#surface').value = q.get('surface');
  if (q.get('server') && q.get('returner')) {
    $('#server').value = q.get('server');
    $('#returner').value = q.get('returner');
    await setTour(tour, true);
  } else {
    await setTour(tour, false);
  }

  bindRepeaters(await model('atp'), await model('wta'));

  document.querySelectorAll('input[name=tour]').forEach(r => r.addEventListener('change', e => setTour(e.target.value, false)));
  for (const id of ['server', 'returner']) $('#' + id).addEventListener('change', renderReport);
  $('#surface').addEventListener('change', renderReport);
  $('#controls').addEventListener('submit', e => { e.preventDefault(); renderReport(); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => drawCharts(state.findings));
}

init().catch(e => {
  console.error(e);
  const err = $('#err');
  err.hidden = false;
  err.textContent = `Could not load the data (${e.message}). If you opened index.html straight from disk, serve the folder instead: python3 -m http.server -d site`;
});
