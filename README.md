# ServeTell

**Which tennis serve patterns are signal, and which are noise.** ServeTell tests 2.56 million charted pro serves
against game theory, then turns what survives into a matchup scouting report for coaches, players and analysts.

**Live demo:** https://ashank-crazy.github.io/servetell/

Built for the AQX Sports Analytics Data Bowl 4.0.

## The problem

Broadcasts and scouting sheets love serve-direction splits: "she wins 71% going T on the deuce court." Coaches build
game plans on them. But a split from a few matches is mostly luck, and nobody tells you how much. ServeTell answers
two questions with 11,785 shot-by-shot charted matches:

1. Which serve patterns are real and persist into the future?
2. For a given server, returner and surface, what should each side actually do?

## What we found

| # | Finding | Evidence |
|---|---|---|
| 01 | **A player's "best spot" is mostly noise.** Only ~15% of an ATP player's T-vs-wide split carries into the next season, and ~0% for WTA. | Splits learned through 2024 vs. realized splits in 2025–26 (203 ATP and 179 WTA player-court cells). Slope 0.15 ± 0.09 (ATP), −0.02 ± 0.09 (WTA). Assuming "no difference" beat the raw splits on forecast error. After shrinkage, only 11 of 896 player-court combinations (hard court, average returner) show a reliable edge, each about 1 point per 100. |
| 01b | **Pros are at the game-theory balance.** T vs. wide win rates are equal within players, as minimax predicts. | 1,211 player-court chi-square tests: 7.4% (ATP) and 5.7% (WTA) significant at 5%. KS test vs. uniform p = 0.35 / 0.72. Zero survive Benjamini–Hochberg. Replicates Walker & Wooders (2001), who studied 10 Wimbledon finals, on 1.3M first serves. |
| 02 | **Body first serves lose ~4.8 more points per 100** than T or wide, on both tours and both courts. | League baselines over 2.56M serves. Partly mis-hits (charters log landing spot). |
| 03 | **The real tell is timing: pros switch targets too often.** 74% of WTA and 52% of ATP servers switch significantly more than a random mixer. | Within-match permutation test (200 shuffles). Repeat rate 35.8% vs. 40.6% expected (WTA), 43.2% vs. 44.7% (ATP). Out of sample, tracking the last serve lifts a returner's guess from 49.9% → 50.6% (ATP) and 44.3% → 44.7% (WTA). |
| 04 | **Break points don't change the mix.** Servers go to their favorite spot 51.2% of the time vs. 51.4% expected (ATP); 46.4% vs. 46.6% (WTA). Servers win ~3 fewer points per 100 on break points, whichever way they aim. | Within-player comparison vs. each server's own non-break-point mix. |
| 05 | **Keep the second serve a second serve.** Only 13 of 351 regular servers would win more by hitting another first serve. Only Alexander Zverev clears significance. | (1st-serve in %) × (win % when in) vs. second-serve win %, per player. |

## The scouting report

Pick a tour, server, returner and surface. The page computes, in the browser:

- **Serve forecast**: service points won, hold %, second-serve points won, and how much this returner costs the server
  vs. an average returner.
- **Where to aim**: a court diagram from the server's side with the model's win chance per target (±90% range), next to
  the raw split a broadcast would show. A verdict says whether any T-vs-wide edge is reliable and what shifting 1 in 5
  serves would be worth per match.
- **The returner's read**: where this server's next serve to the same court goes after each previous target, shrunk
  toward their usual mix, plus their personal switching score.
- **Under pressure**: break-point mix vs. every other point.
- **Second serve check**: real second serve vs. hitting another first serve.

Every report is shareable: the URL holds the tour, players and surface.

## How it works

```
data/raw/*.csv ──build_serves.py──▶ data/serves.pkl ──analyze.py──▶ site/data/*.json ──▶ site/ (static, no build step)
```

1. **Parsing** (`analysis/build_serves.py`): every charted point becomes one row per serve attempt. The target comes from
   the serve code (4 = wide, 5 = body, 6 = T). Faults come from whether a second serve was logged. The court
   (deuce/ad) and break-point status come from the score. 1.88M points → 2.56M serve attempts.
2. **Tests** (`analysis/analyze.py`): minimax chi-square screen with BH control, within-match permutation test for
   serial dependence, within-player pressure comparison, two-first-serves check.
3. **Matchup model**: penalized logistic regression (MAP estimate of a random-effects model) for P(server wins point).
   It has a tour baseline per serve × court × target × surface, plus server and returner effects at three levels:
   overall, by serve number, and by serve × court × target. Each level gets its own penalty. Penalties were tuned on
   2024 matches (trained through 2023) over a 12-point grid. Target-specific effects wanted the heaviest penalty
   (λ = 3,000), which is finding 01 seen from another angle. Approximate 90% ranges use the diagonal of the Laplace
   (inverse-Hessian) covariance.
4. **Validation**: trained through 2024, scored on 2025–26 (763 ATP + 920 WTA matches, 335k serves).

| Model, scored on 2025–26 | ATP log-loss | ATP Brier | WTA log-loss | WTA Brier |
|---|---|---|---|---|
| Coin flip | 0.6931 | 0.2500 | 0.6931 | 0.2500 |
| Tour baseline | 0.6606 | 0.2340 | 0.6838 | 0.2453 |
| + server strength | 0.6592 | 0.2334 | 0.6829 | 0.2449 |
| + server and returner (ServeTell) | **0.6573** | **0.2324** | **0.6819** | **0.2444** |

Forecasts are well calibrated across the full range (see the calibration chart on the site).

## Limitations

- Charted matches lean toward big events and well-known players. Players with fewer matches are pulled harder toward
  the tour norm.
- Targets are recorded where the ball lands, not where it was aimed.
- Direction tests are observational. Players choose targets using things we can't see (returner position, wind, fatigue).
- The switching test shuffles within a match, so in-match plan changes count as "repeating".
- The two-first-serves check assumes first-serve accuracy holds as a second serve. Under pressure it likely won't.

## Run it yourself

Requires Python 3.10+.

```bash
pip install -r requirements.txt
bash analysis/fetch_data.sh          # downloads the Match Charting Project CSVs (~230 MB) into data/raw/
python3 analysis/build_serves.py     # ~15 s
python3 analysis/analyze.py          # ~4 min: tests, penalty tuning, validation, exports site/data/*.json
python3 -m http.server -d site 8000  # open http://localhost:8000
```

## Credits and licenses

- **Data:** [Tennis Abstract Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject) by
  Jeff Sackmann and volunteer charters, licensed
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The derived files in `site/data/` are shared
  under the same license, for non-commercial use only. Raw data is not redistributed in this repo.
- **Method reference:** Walker, M. & Wooders, J. (2001). Minimax Play at Wimbledon. *American Economic Review* 91(5).
- **Code:** MIT (see `LICENSE`). Python (pandas, NumPy, SciPy) and plain HTML/CSS/JavaScript with hand-drawn SVG charts.
  Fonts: Barlow Condensed and Inter via Google Fonts.
- Built with help from Claude Code (AI coding assistant).
- Not affiliated with the ATP, WTA, Tennis Abstract or any player.
