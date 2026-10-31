# Devpost submission text: ServeTell

**Tagline (≤200 chars):** Which tennis serve patterns are real? We tested 2.56M charted pro serves against game theory and built a scouting report that only shows the edges that hold up out of sample.

**Links**
- Live demo: https://ashank-crazy.github.io/servetell/
- Source: https://github.com/ashank-crazy/servetell

---

## Inspiration

Every tennis broadcast shows serve-direction splits: "she wins 71% of points going T on the deuce court." Coaches
build game plans on numbers like these. But a split from a few matches is mostly luck, and no one tells you how much.
Game theory (minimax) even predicts that pros should already be balanced, so no target pays more than another. We
wanted to know which serve patterns are real, and to give coaches a tool that separates signal from noise.

## What it does

ServeTell has two parts.

**1. Five findings from 11,785 shot-by-shot charted pro matches (2.56M serves).**

1. *A player's "best spot" is mostly noise.* Only ~15% of an ATP player's T-vs-wide split carries into the next
   season, and ~0% for WTA. Simply assuming "no difference" forecast 2025–26 better than the raw splits did. Within
   players, T vs. wide tests flag 7.4% (ATP) and 5.7% (WTA) of 1,211 splits at the 5% level, about what chance alone
   produces. None survive multiple-testing correction. Pros sit at the minimax balance. After shrinkage, only 11 of
   896 player-court combinations show a reliable edge, each about 1 point per 100.
2. *Body first serves really do lose:* about 4.8 fewer points per 100 than T or wide, on both tours and both courts.
3. *The real tell is timing.* 74% of WTA and 52% of ATP servers switch targets significantly more often than a random
   mixer would. Out of sample, tracking the last serve improves a returner's guess (49.9% → 50.6% ATP). The ATP's
   biggest repeaters are Djokovic, Federer, Opelka and Kuerten.
4. *Break points don't change the mix.* There is no tour-wide pressure tell (51.2% favorite-spot rate vs. 51.4%
   expected), but servers win ~3 fewer points per 100 on break points whichever way they aim.
5. *Keep the second serve safe.* Only 13 of 351 regular servers would win more by hitting another first serve. Only
   Alexander Zverev clears statistical significance.

**2. An interactive matchup scouting report.** Pick a server, a returner and a surface:

- Forecast service points won, hold %, second-serve points won, and how many points this returner costs the server.
- A court diagram from the server's side: the model's win chance for each target (with a 90% range) next to the raw
  split a broadcast would show, plus a verdict. Either "Lean T, worth X points a match" or "No reliable edge, keep
  mixing", with a note when the raw split is mostly noise.
- The returner's read: where the next serve goes after each previous target, and the server's personal switching score.
- Break-point mix vs. normal points, and a second-serve check.

## Actionable impact

- **Coaches:** stop rebuilding serve plans around small-sample splits. ServeTell shows which edges survive shrinkage
  and how many points per match they are worth.
- **Returners:** after a serve to one spot, shade toward the other. Most pros switch more than chance, and the tell
  holds up on unseen 2025–26 matches. The scout shows each server's specific pattern.
- **Servers:** cut first serves to the body. Repeating a target is less predictable than it feels. Keep the safe
  second serve unless the second-serve check says otherwise.
- **Analysts:** a reusable, validated way to report serve stats with honest uncertainty.

## How we built it

- **Data:** the Tennis Abstract Match Charting Project (CC BY-NC-SA 4.0): 1.88M charted points from 7,532 ATP and
  4,253 WTA matches.
- **Parsing (Python/pandas):** each point becomes serve attempts with target (wide/body/T from the serve code), fault
  status, court (deuce/ad, worked out from the score), break-point flag, surface, server, returner and outcome.
- **Statistics (SciPy):**
  - Minimax chi-square screen with Benjamini–Hochberg control and a KS test of the p-value distribution.
  - Within-match permutation test (200 shuffles) for serial dependence.
  - Within-player break-point comparison.
  - Out-of-sample anticipation test.
- **Matchup model:** penalized logistic regression, the MAP estimate of a random-effects model. It has a tour baseline
  plus server and returner effects at three levels (overall, by serve, by serve × court × target), each with its own
  penalty. Penalties were tuned on 2024 matches. Target-specific effects needed the heaviest penalty, which is
  finding 1 seen from the model's side.
- **Validation:** trained through 2024 and scored on 1,683 matches from 2025–26. Log-loss 0.6606 → 0.6573 (ATP) and
  0.6838 → 0.6819 (WTA) over the tour baseline, with a well-calibrated reliability curve.
- **Front end:** a static site in plain HTML/CSS/JS with hand-drawn SVG charts. The model runs in the browser from
  ~430 KB of JSON. GitHub Pages hosts it.

## Challenges we ran into

Our first idea was a tool telling servers where to aim. The out-of-sample check killed it: raw directional splits had
almost no predictive power. Instead of hiding that, we made it the headline and rebuilt the tool around separating
signal from noise. The other challenge was the null for the switching test. Comparing against a player's career mix
would wrongly flag mix changes between matches, so we shuffled within each match and court.

## Accomplishments we're proud of

- Reproducing a classic economics result (Walker & Wooders' minimax test) on 1.3M first serves.
- Every claim on the site is computed from the data at page load, so the text can't drift from the numbers.
- Being honest about small effects and limitations instead of overselling.

## What we learned

At the pro level the serve game is close to equilibrium, so the remaining edges are small and live in timing
and habits, not in "where." Shrinkage and out-of-sample checks matter most exactly where the data looks most exciting.

## What's next

- Add return position and serve speed from tracking data, to explain why targets balance out.
- Per-match live mode: update a server's read point by point during a match.
- Joint score-state model: does the mix change at 30-30 or in tiebreaks?

## Built with

Python, pandas, NumPy, SciPy, JavaScript, SVG, HTML/CSS, GitHub Pages. Built with help from Claude Code (AI coding
assistant).

## Limitations (also on the site)

Charted matches skew toward big events. Targets are recorded where the ball lands. Direction tests are observational.
The switching test counts in-match plan changes as repeats. The two-first-serves check assumes first-serve accuracy
holds under second-serve pressure.
