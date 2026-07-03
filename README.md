# World Cup Match Prediction Model

Predicts win/draw/loss probabilities, expected goals, and most likely scorelines
for international football matches, built from scratch with pandas, NumPy, and
scikit-learn.

## How the model was built

**1. Data preparation.** Historical results (1872–present, ~49,500 matches) from the
Kaggle dataset `martj42/international-football-results-from-1872-to-2017`
(`datasets/results.csv`). Former team names are merged into current ones via
`datasets/former_names.csv` (e.g. Zaire → DR Congo) so team histories don't fracture.
Rows with empty scores are unplayed fixtures — separated out for prediction, never
used in training.

**2. Feature engineering (from scratch).** A single chronological pass over all
played matches maintains, per team:

- **Elo rating** — updated after every match: `R' = R + K·G·(result − expected)`,
  where K depends on tournament importance (World Cup 60, friendlies 20), G scales
  with goal difference, and the home side gets +80 Elo when not on neutral ground.
- **Exponential moving averages** of goals scored and conceded (α = 0.10, roughly a
  20-match memory).

Features are snapshotted **before** each match's ratings update — the model only
ever sees pre-match information (no target leakage). Final feature set (13):
elo_diff, both Elos, four goal EMAs, two attack-vs-defence edges, neutral flag,
plus three **squad market value** features (log home value, log away value, log
ratio) built from Transfermarkt player valuations: a country's squad value in a
quarter is the sum of its 25 most valuable citizens; a match uses the previous
quarter's value (strictly pre-match). Requires `players.csv` and
`player_valuations.csv` from the Kaggle dataset `davidcariboo/player-scores`
in `datasets/` (predict.py degrades gracefully without them).

**3. Chronological validation.** Train on 1990–2017 (~24,000 matches), test on
2018+ (~8,200 matches). The full pre-1990 history still feeds the Elo/EMA state.
Never shuffle-split time-series data.

**4. Models.**

- *Outcome:* HistGradientBoosting classifier with isotonic calibration
  (logistic regression baseline performs nearly identically — the features are
  the model).
- *Scoreline:* two Poisson-loss regressors predict each side's expected goals;
  an independent-Poisson score matrix with a light Dixon-Coles draw correction
  yields exact-score probabilities.

**5. Results on the 2018+ holdout.**

| Metric | Value | Naive baseline |
|---|---|---|
| LogLoss | 0.87 | 1.05 |
| Outcome accuracy | 60.2% | 48.7% |
| Exact-scoreline hit rate | 13.6% | ~5–6% |
| Calibration | predicted ≈ actual in every probability bucket | — |

Feature importance: `elo_diff` dominates (~30× everything else). Controlled
experiments (same split, same models, only the feature list changes):

- **Squad market value: +0.005–0.006 LogLoss improvement** — the best addition;
  now part of the model.
- **LogReg+HistGB ensemble (50/50): ~+0.008 vs HistGB alone** — now the
  production outcome model in predict.py.
- **Recency weighting** (10-year half-life sample weights): ~+0.001 — small but
  consistent; now part of the model.
- Elo constant tuning (K scale, home advantage, goal multiplier): the folklore
  constants were already near-optimal (+0.0001). The goal-difference multiplier
  itself is clearly valuable (removing it costs ~0.005).
- Market value top-N (11/18/25/40 players): flat within ~0.001 — kept 25.
- Rest-days/schedule features: nothing (FIFA windows synchronize rest).
- Goalscorer-concentration features: nothing (Elo already prices it in).

Scoreline-model experiments (metric: log-likelihood of the actual score under
the predicted score matrix, 2018+ holdout):

- **Outcome coupling: +0.008** — rescaling the score matrix's win/draw/loss
  regions to match the ensemble's outcome probabilities; now in predict.py.
  Scoreline and outcome predictions are now mutually consistent.
- Fitted Dixon-Coles rho (-0.06 vs hardcoded -0.10): +0.003 alone, but
  redundant once coupling is applied — removed.
- Bivariate Poisson (shared component): fitted lam3 = 0 — independence holds.
- Era goal-rate feature: nothing.

## How to predict future matches

The tool is `predict.py`. It rebuilds all ratings from the current data and
retrains on every run (takes seconds), so predictions always reflect the latest
results.

**Step 1 — update results.** After matches are played, either re-download
`results.csv` from Kaggle into `datasets/`, or edit it by hand: find the fixture
row (it has empty scores) and fill the scores in. Don't add duplicate rows.
Knockout matches: record the score after extra time; shootouts count as draws.

**Step 2 — add fixtures.** Any of:

- fixture rows already in `results.csv` with empty scores (picked up automatically);
- rows appended to `fixtures.csv`, format `date,home_team,away_team,neutral`
  — set `neutral` to `False` only if the first team is genuinely at home;
- a one-off matchup: `python3 predict.py Brazil France --neutral`

Team names must match the dataset's spelling ("United States", "South Korea").
Misspelled names trigger a warning instead of a silent bad prediction.

**Step 3 — run.**

```
python3 predict.py
```

Output: win/draw/loss probabilities, expected goals, most likely scoreline per
fixture, saved to `predictions_latest.csv`.

## Files

| File | Purpose |
|---|---|
| `predict.py` | **The tool.** Retrains and predicts fixtures. |
| `build_model.py` | Full pipeline with holdout evaluation, calibration check, feature importances. Rerun to re-verify model quality after big data updates. |
| `data_manipulation.ipynb` | Original data exploration notebook. |
| `fixtures.csv` | Your manually added upcoming fixtures. |
| `predictions_latest.csv` | Latest predictions (regenerated on every run). |
| `datasets/` | results.csv, former_names.csv, goalscorers.csv, shootouts.csv |

Requirements: `pip install pandas numpy scikit-learn scipy`
