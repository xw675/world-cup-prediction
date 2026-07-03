"""
World Cup match prediction — full pipeline.

Steps (matching our original roadmap):
  1. Data prep: load, drop unplayed fixtures, align team names
  2. Feature engineering from scratch: Elo ratings + exponential moving averages
  3. Chronological split (train < 2018, test >= 2018) — no leakage
  4. Models: Logistic Regression baseline -> HistGradientBoosting, calibrated
  5. Evaluation: LogLoss, Brier score, feature importances
  6. Predict the unplayed 2026 World Cup fixtures

Run:  python3 build_model.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.inspection import permutation_importance

RNG = 42

# ----------------------------------------------------------------------
# 1. DATA PREPARATION
# ----------------------------------------------------------------------
df = pd.read_csv("datasets/results.csv", parse_dates=["date"])

# Align team names: map former names -> current names so Elo history
# doesn't fracture (e.g. Zaire -> DR Congo).
fn = pd.read_csv("datasets/former_names.csv")
name_map = dict(zip(fn["former"], fn["current"]))
# Deliberate exception: West Germany is mapped to Germany by the file
# (correct — the DFB is the same federation). East Germany is NOT in the
# file and stays separate (correct — different federation, absorbed 1990).
for col in ["home_team", "away_team"]:
    df[col] = df[col].replace(name_map)

upcoming = df[df["home_score"].isna()].copy()          # 2026 WC fixtures to predict
played = df[df["home_score"].notna()].copy()
played[["home_score", "away_score"]] = played[["home_score", "away_score"]].astype(int)
played = played.sort_values("date").reset_index(drop=True)

# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING FROM SCRATCH
# ----------------------------------------------------------------------
# ---- 2a. Elo ratings ----
# Classic football Elo: R' = R + K * G * (result - expected)
#   expected = 1 / (1 + 10^((R_opp - R_own + home_adv) / 400))
#   K depends on match importance, G scales with goal difference.
BASE_ELO = 1500.0
HOME_ADV = 80.0  # Elo points added to the home side when not neutral

K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50, "Copa América": 50, "African Cup of Nations": 50,
    "AFC Asian Cup": 50, "CONCACAF Championship": 50, "Gold Cup": 50,
    "UEFA Nations League": 40, "Confederations Cup": 40,
    "Friendly": 20,
}
DEFAULT_K = 30  # other qualifiers / regional cups


def k_factor(tournament: str) -> float:
    return K_BY_TOURNAMENT.get(tournament, DEFAULT_K)


def goal_multiplier(goal_diff: int) -> float:
    """Bigger wins move ratings more, with diminishing returns."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8.0


# ---- 2b. Exponential moving averages of goals scored / conceded ----
# EMA update after each match: ema = alpha * x + (1 - alpha) * ema
# alpha = 0.1 ~= "effective memory" of the last ~20 matches.
EMA_ALPHA = 0.10
EMA_INIT_FOR, EMA_INIT_AGAINST = 1.3, 1.3  # ~historic mean goals per team

elo: dict[str, float] = {}
ema_for: dict[str, float] = {}
ema_against: dict[str, float] = {}
matches_played: dict[str, int] = {}


def get(team, store, default):
    return store.get(team, default)


def snapshot_features(row):
    """PRE-match state for both teams. Called before ratings are updated,
    which is what prevents target leakage: the model only ever sees what
    was knowable before kickoff."""
    h, a = row.home_team, row.away_team
    return {
        "home_elo": get(h, elo, BASE_ELO),
        "away_elo": get(a, elo, BASE_ELO),
        "home_ema_gf": get(h, ema_for, EMA_INIT_FOR),
        "home_ema_ga": get(h, ema_against, EMA_INIT_AGAINST),
        "away_ema_gf": get(a, ema_for, EMA_INIT_FOR),
        "away_ema_ga": get(a, ema_against, EMA_INIT_AGAINST),
        "home_matches": get(h, matches_played, 0),
        "away_matches": get(a, matches_played, 0),
    }


def update_state(row):
    """POST-match: update Elo and EMAs with the observed result."""
    h, a = row.home_team, row.away_team
    hs, as_ = row.home_score, row.away_score
    r_h, r_a = get(h, elo, BASE_ELO), get(a, elo, BASE_ELO)

    adv = 0.0 if row.neutral else HOME_ADV
    exp_h = 1.0 / (1.0 + 10 ** ((r_a - (r_h + adv)) / 400.0))
    result_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)

    k = k_factor(row.tournament)
    g = goal_multiplier(abs(hs - as_))
    delta = k * g * (result_h - exp_h)
    elo[h], elo[a] = r_h + delta, r_a - delta

    for team, gf, ga in [(h, hs, as_), (a, as_, hs)]:
        ema_for[team] = EMA_ALPHA * gf + (1 - EMA_ALPHA) * get(team, ema_for, EMA_INIT_FOR)
        ema_against[team] = EMA_ALPHA * ga + (1 - EMA_ALPHA) * get(team, ema_against, EMA_INIT_AGAINST)
        matches_played[team] = get(team, matches_played, 0) + 1


# Single chronological pass over ALL played matches (full history feeds
# the ratings, even though we only train on modern rows).
feature_rows = []
for row in played.itertuples(index=False):
    feature_rows.append(snapshot_features(row))
    update_state(row)

features = pd.DataFrame(feature_rows)
data = pd.concat([played.reset_index(drop=True), features], axis=1)

# Derived (difference) features — models learn matchups more easily from
# relative strength than from two absolute numbers.
data["neutral_flag"] = data["neutral"].astype(int)
data["elo_diff"] = data["home_elo"] - data["away_elo"]
# Home attack vs away defence, and vice versa:
data["home_attack_edge"] = data["home_ema_gf"] - data["away_ema_ga"]
data["away_attack_edge"] = data["away_ema_gf"] - data["home_ema_ga"]

# Target: 0 = home win, 1 = draw, 2 = away win
data["outcome"] = np.select(
    [data.home_score > data.away_score, data.home_score == data.away_score],
    [0, 1], default=2,
)

FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "home_ema_gf", "home_ema_ga", "away_ema_gf", "away_ema_ga",
    "home_attack_edge", "away_attack_edge",
    "neutral_flag",
]
CLASS_NAMES = ["home_win", "draw", "away_win"]

# ----------------------------------------------------------------------
# 3. CHRONOLOGICAL SPLIT — strictly no leakage
# ----------------------------------------------------------------------
# Train on 1990-2017, test on 2018+ (two full World Cup cycles held out).
# Pre-1990 rows still contributed to Elo/EMA history above.
modern = data[data["date"] >= "1990-01-01"].copy()
SPLIT = "2018-01-01"
train = modern[modern["date"] < SPLIT]
test = modern[modern["date"] >= SPLIT]

X_tr, y_tr = train[FEATURES], train["outcome"]
X_te, y_te = test[FEATURES], test["outcome"]
print(f"Train: {len(train):,} matches (1990–2017)   Test: {len(test):,} matches (2018+)")

# ----------------------------------------------------------------------
# 4. MODELS + CALIBRATION
# ----------------------------------------------------------------------
def brier_multiclass(y_true, proba, n_classes=3):
    onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((proba - onehot) ** 2, axis=1))


def evaluate(name, model, X, y):
    proba = model.predict_proba(X)
    print(f"{name:<28} LogLoss={log_loss(y, proba):.4f}  "
          f"Brier={brier_multiclass(y, proba):.4f}  "
          f"Acc={accuracy_score(y, proba.argmax(1)):.3f}")
    return proba


# Naive baseline: predict the training base rates for every match.
base_rates = y_tr.value_counts(normalize=True).sort_index().values
naive = np.tile(base_rates, (len(y_te), 1))
print(f"{'Naive (base rates)':<28} LogLoss={log_loss(y_te, naive):.4f}  "
      f"Brier={brier_multiclass(y_te, naive):.4f}  Acc={max(base_rates):.3f}")

# Baseline: logistic regression (scaled features).
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
logreg.fit(X_tr, y_tr)
evaluate("Logistic Regression", logreg, X_te, y_te)

# Gradient-boosted trees (sklearn's LightGBM-style implementation).
gbm = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_depth=4,
    l2_regularization=1.0, random_state=RNG,
)
gbm.fit(X_tr, y_tr)
evaluate("HistGradientBoosting", gbm, X_te, y_te)

# Calibration: isotonic, with chronological 3-fold-style refit via
# CalibratedClassifierCV on the training data.
gbm_cal = CalibratedClassifierCV(
    HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=RNG,
    ),
    method="isotonic", cv=3,
)
gbm_cal.fit(X_tr, y_tr)
proba_cal = evaluate("HistGB + isotonic calib.", gbm_cal, X_te, y_te)

# Reliability check: within probability buckets, does predicted ~= actual?
print("\nCalibration check (home-win probability, test set):")
bucket = pd.cut(proba_cal[:, 0], bins=[0, .2, .4, .6, .8, 1.0])
rel = pd.DataFrame({"pred": proba_cal[:, 0], "actual": (y_te == 0).values, "bucket": bucket})
print(rel.groupby("bucket", observed=True).agg(
    mean_pred=("pred", "mean"), actual_rate=("actual", "mean"), n=("actual", "size")
).round(3).to_string())

# ----------------------------------------------------------------------
# 5. FEATURE IMPORTANCE (permutation, on the test set)
# ----------------------------------------------------------------------
imp = permutation_importance(gbm, X_te, y_te, n_repeats=5, random_state=RNG,
                             scoring="neg_log_loss")
order = imp.importances_mean.argsort()[::-1]
print("\nPermutation feature importances (drop in LogLoss when shuffled):")
for i in order:
    print(f"  {FEATURES[i]:<18} {imp.importances_mean[i]:.4f}")

# ----------------------------------------------------------------------
# 6. PREDICT THE 2026 WORLD CUP FIXTURES
# ----------------------------------------------------------------------
# Refit the calibrated model on ALL modern data (train+test) — for real
# predictions we want every match we have. Elo/EMA dicts already contain
# the state after the last played match, i.e. exactly the pre-match state
# for the upcoming fixtures.
final_model = CalibratedClassifierCV(
    HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=RNG,
    ),
    method="isotonic", cv=3,
)
final_model.fit(modern[FEATURES], modern["outcome"])

up_rows = [snapshot_features(row) for row in upcoming.itertuples(index=False)]
up = pd.DataFrame(up_rows)
up["neutral_flag"] = upcoming["neutral"].astype(int).values
up["elo_diff"] = up["home_elo"] - up["away_elo"]
up["home_attack_edge"] = up["home_ema_gf"] - up["away_ema_ga"]
up["away_attack_edge"] = up["away_ema_gf"] - up["home_ema_ga"]

proba_up = final_model.predict_proba(up[FEATURES])
pred = upcoming[["date", "home_team", "away_team", "neutral"]].reset_index(drop=True)
pred[["p_home_win", "p_draw", "p_away_win"]] = proba_up.round(3)
pred["favourite"] = np.where(proba_up[:, 0] >= proba_up[:, 2],
                             pred["home_team"], pred["away_team"])

print("\n=== 2026 World Cup — upcoming fixture predictions ===")
print(pred.to_string(index=False))
pred.to_csv("predictions_2026.csv", index=False)

# Top-15 current Elo ratings, as a sanity check on the whole pipeline.
top = sorted(elo.items(), key=lambda kv: -kv[1])[:15]
print("\nTop 15 current Elo ratings:")
for i, (team, r) in enumerate(top, 1):
    print(f"  {i:>2}. {team:<15} {r:7.1f}")

print("\nSaved: predictions_2026.csv")
