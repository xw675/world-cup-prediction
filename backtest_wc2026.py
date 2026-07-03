"""
Out-of-sample backtest: the 2026 World Cup.

Simulates real usage honestly:
  * The model is trained ONLY on matches before the tournament
    (cutoff 2026-06-11, the Mexico vs South Africa opener).
  * Elo/EMA ratings roll forward match-by-match during the tournament
    (as they would in live use — each prediction uses only past results).
  * Market values are from the pre-tournament quarter.

Outputs per-match predictions vs reality, headline metrics, and the
biggest hits and misses. Saves backtest_wc2026.csv.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

RNG, TOP_N, MAX_GOALS = 42, 25, 8
CUTOFF = pd.Timestamp("2026-06-11")   # World Cup opening day

# ---------------- squad market values ----------------
vals = pd.read_csv("datasets/player_valuations.csv",
                   usecols=["player_id", "date", "market_value_in_eur"],
                   parse_dates=["date"])
players = pd.read_csv("datasets/players.csv",
                      usecols=["player_id", "country_of_citizenship"]).dropna()
vals["q"] = vals["date"].dt.to_period("Q")
panel = (vals.sort_values("date")
             .groupby(["player_id", "q"])["market_value_in_eur"].last()
             .unstack("q").sort_index(axis=1).ffill(axis=1, limit=8))
long = (panel.stack().rename("value").reset_index()
             .merge(players, on="player_id", how="inner"))
squad_dict = (long.groupby(["country_of_citizenship", "q"])["value"]
                  .apply(lambda s: s.nlargest(TOP_N).sum()).to_dict())

TM_NAME = {
    "South Korea": "Korea, South", "North Korea": "Korea, North",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Republic of Ireland": "Ireland", "Curaçao": "Curacao",
    "Hong Kong": "Hongkong", "Gambia": "The Gambia",
    "Brunei": "Brunei Darussalam", "South Sudan": "Southern Sudan",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Ivory Coast": "Cote d'Ivoire", "Taiwan": "Chinese Taipei",
}


def team_value(team, match_q):
    return squad_dict.get((TM_NAME.get(team, team), match_q - 1), np.nan)


# ---------------- matches + feature pipeline ----------------
df = pd.read_csv("datasets/results.csv", parse_dates=["date"])
fn = pd.read_csv("datasets/former_names.csv")
name_map = dict(zip(fn["former"], fn["current"]))
for col in ["home_team", "away_team"]:
    df[col] = df[col].replace(name_map)

played = df[df["home_score"].notna()].copy()
played[["home_score", "away_score"]] = played[["home_score", "away_score"]].astype(int)
played = played.sort_values("date").reset_index(drop=True)

BASE_ELO, HOME_ADV = 1500.0, 80.0
K_BY_TOURNAMENT = {
    "FIFA World Cup": 60, "FIFA World Cup qualification": 40,
    "UEFA Euro": 50, "Copa América": 50, "African Cup of Nations": 50,
    "AFC Asian Cup": 50, "CONCACAF Championship": 50, "Gold Cup": 50,
    "UEFA Nations League": 40, "Confederations Cup": 40, "Friendly": 20,
}
DEFAULT_K = 30
EMA_ALPHA, EMA_INIT = 0.10, 1.3
elo, ema_for, ema_against = {}, {}, {}

FEATURES = ["elo_diff", "home_elo", "away_elo",
            "home_ema_gf", "home_ema_ga", "away_ema_gf", "away_ema_ga",
            "home_attack_edge", "away_attack_edge", "neutral_flag",
            "log_mv_ratio", "log_mv_home", "log_mv_away"]


def snapshot(row):
    h, a = row.home_team, row.away_team
    f = {
        "home_elo": elo.get(h, BASE_ELO), "away_elo": elo.get(a, BASE_ELO),
        "home_ema_gf": ema_for.get(h, EMA_INIT), "home_ema_ga": ema_against.get(h, EMA_INIT),
        "away_ema_gf": ema_for.get(a, EMA_INIT), "away_ema_ga": ema_against.get(a, EMA_INIT),
        "neutral_flag": int(row.neutral),
    }
    f["elo_diff"] = f["home_elo"] - f["away_elo"]
    f["home_attack_edge"] = f["home_ema_gf"] - f["away_ema_ga"]
    f["away_attack_edge"] = f["away_ema_gf"] - f["home_ema_ga"]
    q = row.date.to_period("Q")
    mv_h, mv_a = team_value(h, q), team_value(a, q)
    f["log_mv_home"] = np.log(mv_h) if mv_h and not np.isnan(mv_h) else np.nan
    f["log_mv_away"] = np.log(mv_a) if mv_a and not np.isnan(mv_a) else np.nan
    f["log_mv_ratio"] = f["log_mv_home"] - f["log_mv_away"]
    return f


def update_state(row):
    h, a = row.home_team, row.away_team
    hs, as_ = row.home_score, row.away_score
    r_h, r_a = elo.get(h, BASE_ELO), elo.get(a, BASE_ELO)
    adv = 0.0 if row.neutral else HOME_ADV
    exp_h = 1.0 / (1.0 + 10 ** ((r_a - (r_h + adv)) / 400.0))
    result_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
    gd = abs(hs - as_)
    g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
    delta = K_BY_TOURNAMENT.get(row.tournament, DEFAULT_K) * g * (result_h - exp_h)
    elo[h], elo[a] = r_h + delta, r_a - delta
    for team, gf, ga in [(h, hs, as_), (a, as_, hs)]:
        ema_for[team] = EMA_ALPHA * gf + (1 - EMA_ALPHA) * ema_for.get(team, EMA_INIT)
        ema_against[team] = EMA_ALPHA * ga + (1 - EMA_ALPHA) * ema_against.get(team, EMA_INIT)


rows = []
for row in played.itertuples(index=False):
    rows.append(snapshot(row))
    update_state(row)

data = pd.concat([played, pd.DataFrame(rows)], axis=1)
data["outcome"] = np.select(
    [data.home_score > data.away_score, data.home_score == data.away_score], [0, 1], 2)

# ---------------- train strictly pre-tournament ----------------
modern = data[data["date"] >= "1990-01-01"]
train = modern[modern["date"] < CUTOFF]
wc = data[(data["date"] >= CUTOFF) & (data["tournament"] == "FIFA World Cup")].copy()
print(f"Train: {len(train):,} matches (through {train['date'].max().date()})")
print(f"Test:  {len(wc):,} played World Cup 2026 matches "
      f"({wc['date'].min().date()} to {wc['date'].max().date()})")

age = (CUTOFF - train["date"]).dt.days / 365.25
w = np.asarray(0.5 ** (age / 10))

clf = CalibratedClassifierCV(
    HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                   l2_regularization=1.0, random_state=RNG),
    method="isotonic", cv=3)
clf.fit(train[FEATURES], train["outcome"], sample_weight=w)

reg_h = HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05,
                                      max_depth=4, l2_regularization=1.0, random_state=RNG)
reg_a = HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05,
                                      max_depth=4, l2_regularization=1.0, random_state=RNG)
reg_h.fit(train[FEATURES], train["home_score"], sample_weight=w)
reg_a.fit(train[FEATURES], train["away_score"], sample_weight=w)

# ---------------- predict the tournament ----------------
proba = clf.predict_proba(wc[FEATURES])
lam_h = np.clip(reg_h.predict(wc[FEATURES]), 0.01, 8)
lam_a = np.clip(reg_a.predict(wc[FEATURES]), 0.01, 8)

CLASS = np.array(["home", "draw", "away"])
res = wc[["date", "home_team", "away_team", "home_score", "away_score", "outcome"]].copy()
res[["p_home", "p_draw", "p_away"]] = proba.round(3)
res["pred"] = CLASS[proba.argmax(1)]
res["actual"] = CLASS[res["outcome"]]
res["hit"] = res["pred"] == res["actual"]
res["p_actual"] = proba[np.arange(len(res)), res["outcome"]]

modal = []
for lh, la in zip(lam_h, lam_a):
    m = np.outer(poisson.pmf(np.arange(MAX_GOALS + 1), lh),
                 poisson.pmf(np.arange(MAX_GOALS + 1), la))
    i, j = np.unravel_index(m.argmax(), m.shape)
    modal.append(f"{i}-{j}")
res["pred_score"] = modal
res["score_hit"] = res["pred_score"] == (res["home_score"].astype(str) + "-" + res["away_score"].astype(str))

# ---------------- metrics ----------------
y = wc["outcome"].values
base_rates = train["outcome"].value_counts(normalize=True).sort_index().values
naive = np.tile(base_rates, (len(y), 1))


def brier(y, p):
    return np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))


print(f"\n=== World Cup 2026 backtest ({len(wc)} matches) ===")
print(f"{'Model':<22} LogLoss={log_loss(y, proba, labels=[0,1,2]):.4f}  "
      f"Brier={brier(y, proba):.4f}  Acc={res['hit'].mean():.1%}")
print(f"{'Naive base rates':<22} LogLoss={log_loss(y, naive, labels=[0,1,2]):.4f}  "
      f"Brier={brier(y, naive):.4f}  Acc={max(base_rates):.1%}")
print(f"Exact scoreline hit rate: {res['score_hit'].mean():.1%}")
print(f"Outcome distribution: {np.bincount(y, minlength=3)} (home/draw/away)")

print("\nBest calls (model most confident AND right):")
top = res[res["hit"]].nlargest(5, "p_actual")
for r in top.itertuples():
    print(f"  {r.date.date()} {r.home_team} {r.home_score}-{r.away_score} {r.away_team}"
          f"  (predicted {r.pred} @ {r.p_actual:.0%}, score {r.pred_score})")

print("\nWorst misses (model most confident AND wrong):")
miss = res[~res["hit"]].nsmallest(5, "p_actual")
for r in miss.itertuples():
    print(f"  {r.date.date()} {r.home_team} {r.home_score}-{r.away_score} {r.away_team}"
          f"  (gave actual outcome only {r.p_actual:.0%})")

res.drop(columns=["outcome"]).to_csv("backtest_wc2026.csv", index=False)
print("\nSaved: backtest_wc2026.csv")
