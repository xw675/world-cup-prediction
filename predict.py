"""
Reusable prediction tool — retrains on the current data and predicts fixtures.

HOW TO KEEP DATA FRESH
  * Played results: re-download results.csv from the Kaggle dataset
    (martj42/international-football-results...) into datasets/ — it is
    updated after every international window. Or append rows by hand.
  * Future fixtures, three options (combined automatically):
      1. Rows in datasets/results.csv with empty scores (as Kaggle ships them)
      2. Rows you add to fixtures.csv  (date,home_team,away_team,neutral)
      3. One-off from the command line

USAGE
  python3 predict.py                          # predict all known fixtures
  python3 predict.py Brazil France --neutral  # ad-hoc matchup
  python3 predict.py Mexico England           # Mexico at home (not neutral)

Output: table + predictions_latest.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RNG, MAX_GOALS = 42, 8
HERE = Path(__file__).parent

# ----------------------------------------------------------------------
# Feature pipeline (same from-scratch Elo + EMA as the other scripts)
# ----------------------------------------------------------------------
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

# ---- squad market value (validated in experiment_market_value.py) ----
TOP_N = 25
TM_NAME = {
    "South Korea": "Korea, South", "North Korea": "Korea, North",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Republic of Ireland": "Ireland", "Curaçao": "Curacao",
    "Hong Kong": "Hongkong", "Gambia": "The Gambia",
    "Brunei": "Brunei Darussalam", "South Sudan": "Southern Sudan",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Ivory Coast": "Cote d'Ivoire", "Taiwan": "Chinese Taipei",
}
squad_dict, LATEST_Q = {}, None


def build_squad_values():
    """Quarterly squad value per country = sum of top-25 citizens' values."""
    global squad_dict, LATEST_Q
    vpath, ppath = HERE / "datasets/player_valuations.csv", HERE / "datasets/players.csv"
    if not (vpath.exists() and ppath.exists()):
        print("NOTE: player_valuations.csv/players.csv not found — "
              "predicting without market value features.")
        return
    vals = pd.read_csv(vpath, usecols=["player_id", "date", "market_value_in_eur"],
                       parse_dates=["date"])
    players = pd.read_csv(ppath, usecols=["player_id", "country_of_citizenship"]).dropna()
    vals["q"] = vals["date"].dt.to_period("Q")
    panel = (vals.sort_values("date")
                 .groupby(["player_id", "q"])["market_value_in_eur"].last()
                 .unstack("q").sort_index(axis=1).ffill(axis=1, limit=8))
    long = (panel.stack().rename("value").reset_index()
                 .merge(players, on="player_id", how="inner"))
    squad_dict = (long.groupby(["country_of_citizenship", "q"])["value"]
                      .apply(lambda s: s.nlargest(TOP_N).sum()).to_dict())
    LATEST_Q = vals["q"].max()


def team_value(team, match_q):
    """Squad value in the quarter BEFORE the match (strictly pre-match)."""
    return squad_dict.get((TM_NAME.get(team, team), match_q - 1), np.nan)


def snapshot(home, away, neutral, date=None):
    f = {
        "home_elo": elo.get(home, BASE_ELO), "away_elo": elo.get(away, BASE_ELO),
        "home_ema_gf": ema_for.get(home, EMA_INIT), "home_ema_ga": ema_against.get(home, EMA_INIT),
        "away_ema_gf": ema_for.get(away, EMA_INIT), "away_ema_ga": ema_against.get(away, EMA_INIT),
        "neutral_flag": int(neutral),
    }
    f["elo_diff"] = f["home_elo"] - f["away_elo"]
    f["home_attack_edge"] = f["home_ema_gf"] - f["away_ema_ga"]
    f["away_attack_edge"] = f["away_ema_gf"] - f["home_ema_ga"]
    q = pd.Timestamp(date).to_period("Q") if date is not None else (
        LATEST_Q + 1 if LATEST_Q is not None else pd.Timestamp.today().to_period("Q"))
    mv_h, mv_a = team_value(home, q), team_value(away, q)
    f["log_mv_home"] = np.log(mv_h) if mv_h and not np.isnan(mv_h) else np.nan
    f["log_mv_away"] = np.log(mv_a) if mv_a and not np.isnan(mv_a) else np.nan
    f["log_mv_ratio"] = f["log_mv_home"] - f["log_mv_away"]
    return f


def update_state(row):
    h, a, hs, as_ = row.home_team, row.away_team, row.home_score, row.away_score
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


def score_matrix(lam_h, lam_a):
    """Independent Poisson score grid (coupling below handles draw mass)."""
    ph = poisson.pmf(np.arange(MAX_GOALS + 1), lam_h)
    pa = poisson.pmf(np.arange(MAX_GOALS + 1), lam_a)
    m = np.outer(ph, pa)
    return m / m.sum()


_IL = np.tril_indices(MAX_GOALS + 1, -1)
_IU = np.triu_indices(MAX_GOALS + 1, 1)


def couple_matrix(m, p):
    """Rescale the win/draw/loss regions of the score matrix so its implied
    outcome probabilities match the (better-calibrated) ensemble's p.
    Validated in experiment_scorelines.py."""
    m = m.copy()
    diag = np.arange(MAX_GOALS + 1)
    m[_IL] *= p[0] / m[_IL].sum()
    m[diag, diag] *= p[1] / np.trace(m)
    m[_IU] *= p[2] / m[_IU].sum()
    return m / m.sum()


# ----------------------------------------------------------------------
# Load data, build features chronologically
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Predict international matches.")
    ap.add_argument("home", nargs="?", help="home team (ad-hoc matchup)")
    ap.add_argument("away", nargs="?", help="away team (ad-hoc matchup)")
    ap.add_argument("--neutral", action="store_true", help="ad-hoc match on neutral ground")
    args = ap.parse_args()

    df = pd.read_csv(HERE / "datasets/results.csv", parse_dates=["date"])
    fn = pd.read_csv(HERE / "datasets/former_names.csv")
    name_map = dict(zip(fn["former"], fn["current"]))
    for col in ["home_team", "away_team"]:
        df[col] = df[col].replace(name_map)

    played = df[df["home_score"].notna()].copy()
    played[["home_score", "away_score"]] = played[["home_score", "away_score"]].astype(int)
    played = played.sort_values("date").reset_index(drop=True)
    print(f"Data: {len(played):,} played matches through {played['date'].max().date()}")

    build_squad_values()

    rows = []
    for row in played.itertuples(index=False):
        rows.append(snapshot(row.home_team, row.away_team, row.neutral, row.date))
        update_state(row)
    data = pd.concat([played, pd.DataFrame(rows)], axis=1)
    data["outcome"] = np.select(
        [data.home_score > data.away_score, data.home_score == data.away_score], [0, 1], 2)
    modern = data[data["date"] >= "1990-01-01"]

    # ------------------------------------------------------------------
    # Train final models on everything up to now.
    # Recency weighting: recent matches count more (10-year half-life) —
    # validated in experiment_recency_weighting.py.
    # ------------------------------------------------------------------
    HALF_LIFE_YEARS = 10
    age = (modern["date"].max() - modern["date"]).dt.days / 365.25
    w = np.asarray(0.5 ** (age / HALF_LIFE_YEARS))

    # Outcome model: 50/50 ensemble of LogReg and HistGB
    # (validated in experiment_ensemble.py — beats either model alone).
    gb_clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                            l2_regularization=1.0, random_state=RNG)
    gb_clf.fit(modern[FEATURES], modern["outcome"], sample_weight=w)

    med = modern[FEATURES].median()

    def impute(X):
        return X.fillna({"log_mv_ratio": 0}).fillna(med)

    sc = StandardScaler().fit(impute(modern[FEATURES]))
    lr_clf = LogisticRegression(max_iter=2000)
    lr_clf.fit(sc.transform(impute(modern[FEATURES])), modern["outcome"], sample_weight=w)

    def predict_outcome_proba(X):
        return 0.5 * gb_clf.predict_proba(X) + \
               0.5 * lr_clf.predict_proba(sc.transform(impute(X)))

    def make_reg():
        return HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05,
                                             max_depth=4, l2_regularization=1.0, random_state=RNG)
    reg_h = make_reg().fit(modern[FEATURES], modern["home_score"], sample_weight=w)
    reg_a = make_reg().fit(modern[FEATURES], modern["away_score"], sample_weight=w)

    # ------------------------------------------------------------------
    # Collect fixtures to predict
    # ------------------------------------------------------------------
    fixtures = []  # (date, home, away, neutral, source)
    for r in df[df["home_score"].isna()].itertuples(index=False):
        fixtures.append((r.date.date(), r.home_team, r.away_team, bool(r.neutral), "results.csv"))

    fx_path = HERE / "fixtures.csv"
    if fx_path.exists():
        fx = pd.read_csv(fx_path, parse_dates=["date"])
        for r in fx.itertuples(index=False):
            fixtures.append((r.date.date(), r.home_team, r.away_team, bool(r.neutral), "fixtures.csv"))

    if args.home and args.away:
        fixtures.append((pd.Timestamp.today().date(), args.home, args.away, args.neutral, "cli"))

    if not fixtures:
        sys.exit("No fixtures found. Add rows to fixtures.csv or pass teams on the command line.")

    # Warn about unknown team names (typos, unmapped names)
    known = set(elo)
    for _, h, a, _, src in fixtures:
        for t in (h, a):
            if t not in known:
                print(f"WARNING: '{t}' ({src}) has no match history — using default ratings. "
                      f"Check spelling / former_names.csv.")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    out = []
    for date, h, a, neutral, src in fixtures:
        X = pd.DataFrame([snapshot(h, a, neutral, date)])[FEATURES]
        p = predict_outcome_proba(X)[0]
        lam_h = float(np.clip(reg_h.predict(X)[0], 0.01, 8))
        lam_a = float(np.clip(reg_a.predict(X)[0], 0.01, 8))
        m = couple_matrix(score_matrix(lam_h, lam_a), p)
        flat = m.ravel().argsort()[::-1][:3]
        top3 = [(i // (MAX_GOALS + 1), i % (MAX_GOALS + 1), m.ravel()[i]) for i in flat]
        out.append({
            "date": date, "home_team": h, "away_team": a, "neutral": neutral,
            "p_home_win": round(p[0], 3), "p_draw": round(p[1], 3), "p_away_win": round(p[2], 3),
            "xg_home": round(lam_h, 2), "xg_away": round(lam_a, 2),
            "score_1": f"{top3[0][0]}-{top3[0][1]} ({top3[0][2]:.1%})",
            "score_2": f"{top3[1][0]}-{top3[1][1]} ({top3[1][2]:.1%})",
            "score_3": f"{top3[2][0]}-{top3[2][1]} ({top3[2][2]:.1%})",
            "source": src,
        })

    res = pd.DataFrame(out)
    print()
    print(res.to_string(index=False))
    res.to_csv(HERE / "predictions_latest.csv", index=False)
    print("\nSaved: predictions_latest.csv")


if __name__ == "__main__":
    main()
