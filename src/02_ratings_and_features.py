"""
Step 2: team ratings and pregame features, built from TRAINING games only.

Reads  (DATA dir, default data/processed):
    panel_v2.parquet      from step 1 (receiver-game rows, has is_test)
    game_split.parquet    from step 1 (game_id, team, season, is_test)
    team_games.parquet    team-game table (pass_att, season, week, opponent, ...)
    receiver_games.parquet receiver stat lines (used only to get team passing yards)
Writes:
    panel_v2_feat.parquet panel plus frozen context features
    team_game_ratings.parquet  team-game level ratings (for the team-level M0 checks)

Rules enforced here:
  * Every rating and prior uses training games only (is_test == False).
  * A TRAINING row's feature leaves out its own game, so the model is not fit on a
    rating that contains the outcome being fit.  A TEST row's feature uses all
    training games (its own game is not in training, so nothing to leave out).
  * No quarterback term.  Context = leave-one-out opponent pass defense (K=220)
    and teammates-only offense (K=100), as frozen after the QB test.
  * A self-check scrambles test-period yards and asserts no feature changes.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")

K_OFF = 100.0
K_DEF = 220.0

FEATURES = ["off_team_ypa", "def_rating", "off_oth_ypa", "rec_tr_rate", "rec_tr_games"]


def team_game_table(tg, rg, split):
    """One row per team-game with passing yards (sum of receiver yards, as in the QB test)."""
    yd = rg.group_by(["game_id", "team"]).agg(pass_yds=pl.col("yds").sum())
    return (tg.filter(pl.col("pass_att") >= 10)
              .join(yd, on=["game_id", "team"], how="left")
              .with_columns(pass_yds=pl.col("pass_yds").fill_null(0).clip(lower_bound=0))
              .join(split.select("game_id", "team", "is_test"), on=["game_id", "team"], how="inner")
              .select("game_id", "season", "week", "seasonType", "team", "opponent",
                      "pass_att", "pass_yds", "is_test"))


def build_features(panel, T):
    """panel: receiver-game rows (needs game_id, team, athlete_id, season, yds, pass_att, is_test).
       T: team-game table from team_game_table. Returns (panel_with_features, team_ratings)."""
    tr_t = (~pl.col("is_test")).cast(pl.Float64)     # 1.0 on training rows

    # ---- priors from training games only
    Ttr = T.filter(~pl.col("is_test"))
    m = Ttr["pass_yds"].sum() / Ttr["pass_att"].sum()          # team yards per attempt
    ptr = panel.filter(~pl.col("is_test"))
    mean_rec_ypa = ptr["yds"].sum() / ptr["pass_att"].sum()    # one receiver's yards per team attempt
    m_oth = m - mean_rec_ypa                                   # everyone else's yards per attempt

    # ---- team offense (shrunk yards per attempt), leave-own-game-out on training rows
    T = T.with_columns(
        S=(tr_t * pl.col("pass_yds")).sum().over(["season", "team"]),
        A=(tr_t * pl.col("pass_att")).sum().over(["season", "team"]),
    ).with_columns(
        off_team_ypa=(pl.col("S") - tr_t * pl.col("pass_yds") + K_OFF * m)
                     / (pl.col("A") - tr_t * pl.col("pass_att") + K_OFF),
    )

    # ---- opponent pass defense: attempt-weighted excess yards per attempt allowed
    # excess is only defined for training games; test games contribute nothing
    T = T.with_columns(
        exc_w=pl.when(~pl.col("is_test"))
               .then(pl.col("pass_att") * (pl.col("pass_yds") / pl.col("pass_att") - pl.col("off_team_ypa")))
               .otherwise(0.0),
        att_tr=tr_t * pl.col("pass_att"),
    )
    D = (T.group_by(["season", "opponent"])
          .agg(W=pl.col("exc_w").sum(), A_d=pl.col("att_tr").sum()))
    T = (T.join(D, on=["season", "opponent"], how="left")
           .with_columns(
               def_rating=(pl.col("W") - pl.col("exc_w"))
                          / (pl.col("A_d") - pl.col("att_tr") + K_DEF))
           .drop("S", "A", "W", "A_d", "exc_w", "att_tr"))

    # ---- receiver-level features
    P = (panel.drop([c for c in FEATURES if c in panel.columns])
              .join(T.select("game_id", "team", "pass_yds", "off_team_ypa", "def_rating"),
                    on=["game_id", "team"], how="left", suffix="_t")
              .with_columns(
                  cl_rec=pl.col("athlete_id").cast(pl.String) + "|" + pl.col("team") + "|"
                         + pl.col("season").cast(pl.String),
                  oth_yds=(pl.col("pass_yds") - pl.col("yds")).clip(lower_bound=0)))
    P = P.with_columns(
        tr=(~pl.col("is_test")).cast(pl.Float64),
    ).with_columns(
        o_y=(pl.col("tr") * pl.col("oth_yds")).sum().over("cl_rec"),
        o_a=(pl.col("tr") * pl.col("pass_att")).sum().over("cl_rec"),
        r_y=(pl.col("tr") * pl.col("yds")).sum().over("cl_rec"),
        r_g=pl.col("tr").sum().over("cl_rec"),
    ).with_columns(
        # teammates-only offense: everyone else's yards per attempt, receiver excluded
        off_oth_ypa=(pl.col("o_y") - pl.col("tr") * pl.col("oth_yds") + K_OFF * m_oth)
                    / (pl.col("o_a") - pl.col("tr") * pl.col("pass_att") + K_OFF),
        # the receiver's own training-period rate and games (the "own average" baseline)
        rec_tr_games=pl.col("r_g") - pl.col("tr"),
        rec_tr_rate=pl.when(pl.col("o_a") - pl.col("tr") * pl.col("pass_att") > 0)
                      .then((pl.col("r_y") - pl.col("tr") * pl.col("yds"))
                            / (pl.col("o_a") - pl.col("tr") * pl.col("pass_att")))
                      .otherwise(None),
    ).drop("o_y", "o_a", "r_y", "r_g", "tr", "oth_yds", "cl_rec")

    return P, T, dict(m=m, mean_rec_ypa=mean_rec_ypa, m_oth=m_oth)


def leakage_selfcheck(panel, T):
    """Scramble every yardage number in TEST games. No feature may change, on any row."""
    rng = np.random.default_rng(1)
    base, _, _ = build_features(panel, T)

    p2 = panel.with_columns(
        f=pl.Series(rng.uniform(0.0, 3.0, panel.height)))
    p2 = p2.with_columns(
        yds=pl.when(pl.col("is_test")).then(pl.col("yds") * pl.col("f")).otherwise(pl.col("yds")),
        rec=pl.when(pl.col("is_test")).then(pl.col("rec") * 2).otherwise(pl.col("rec"))).drop("f")
    t2 = T.with_columns(
        f=pl.Series(rng.uniform(0.0, 3.0, T.height)))
    t2 = t2.with_columns(
        pass_yds=pl.when(pl.col("is_test")).then(pl.col("pass_yds") * pl.col("f")).otherwise(pl.col("pass_yds"))).drop("f")
    scr, _, _ = build_features(p2, t2)

    key = ["game_id", "team", "athlete_id"]
    a = base.sort(key).select(key + FEATURES)
    b = scr.sort(key).select(key + FEATURES)
    for c in FEATURES:
        x, y = a[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
        same_nan = np.array_equal(np.isnan(x), np.isnan(y))
        diff = np.nanmax(np.abs(x - y)) if np.isfinite(x).any() else 0.0
        assert same_nan and diff < 1e-9, f"LEAKAGE: {c} changed when test-period yards were scrambled (max diff {diff})"
    print("leakage self-check passed: scrambling test-period yards changed no feature on any row")


def main():
    panel = pl.read_parquet(DATA / "panel_v2.parquet")
    split = pl.read_parquet(DATA / "game_split.parquet")
    tg = pl.read_parquet(DATA / "team_games.parquet")
    rg = pl.read_parquet(DATA / "receiver_games.parquet")

    T = team_game_table(tg, rg, split)
    # panel already carries pass_att and is_test from step 1; its team passing yards come from T
    leakage_selfcheck(panel, T)

    P, TR, pri = build_features(panel, T)
    print(f"training priors: team YPA {pri['m']:.3f} | one receiver {pri['mean_rec_ypa']:.3f} | everyone else {pri['m_oth']:.3f}")

    # ---- sanity prints
    for lab, flag in (("train", False), ("test", True)):
        s = P.filter(pl.col("is_test") == flag)
        print(f"{lab}: rows {s.height} | nulls " + ", ".join(f"{c}={s[c].null_count()}" for c in FEATURES))
    s = P.filter(pl.col("is_test")).drop_nulls("def_rating")
    w = s["pass_att"]
    print("test def_rating (att-weighted mean should be near 0):",
          round(float((s["def_rating"] * w).sum() / w.sum()), 4),
          "| sd:", round(float(s["def_rating"].std()), 3))
    print("games with no training games for the opponent (rating = 0 by shrinkage):",
          int((TR.filter(pl.col("is_test"))["def_rating"] == 0).sum()))

    P.write_parquet(DATA / "panel_v2_feat.parquet")
    TR.write_parquet(DATA / "team_game_ratings.parquet")
    print("wrote panel_v2_feat.parquet and team_game_ratings.parquet")


if __name__ == "__main__":
    main()