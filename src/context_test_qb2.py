import gzip, json, glob
import numpy as np, polars as pl

K_PRIMARY = 200.0
K_SENS = (75.0, 500.0)
K_OFF = 100.0
K_DEF = 220.0
MIN_ATT_START = 10     # a passer needs this many attempts in a game for it to count as a start
MIN_REC_GAMES_PER_QB = 2  # minimum games a receiver must play with a given QB in that season
N_BOOT = 1000

tg = pl.read_parquet("data/processed/team_games.parquet")
rg = pl.read_parquet("data/processed/receiver_games.parquet")

REC_ID = next(
    c for c in ("player_id", "athlete_id", "receiver_id", "id")
    if c in rg.columns
)

yd = rg.group_by(["game_id", "team"]).agg(pass_yds=pl.col("yds").sum())
T0 = (tg.filter(pl.col("pass_att") >= 10)
        .join(yd, on=["game_id", "team"], how="left")
        .with_columns(pass_yds=pl.col("pass_yds").fill_null(0).clip(lower_bound=0))
        .select("game_id", "season", "team", "opponent", "pass_att", "pass_yds"))
m = T0["pass_yds"].sum() / T0["pass_att"].sum()

# the passer with the most attempts in each team-game, and how many he threw
rows = []
for f in sorted(glob.glob("data/raw/cfbd/games_players/*.json.gz")):
    for g in json.load(gzip.open(f, "rt")):
        for t in g["teams"]:
            att = {}
            for cat in t["categories"]:
                if cat["name"] != "passing":
                    continue
                for ty in cat["types"]:
                    if ty["name"] != "C/ATT":
                        continue
                    for a in ty["athletes"]:
                        try:
                            att[a["id"]] = int(a["stat"].split("/")[1])
                        except (ValueError, IndexError, AttributeError):
                            pass
            if att:
                q = max(att, key=att.get)
                rows.append({"game_id": g["id"], "team": t["team"], "qb_id": q, "qb_att": att[q]})
qb = pl.DataFrame(rows).unique(subset=["game_id", "team"])


def defense_ratings(k_def):
    T = T0.with_columns(
        S_o=pl.col("pass_yds").sum().over(["season", "team"]),
        A_o=pl.col("pass_att").sum().over(["season", "team"]),
        ypa=pl.col("pass_yds") / pl.col("pass_att"))
    T = T.with_columns(off_loo=(pl.col("S_o") - pl.col("pass_yds") + K_OFF * m)
                               / (pl.col("A_o") - pl.col("pass_att") + K_OFF))
    T = T.with_columns(exc=pl.col("ypa") - pl.col("off_loo"))
    Dd = (T.group_by(["season", "opponent"])
            .agg(W=(pl.col("pass_att") * pl.col("exc")).sum(), A_d=pl.col("pass_att").sum()))
    T = T.join(Dd, on=["season", "opponent"], how="left")
    T = T.with_columns(def_loo=(pl.col("W") - pl.col("pass_att") * pl.col("exc"))
                               / (pl.col("A_d") - pl.col("pass_att") + k_def))
    return T.select("game_id", "team", "def_loo")


DR = defense_ratings(K_DEF)

S = (T0.join(qb, on=["game_id", "team"], how="inner")
       .join(DR, on=["game_id", "team"], how="left")
       .filter((pl.col("qb_att") >= MIN_ATT_START) & pl.col("def_loo").is_not_null())
       .with_columns(
           ypa=pl.col("pass_yds") / pl.col("pass_att"),
           adj_pass_yds=pl.col("pass_yds") - pl.col("pass_att") * pl.col("def_loo")
       )
       .sort(["qb_id", "season", "game_id"]))

# Chronological game index within each team-season (1..N_games) for the primary 2/3 train -> 1/3 holdout split
team_sched = (T0.select("game_id", "season", "team")
                .unique()
                .sort(["season", "team", "game_id"])
                .with_columns(
                    team_game_num=pl.int_range(1, pl.len() + 1).over(["season", "team"]),
                    team_n_games=pl.len().over(["season", "team"]),
                )
                .with_columns(
                    # First 2/3 of each team's season = training window; last 1/3 = holdout test window
                    is_train_split=pl.col("team_game_num") <= (pl.col("team_n_games") * 2 // 3)
                ))

S = S.join(team_sched.select("game_id", "team", "team_game_num", "team_n_games", "is_train_split"),
           on=["game_id", "team"], how="left")

# Join receiver games to team-QB games
RG = (rg.select("game_id", "team", REC_ID, pl.col("yds").fill_null(0).clip(lower_bound=0).alias("rec_yds"))
        .join(S, on=["game_id", "team"], how="inner")
        .with_columns(
            rec_rate=pl.col("rec_yds") / pl.col("pass_att"),
            # QB's defense-adjusted passing yards in this game thrown to OTHER teammates (excluding receiver r)
            other_adj_yds=pl.col("adj_pass_yds") - pl.col("rec_yds"),
            cl_rec=pl.col(REC_ID).cast(pl.String) + "|" + pl.col("team") + "|" + pl.col("season").cast(pl.String),
            cl_team=pl.col("team") + "|" + pl.col("season").cast(pl.String),
        )
        .sort([REC_ID, "qb_id", "season", "game_id"]))

mean_rec_ypa = float(RG["rec_yds"].sum() / RG["pass_att"].sum())
mean_other_ypa = m - mean_rec_ypa  # Exact baseline prior for (adj_pass_yds - rec_yds) / pass_att

RG = RG.with_columns(
    # 1. Career leave-one-game-out QB production w/o receiver r (all other starts of QB's career)
    tot_other_yds_career=pl.col("other_adj_yds").sum().over(["qb_id", REC_ID]),
    tot_att_career=pl.col("pass_att").sum().over(["qb_id", REC_ID]),
    # 2. Within-team-season leave-one-game-out QB production w/o receiver r
    tot_other_yds_season=pl.col("other_adj_yds").sum().over(["cl_team", "qb_id", REC_ID]),
    tot_att_season=pl.col("pass_att").sum().over(["cl_team", "qb_id", REC_ID]),
    # 3. Strictly pre-game cumulative QB production w/o receiver r across ALL career starts
    prior_other_yds_career=pl.col("other_adj_yds").cum_sum().over(["qb_id", REC_ID]) - pl.col("other_adj_yds"),
    prior_att_career=pl.col("pass_att").cum_sum().over(["qb_id", REC_ID]) - pl.col("pass_att"),
    # 4. Strictly pre-game cumulative QB production w/o receiver r within the CURRENT team-season ONLY
    prior_other_yds_season=pl.col("other_adj_yds").cum_sum().over(["cl_team", "qb_id", REC_ID]) - pl.col("other_adj_yds"),
    prior_att_season=pl.col("pass_att").cum_sum().over(["cl_team", "qb_id", REC_ID]) - pl.col("pass_att"),
).with_columns(
    yds_career_loo=pl.col("tot_other_yds_career") - pl.col("other_adj_yds"),
    att_career_loo=pl.col("tot_att_career") - pl.col("pass_att"),
    yds_season_loo=pl.col("tot_other_yds_season") - pl.col("other_adj_yds"),
    att_season_loo=pl.col("tot_att_season") - pl.col("pass_att"),
    # Prior-season-only cumulative stats (before the current season started)
    prior_other_yds_prev_seasons=pl.col("prior_other_yds_career") - pl.col("prior_other_yds_season"),
    prior_att_prev_seasons=pl.col("prior_att_career") - pl.col("prior_att_season"),
)

w = pl.col("pass_att")


def wd(c, cluster_col):
    return pl.col(c) - (pl.col(c) * w).sum().over(cluster_col) / w.sum().over(cluster_col)


def solve(M):
    s = M.sum(0)
    A = np.array([[s[0], s[1]], [s[1], s[2]]])
    b = np.array([s[3], s[4]])
    return np.linalg.solve(A, b)


def run_test(k, label, mode="season_loo"):
    """
    Modes:
    - "same_game_other":        QB's defense-adjusted YPA to OTHER teammates in the same game
    - "season_loo":             QB's defense-adjusted YPA to OTHER teammates in OTHER games of the SAME season
    - "season_plus_prior":      Hybrid LOO (current season + 0.5 * prior seasons w/o receiver r)
    - "pregame_career":         Strictly pre-game starts across entire career w/o receiver r (unweighted)
    - "pregame_current_season": Strictly pre-game starts in the CURRENT season only w/o receiver r
    - "pregame_weighted":       Strictly pre-game starts (1.0 * current-season prior games + 0.2 * prior-season games)
    """
    if mode == "same_game_other":
        x_expr = pl.col("other_adj_yds") / pl.col("pass_att")
    elif mode == "season_loo":
        x_expr = (pl.col("yds_season_loo") + k * mean_other_ypa) / (pl.col("att_season_loo") + k)
    elif mode == "season_plus_prior":
        prior_season_yds = pl.col("yds_career_loo") - pl.col("yds_season_loo")
        prior_season_att = pl.col("att_career_loo") - pl.col("att_season_loo")
        eff_yds = pl.col("yds_season_loo") + 0.5 * prior_season_yds
        eff_att = pl.col("att_season_loo") + 0.5 * prior_season_att
        x_expr = (eff_yds + k * mean_other_ypa) / (eff_att + k)
    elif mode == "pregame_current_season":
        x_expr = (pl.col("prior_other_yds_season") + k * mean_other_ypa) / (pl.col("prior_att_season") + k)
    elif mode == "pregame_weighted":
        eff_yds = pl.col("prior_other_yds_season") + 0.2 * pl.col("prior_other_yds_prev_seasons")
        eff_att = pl.col("prior_att_season") + 0.2 * pl.col("prior_att_prev_seasons")
        x_expr = (eff_yds + k * mean_other_ypa) / (eff_att + k)
    else:  # pregame_career
        x_expr = (pl.col("prior_other_yds_career") + k * mean_other_ypa) / (pl.col("prior_att_career") + k)

    scale_to_team_ypa = m / mean_other_ypa
    D = (RG.filter(pl.col("season") >= 2011)
           .with_columns(
               x=x_expr * scale_to_team_ypa,
               rec_rate_team_eq=pl.col("rec_rate") * (m / mean_rec_ypa)
           ))

    ns = (D.group_by(["cl_rec", "qb_id"])
           .agg(n=pl.len())
           .filter(pl.col("n") >= MIN_REC_GAMES_PER_QB))
    multi = ns.group_by("cl_rec").agg(nq=pl.len()).filter(pl.col("nq") >= 2)

    D = (D.join(ns.select("cl_rec", "qb_id"), on=["cl_rec", "qb_id"], how="inner")
          .join(multi.select("cl_rec"), on="cl_rec", how="inner"))

    D = D.with_columns(
        xd=wd("x", "cl_rec"),
        dd=wd("def_loo", "cl_rec"),
        yd_raw=wd("rec_rate", "cl_rec"),
        yd_eq=wd("rec_rate_team_eq", "cl_rec"),
    )

    g = D.group_by("cl_team").agg(
        sxx=(w * pl.col("xd") ** 2).sum(),
        sxd=(w * pl.col("xd") * pl.col("dd")).sum(),
        sdd=(w * pl.col("dd") ** 2).sum(),
        sxy_raw=(w * pl.col("xd") * pl.col("yd_raw")).sum(),
        sdy_raw=(w * pl.col("dd") * pl.col("yd_raw")).sum(),
        sxy_eq=(w * pl.col("xd") * pl.col("yd_eq")).sum(),
        sdy_eq=(w * pl.col("dd") * pl.col("yd_eq")).sum(),
    )
    M_raw = g.select("sxx", "sxd", "sdd", "sxy_raw", "sdy_raw").to_numpy()
    M_eq = g.select("sxx", "sxd", "sdd", "sxy_eq", "sdy_eq").to_numpy()
    n_teams = M_raw.shape[0]
    beta_raw = solve(M_raw)
    beta_eq = solve(M_eq)

    rng = np.random.default_rng(0)
    bx_eq, bd_eq = [], []
    for _ in range(N_BOOT):
        s = rng.integers(0, n_teams, n_teams)
        b = solve(M_eq[s])
        bx_eq.append(b[0])
        bd_eq.append(b[1])

    sd_x = float(np.sqrt((D["pass_att"] * D["xd"] ** 2).sum() / D["pass_att"].sum()))
    print(f"[{mode}] {label}: team-seasons {n_teams}, rec-seasons {D['cl_rec'].n_unique()}, rec-games {D.height}")
    print(f"   QB slope (team-equiv scale):  {beta_eq[0]:.3f} (95% {np.percentile(bx_eq, 2.5):.3f} to {np.percentile(bx_eq, 97.5):.3f})"
          f" | per-receiver raw slope: {beta_raw[0]:.4f} | within-rec SD(x): {sd_x:.3f}")
    print(f"   DEF slope (team-equiv scale): {beta_eq[1]:.3f} (95% {np.percentile(bd_eq, 2.5):.3f} to {np.percentile(bd_eq, 97.5):.3f})"
          f" | per-receiver raw slope: {beta_raw[1]:.4f}")


print(f"Baseline team YPA (m): {m:.3f} | Mean single-receiver YPA: {mean_rec_ypa:.3f} "
      f"(1/{m/mean_rec_ypa:.2f} of team) | Mean OTHER-teammates YPA: {mean_other_ypa:.3f}\n")

print("=== 0. SANITY CHECK: SAME-GAME QB YPA TO OTHER TEAMMATES (excluding receiver r) ===")
run_test(K_PRIMARY, "same-game other WRs", mode="same_game_other")

print("\n=== 1. WITHIN-SEASON LOO QB PRODUCTION W/O TARGET RECEIVER (other games this season) ===")
run_test(K_PRIMARY, f"PRIMARY k={K_PRIMARY:.0f}", mode="season_loo")
for k in K_SENS:
    run_test(k, f"sensitivity k={k:.0f}", mode="season_loo")

print("\n=== 2. HYBRID LOO (current season + 0.5 * prior seasons w/o target receiver) ===")
run_test(K_PRIMARY, f"PRIMARY k={K_PRIMARY:.0f}", mode="season_plus_prior")
for k in K_SENS:
    run_test(k, f"sensitivity k={k:.0f}", mode="season_plus_prior")

print("\n=== 3. STRICTLY PRE-GAME: CURRENT-SEASON PRIOR GAMES ONLY (w/o target receiver) ===")
run_test(K_PRIMARY, f"PRIMARY k={K_PRIMARY:.0f}", mode="pregame_current_season")
for k in K_SENS:
    run_test(k, f"sensitivity k={k:.0f}", mode="pregame_current_season")

print("\n=== 4. STRICTLY PRE-GAME: WEIGHTED (1.0 * current season + 0.2 * prior seasons w/o target receiver) ===")
run_test(K_PRIMARY, f"PRIMARY k={K_PRIMARY:.0f}", mode="pregame_weighted")
for k in K_SENS:
    run_test(k, f"sensitivity k={k:.0f}", mode="pregame_weighted")