import numpy as np, polars as pl

K_PRIMARY = 300.0     # shrinkage in pass attempts, set in advance
K_SENS = (100.0, 900.0)
N_BOOT = 1000

tg = pl.read_parquet("data/processed/team_games.parquet")
rg = pl.read_parquet("data/processed/receiver_games.parquet")
qb = pl.read_parquet("data/processed/team_game_qb.parquet").select("game_id", "team", "qb_id")

yd = rg.group_by(["game_id", "team"]).agg(pass_yds=pl.col("yds").sum())
T = (tg.filter(pl.col("pass_att") >= 10)
       .join(yd, on=["game_id", "team"], how="left")
       .with_columns(pass_yds=pl.col("pass_yds").fill_null(0).clip(lower_bound=0))
       .join(qb, on=["game_id", "team"], how="inner")
       .select("game_id", "season", "team", "qb_id", "pass_att", "pass_yds")
       .with_columns(ypa=pl.col("pass_yds") / pl.col("pass_att")))
m = T["pass_yds"].sum() / T["pass_att"].sum()

# each quarterback's cumulative record in strictly earlier seasons, at any team
qs = (T.group_by(["qb_id", "season"])
        .agg(S=pl.col("pass_yds").sum(), A=pl.col("pass_att").sum())
        .sort(["qb_id", "season"])
        .with_columns(cS=pl.col("S").cum_sum().over("qb_id") - pl.col("S"),
                      cA=pl.col("A").cum_sum().over("qb_id") - pl.col("A")))


def run(k, label):
    D = (T.join(qs.select("qb_id", "season", "cS", "cA"), on=["qb_id", "season"], how="left")
           .filter(pl.col("season") >= 2011)
           .with_columns(x=(pl.col("cS") + k * m) / (pl.col("cA") + k),
                         cl=pl.col("team") + "|" + pl.col("season").cast(pl.String)))
    w = pl.col("pass_att")
    D = D.with_columns(
        xd=pl.col("x") - (pl.col("x") * w).sum().over("cl") / w.sum().over("cl"),
        yd=pl.col("ypa") - (pl.col("ypa") * w).sum().over("cl") / w.sum().over("cl"),
        nq=pl.col("qb_id").n_unique().over("cl"))
    sw = D.filter(pl.col("nq") >= 2)       # only team-seasons with 2+ starting quarterbacks carry information
    g = sw.group_by("cl").agg(sxy=(pl.col("pass_att") * pl.col("xd") * pl.col("yd")).sum(),
                              sxx=(pl.col("pass_att") * pl.col("xd") ** 2).sum())
    a_xy, a_xx = g["sxy"].to_numpy(), g["sxx"].to_numpy()
    n = len(a_xy)
    beta = a_xy.sum() / a_xx.sum()
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        bs.append(a_xy[s].sum() / a_xx[s].sum())
    sd = float(np.sqrt((sw["pass_att"] * sw["xd"] ** 2).sum() / sw["pass_att"].sum()))
    print(f"{label}: team-seasons {D['cl'].n_unique()}, with 2+ starting QBs {n}, games {sw.height}")
    print(f"   slope {beta:.3f} (95% {np.percentile(bs, 2.5):.3f} to {np.percentile(bs, 97.5):.3f})"
          f" | within-team-season SD of the prior rating {sd:.3f} yards per attempt")


run(K_PRIMARY, f"PRIMARY k={K_PRIMARY:.0f}")
for k in K_SENS:
    run(k, f"sensitivity k={k:.0f}")