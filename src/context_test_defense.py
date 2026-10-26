import numpy as np, polars as pl

K_OFF = 100.0                 # shrinkage in pass attempts for the offense baseline
K_PRIMARY = 150.0             # defense rating shrinkage, set in advance
K_SENS = (50.0, 400.0)        # sensitivity only
N_BOOT = 1000
pl.Config.set_tbl_rows(30)

tg = pl.read_parquet("data/processed/team_games.parquet")
rg = pl.read_parquet("data/processed/receiver_games.parquet")
panel = pl.read_parquet("data/processed/panel_v2.parquet")
gm = pl.read_parquet("data/processed/games.parquet").select(
    pl.col("id").alias("game_id"), "homeTeam", "homeClassification", "awayClassification")
panel = (panel.join(gm, on="game_id", how="left")
              .with_columns(own_class=pl.when(pl.col("team") == pl.col("homeTeam"))
                            .then(pl.col("homeClassification")).otherwise(pl.col("awayClassification"))))

yd = rg.group_by(["game_id", "team"]).agg(pass_yds=pl.col("yds").sum())
T0 = (tg.filter(pl.col("pass_att") >= 10)
        .join(yd, on=["game_id", "team"], how="left")
        .with_columns(pass_yds=pl.col("pass_yds").fill_null(0).clip(lower_bound=0))
        .select("game_id", "season", "team", "opponent", "pass_att", "pass_yds"))
m = T0["pass_yds"].sum() / T0["pass_att"].sum()

P0 = (panel.select("game_id", "team", "athlete_id", "season", "pass_att", "yds", "own_class")
           .with_columns(y=pl.col("yds").clip(lower_bound=0) / pl.col("pass_att"),
                         rs=pl.col("athlete_id") + "|" + pl.col("season").cast(pl.String)))

def ratings(k_def):
    T = T0.with_columns(
        S_o=pl.col("pass_yds").sum().over(["season", "team"]),
        A_o=pl.col("pass_att").sum().over(["season", "team"]),
        ypa=pl.col("pass_yds") / pl.col("pass_att"))
    T = T.with_columns(off_loo=(pl.col("S_o") - pl.col("pass_yds") + K_OFF * m)
                               / (pl.col("A_o") - pl.col("pass_att") + K_OFF))
    T = T.with_columns(exc=pl.col("ypa") - pl.col("off_loo"))
    D = (T.group_by(["season", "opponent"])
           .agg(W=(pl.col("pass_att") * pl.col("exc")).sum(), A_d=pl.col("pass_att").sum()))
    T = T.join(D, on=["season", "opponent"], how="left")
    return T.with_columns(def_loo=(pl.col("W") - pl.col("pass_att") * pl.col("exc"))
                                  / (pl.col("A_d") - pl.col("pass_att") + k_def))


def prep(df, g):
    w = pl.col("pass_att")
    return df.with_columns(
        xd=pl.col("def_loo") - (pl.col("def_loo") * w).sum().over(g) / w.sum().over(g),
        yd=pl.col("y") - (pl.col("y") * w).sum().over(g) / w.sum().over(g),
        cl=pl.col("team") + "|" + pl.col("season").cast(pl.String))


def sums(df):
    return df.group_by("cl").agg(
        sxy=(pl.col("pass_att") * pl.col("xd") * pl.col("yd")).sum(),
        sxx=(pl.col("pass_att") * pl.col("xd") ** 2).sum())


def ci(v):
    return f"({np.percentile(v, 2.5):.3f} to {np.percentile(v, 97.5):.3f})"


def run(k_def, label):
    T = ratings(k_def).select("game_id", "team", "season", "pass_att", "ypa", "def_loo")
    P = (P0.join(T.select("game_id", "team", "ypa", "def_loo"), on=["game_id", "team"], how="inner")
           .drop_nulls("def_loo"))
    for lab, fbs_only in (("all", False), ("FBS only", True)):
        Pr = P.filter(pl.col("own_class") == "fbs") if fbs_only else P
        ts = Pr.select("season", "team").unique()
        Tt = (T.join(ts, on=["season", "team"], how="inner")
                .with_columns(y=pl.col("ypa")).drop_nulls("def_loo"))
        R, Tm = sums(prep(Pr, "rs")), sums(prep(Tt, ["team", "season"]))
        cls = sorted(set(R["cl"].to_list()) | set(Tm["cl"].to_list()))
        ix = {c: i for i, c in enumerate(cls)}
        n = len(cls)
        a_xy, a_xx, b_xy, b_xx = (np.zeros(n) for _ in range(4))
        for c, sxy, sxx in R.iter_rows():
            a_xy[ix[c]], a_xx[ix[c]] = sxy, sxx
        for c, sxy, sxx in Tm.iter_rows():
            b_xy[ix[c]], b_xx[ix[c]] = sxy, sxx
        beta_r, beta_t = a_xy.sum() / a_xx.sum(), b_xy.sum() / b_xx.sum()
        rng = np.random.default_rng(0)
        br, bt, rr = [], [], []
        for _ in range(N_BOOT):
            s = rng.integers(0, n, n)
            x1, x2 = a_xy[s].sum() / a_xx[s].sum(), b_xy[s].sum() / b_xx[s].sum()
            br.append(x1); bt.append(x2); rr.append(x1 / x2)
        share = (Pr["pass_att"] * Pr["y"]).sum() / (Pr["pass_att"] * Pr["ypa"]).sum()
        print(f"{label} | {lab}: receiver-rows {Pr.height}, team-games {Tt.height}, team-seasons {n}")
        print(f"   team-level slope {beta_t:.3f} {ci(bt)} | receiver-level slope {beta_r:.3f} {ci(br)}")
        print(f"   pass-through {beta_r / beta_t:.3f} {ci(rr)} | average receiver share of team yards {share:.3f}")


run(K_PRIMARY, f"PRIMARY k={K_PRIMARY:.0f}")
for k in K_SENS:
    run(k, f"sensitivity k={k:.0f}")