"""
Global Configuration & Shared Utilities for the NFL WR Prospect Translation Pipeline
====================================================================================
Single source of truth for:
  1. Directory & Parquet file paths (works from repo root or `src/`).
  2. Season & cohort year windows (CFB pull range, NFL Draft range, LOOCV evaluation cutoff).
  3. Model freezing & inference flags (`USE_FROZEN_MODEL`, `FROZEN_MODEL_JSON_PATH`).
  4. Canonical name, college school, NFL team, and height normalizers.
  5. Shared model thresholds and uncertainty constants.
"""

from __future__ import annotations

import re
from pathlib import Path

# ==============================================================================
# 1. DIRECTORY & FILE PATHS
# ==============================================================================
_THIS_DIR = Path(__file__).resolve().parent
if (_THIS_DIR / "data").exists():
    ROOT_DIR = _THIS_DIR
elif (_THIS_DIR.parent / "data").exists():
    ROOT_DIR = _THIS_DIR.parent
else:
    ROOT_DIR = Path.cwd().resolve()

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BOX_DIR = RAW_DIR / "boxes"
CFBD_RAW_DIR = RAW_DIR / "cfbd"
CFBD_GAMES_DIR = CFBD_RAW_DIR / "games"
CFBD_BOXES_DIR = CFBD_RAW_DIR / "games_players"

# Ensure core directories exist when imported
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Remote Data URLs & API Settings
CFBD_BASE_URL: str = "https://api.collegefootballdata.com"
CFBD_API_RESERVE: int = 150
CFBD_EXTRA_WEEKS: dict[int, tuple[int, ...]] = {2020: (16, 17)}

COMBINE_URL: str = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.parquet"
COMBINE_PRO_DAY_URL: str = (
    "https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_pro_day.parquet"
)

# Standard Raw Parquet Paths
DRAFT_RAW_PATH = RAW_DIR / "draft.parquet"
COMBINE_RAW_PATH = RAW_DIR / "combine.parquet"
COMBINE_PRO_DAY_RAW_PATH = RAW_DIR / "combine_pro_day.parquet"
SNAPS_RAW_PATH = RAW_DIR / "snaps.parquet"
STATS_RAW_PATH = RAW_DIR / "stats.parquet"
CFB_GAMES_RAW_PATH = RAW_DIR / "games.parquet"
CFB_RECRUITING_RAW_PATH = RAW_DIR / "recruiting.parquet"

# Standard Processed Parquet Paths
GAMES_PROCESSED_PATH = PROCESSED_DIR / "games.parquet"
RECEIVER_GAMES_PATH = PROCESSED_DIR / "receiver_games.parquet"
TEAM_GAMES_PATH = PROCESSED_DIR / "team_games.parquet"
GAME_SPLIT_PATH = PROCESSED_DIR / "game_split.parquet"
RECEIVER_SEASONS_PATH = PROCESSED_DIR / "receiver_seasons.parquet"
PANEL_V2_PATH = PROCESSED_DIR / "panel_v2.parquet"
PANEL_V2_FEAT_PATH = PROCESSED_DIR / "panel_v2_feat.parquet"
TEAM_GAME_RATINGS_PATH = PROCESSED_DIR / "team_game_ratings.parquet"
LEGACY_RATINGS_PATH = PROCESSED_DIR / "college_prospect_ratings.parquet"
WR_LINKS_PATH = PROCESSED_DIR / "wr_links.parquet"
WR_LINKS_FINAL_PATH = PROCESSED_DIR / "wr_links_final.parquet"
NFL_DRAFT_PICKS_PATH = PROCESSED_DIR / "nfl_draft_picks.parquet"
NFL_STATS_PLAYER_PATH = PROCESSED_DIR / "nfl_stats_player.parquet"

# Frozen Model Artifact Paths
FROZEN_MODEL_JSON_PATH = PROCESSED_DIR / "frozen_wr_model.json"
FROZEN_MODEL_PKL_PATH = PROCESSED_DIR / "frozen_wr_model.pkl"

# Pipeline Output Paths
STAGE1_RANKINGS_PATH = PROCESSED_DIR / "stage1_prospect_rankings.parquet"
STAGE2_RANKINGS_PATH = PROCESSED_DIR / "stage2_team_fit_rankings.parquet"
RECENT_PROSPECTS_PATH = PROCESSED_DIR / "stage1_recent_prospects.parquet"
STAGE2_AUDIT_CSV_PATH = PROCESSED_DIR / "stage2_team_fit_audit.csv"
APP_HTML_OUT_PATH = ROOT_DIR / "index.html"


# ==============================================================================
# 2. GLOBAL SEASON, COHORT & FREEZE FLAGS
# ==============================================================================
# College football data window (2004 is the first season of CFBD player box scores
# and covers Calvin Johnson's full 2004–2006 Georgia Tech career).
CFB_START_YEAR: int = 2004
CFB_END_YEAR: int = 2026

# Include both FBS and FCS so small-school NFL draftees (Cooper Kupp, Christian Watson, etc.) are linked.
CFB_DIVISIONS: tuple[str, ...] = ("fbs", "fcs")

# NFL Draft classes covered in the product (2007–2026, starting with Calvin Johnson's 2007 class)
DRAFT_START_YEAR: int = 2007
DRAFT_END_YEAR: int = 2026

# Latest completed NFL regular season in `nfl_stats_player.parquet`
COMPLETED_NFL_SEASON: int = 2026
NFL_PULL_START_YEAR: int = DRAFT_START_YEAR - 1  # 2006 (needed for 2007 ex-ante team environments)

# Evaluation / Training cohort window (model weights strictly learned on 2013–2022)
# This keeps the model training pure while allowing 2007–2012 and 2023–2026 to be scored as inference!
EVAL_START_YEAR: int = 2013
EVAL_END_YEAR: int = 2022
HELDOUT_START_YEAR: int = EVAL_END_YEAR + 1

# MODEL FREEZE & INFERENCE FLAGS:
#   - FREEZE_ON_TRAIN: When True, training on `EVAL_START_YEAR..EVAL_END_YEAR` automatically
#     writes the learned weights, scaler, and quantile regressors to `FROZEN_MODEL_JSON_PATH`.
#   - USE_FROZEN_MODEL: Set to True once you have trained & frozen the model and want subsequent
#     runs to score prospects strictly through the saved frozen artifact without re-fitting.
#   - SCORE_UNDRAFTED_CLASS_YEAR: Optional int (e.g., `2027`). When set (not None), automatically
#     appends eligible FBS college WRs from `CFB_END_YEAR` who are not yet in `wr_links_final.parquet`
#     as pre-draft prospects for that draft year (`MIN_UNDRAFTED_SEASON_YDS = 550`).
FREEZE_ON_TRAIN: bool = True
USE_FROZEN_MODEL: bool = False
SCORE_UNDRAFTED_CLASS_YEAR: int | None = 2027
MIN_UNDRAFTED_SEASON_YDS: int = 550

# Maximum years prior to `draft_year` to search for a player's college seasons
CFB_LINK_LOOKBACK_YEARS: int = 5
THIN_COLLEGE_GAMES_THRESHOLD: int = 10


# ==============================================================================
# 3. PANEL & MODEL HYPERPARAMETERS
# ==============================================================================
# College game-panel construction parameters (`build_panel_v2.py` & `ratings_and_features.py`)
MIN_PASS_ATT_GAME: int = 10
MIN_REC_TRAIN: int = 5
MIN_REC_SENSITIVITY: tuple[int, ...] = (3, 5, 8)
PANEL_TRAIN_FRAC: float = 2.0 / 3.0
K_OFF_SHRINK: float = 100.0
K_DEF_SHRINK: float = 220.0

# Tweedie M2 reliability check & export parameters (`model_checks.py`)
TWEEDIE_POWER: float = 1.5
TWEEDIE_KAPPA_REF: float = 200.0
TWEEDIE_KAPPA_CTX: float = 10.0
TWEEDIE_KC: float = 100.0
TWEEDIE_KD: float = 300.0

# Stage 1 Empirical Weight Learning Prior & Regularization
# Anchors weight distribution so Final-Season Ascension/Efficiency (`M2_Final_Eff`),
# Multi-Season Career Consistency (`M2_Career`), and Complete 3-Level Dominator (`SOS_Pie_Dominator`)
# drive >=58% of the skill index, preventing `Underclass_Freshman_Bonus` from over-inflating
# 1-year wonders or early-peak stagnators (e.g., Devin Thomas, Sidney Rice, Dwayne Jarrett).
STAGE1_FEATURE_NAMES: list[str] = [
    "M2_Final_Eff",
    "M2_Career",
    "Underclass_Freshman_Bonus",
    "SOS_Competition",
    "Teammate_Pie_Credit",
    "SOS_Pie_Dominator",
    "Consistency_Resilience",
]
STAGE1_PRIOR_WEIGHTS: tuple[float, ...] = (0.26, 0.23, 0.16, 0.10, 0.11, 0.09, 0.05)
STAGE1_NNLS_LAMBDA: float = 180.0

# Pro Day 40-yard dash adjustment (seconds added when backfilling a missing
# electronic Combine 40 with a Pro Day 40; set to 0.03s standard convention)
PRO_DAY_FORTY_OFFSET_SEC: float = 0.03

# Missing-40 handling:
#   - MISSING_FORTY_MED_PEN: median penalty for healthy skips (keep 0.0 unless pick-controlled CI excludes 0).
#   - MISSING_FORTY_SPREAD: widens p10/p90 interval for prospects without a 40 time.
MISSING_FORTY_MED_PEN: float = 0.0
MISSING_FORTY_SPREAD: float = 0.06

# 3-Year Cumulative NFL Receiving Yard Thresholds
STARTER_YDS_THRESHOLD: float = 1500.0
BOOM_YDS_THRESHOLD: float = 2500.0
BUST_YDS_THRESHOLD: float = 500.0


# ==============================================================================
# 4. CANONICAL NAME, SCHOOL & NFL TEAM NORMALIZERS
# ==============================================================================
NAME_SUFFIXES: set[str] = {"jr", "sr", "ii", "iii", "iv", "v"}

# Known NFL Draft nickname -> (CFBD legal name, CFBD school) overrides
PLAYER_NAME_OVERRIDES: dict[str, tuple[str, str]] = {
    "Bub Means": ("Jerrod Means", "Pittsburgh"),
    "Devin Fuller": ("Devin Lewis Fuller", "UCLA"),
    "Gabe Davis": ("Gabriel Davis", "UCF"),
    "Mike Thomas": ("Michael Thomas", "Southern Miss"),
    "Pat Bryant": ("Patrick Bryant", "Illinois"),
    "Scotty Miller": ("Scott Miller", "Bowling Green"),
    "Shaquelle Evans": ("Shaq Evans", "UCLA"),
    "Tank Dell": ("Nathaniel Dell", "Houston"),
    "Tutu Atwell": ("Chatarius Atwell", "Louisville"),
    "Walt Powell": ("Walter Powell", "Murray State"),
}

# Maps NFL/PFR college names (after punctuation & 'st' -> 'state' cleanup) to CFBD team names
SCHOOL_ALIASES: dict[str, str] = {
    "mississippi": "ole miss",
    "central florida": "ucf",
    "miami fl": "miami",
    "ala birmingham": "uab",
    "connecticut": "uconn",
    "hawaii": "hawai i",
    "north carolina state": "nc state",
    "middle tenn state": "middle tennessee",
    "la monroe": "ul monroe",
    "louisiana monroe": "ul monroe",
    "la lafayette": "louisiana",
    "louisiana lafayette": "louisiana",
    "boston col": "boston college",
    "southern cal": "usc",
    "southern california": "usc",
    "texas christian": "tcu",
    "brigham young": "byu",
    "nevada las vegas": "unlv",
    "texas el paso": "utep",
    "texas san antonio": "utsa",
    "southern methodist": "smu",
    "southern mississippi": "southern miss",
    "pittsburgh": "pitt",
    "massachusetts": "umass",
    "pennsylvania": "penn",
    "northern illinois": "northern illinois",
    "western michigan": "western michigan",
    "central michigan": "central michigan",
    "eastern michigan": "eastern michigan",
    "eastern washington": "eastern washington",
    "north dakota state": "north dakota state",
    "south dakota state": "south dakota state",
    "northern iowa": "northern iowa",
    "western illinois": "western illinois",
    "southern illinois": "southern illinois",
    "stephen f austin": "stephen f austin",
    "sam houston state": "sam houston",
    "appalachian state": "app state",
    "georgia tech": "georgia tech",
}

NFL_TEAM_NORM: dict[str, str] = {
    "OAK": "LV", "LVR": "LV", "RAIDERS": "LV",
    "SD": "LAC", "SDG": "LAC",
    "STL": "LAR", "LAR": "LAR", "LA": "LAR",
    "WAS": "WAS", "WSH": "WAS",
    "JAC": "JAX",
    "GNB": "GB", "KAN": "KC", "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB",
    "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "ARZ": "ARI",
}


def norm_player_name(s: str | None) -> str:
    """
    Normalizes a player name to space-separated lowercase words with suffixes stripped.
    Example: 'Marvin Harrison Jr.' -> 'marvin harrison'
    """
    if not s:
        return ""
    cleaned = re.sub(r"[^a-z ]", "", str(s).lower().replace("-", " ").replace(".", "").replace("'", ""))
    return " ".join(w for w in cleaned.split() if w not in NAME_SUFFIXES)


def norm_player_key(s: str | None) -> str:
    """
    Normalizes a player name to a compact alphanumeric key without spaces.
    Example: 'DK Metcalf' -> 'dkmetcalf'
    """
    return norm_player_name(s).replace(" ", "")


def extract_last_name(s: str | None) -> str:
    """
    Extracts the normalized surname (last token after stripping suffixes).
    """
    parts = norm_player_name(s).split()
    return parts[-1] if parts else ""


def norm_school(s: str | None) -> str:
    """
    Normalizes college/university names across PFR, NFL Draft, and CFBD.
    Example: 'Miami (FL)' -> 'miami', 'N.C. State' -> 'nc state', 'Mississippi' -> 'ole miss'
    """
    if not s:
        return ""
    k = str(s).lower().replace(".", "").replace("&", "and")
    k = re.sub(r"\bst\b", "state", k)
    k = re.sub(r"[^a-z0-9]+", " ", k).strip()
    return SCHOOL_ALIASES.get(k, k)


def norm_nfl_team(t: str | None) -> str:
    """
    Normalizes historical and PFR/nflverse NFL team abbreviations to modern 32-team codes.
    """
    if not t:
        return "UNK"
    t_up = str(t).strip().upper()
    return NFL_TEAM_NORM.get(t_up, t_up)


def parse_height_inches(h_raw) -> float | None:
    """
    Parses Combine/Pro-Day height formats ('6-2', '74', '6.167') into total inches.
    """
    if h_raw is None:
        return None
    h_str = str(h_raw).strip()
    if "-" in h_str:
        parts = h_str.split("-")
        try:
            return float(parts[0]) * 12.0 + float(parts[1])
        except Exception:
            return None
    try:
        val = float(h_str)
        return val if val > 50 else val * 12.0
    except Exception:
        return None