import warnings
warnings.filterwarnings("ignore")

import os
import json
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet

# ─── Config ───────────────────────────────────────────────────────────────────
SHEET_ID   = "1RKyeb4CfU4wACUKqpbiR_--PJJ9KKO-LH-Ov9e6zNQs"
PITCHER_POSITIONS = {"P", "SP", "RP", "Starting Pitcher", "Relief Pitcher",
                     "Right Hand Pitcher", "Left Hand Pitcher", "SC", "PC",
                     "Starting pitcher", "starting pitcher"}

# ─── CI Tiers ─────────────────────────────────────────────────────────────────
CI_TIERS = [
    (None, 285,  "< 285"),
    (285,  315,  "285–315"),
    (315,  None, "315+"),
]

def ci_tier_label(ci):
    if pd.isna(ci): return "—"
    for lo, hi, label in CI_TIERS:
        if (lo is None or ci >= lo) and (hi is None or ci < hi):
            return label
    return "315+"

def ci_tier_index(ci):
    if pd.isna(ci): return -1
    for i, (lo, hi, label) in enumerate(CI_TIERS):
        if (lo is None or ci >= lo) and (hi is None or ci < hi):
            return i
    return len(CI_TIERS) - 1

def ci_next_tier_target(ci):
    """Returns (next_tier_label, target_ci) or None if already at top."""
    idx = ci_tier_index(ci)
    if idx < 0 or idx >= len(CI_TIERS) - 1:
        return None
    lo, hi, label = CI_TIERS[idx + 1]
    return label, lo  # next tier lower bound is the target

def lbs_to_target(ci, mass_kg, target_ci, penalty=0.03):
    """
    How many lbs needed to reach target_ci given current CI/kg and 3% penalty?
    Formula: target_ci = (ci/mass_kg * (1 - penalty)) * (mass_kg + gain_kg)
    Solving: gain_kg = target_ci / (ci/mass_kg * (1-penalty)) - mass_kg
    """
    if pd.isna(ci) or pd.isna(mass_kg) or mass_kg == 0 or ci == 0:
        return np.nan
    ci_per_kg_adj = (ci / mass_kg) * (1 - penalty)
    if ci_per_kg_adj <= 0:
        return np.nan
    new_mass_kg = target_ci / ci_per_kg_adj
    gain_kg     = new_mass_kg - mass_kg
    return gain_kg * 2.20462  # convert to lbs

def weight_gain_classification(lbs):
    if pd.isna(lbs) or lbs <= 0:
        return "At or above target"
    if lbs < 10:
        return "Standard Off-Season"
    if lbs < 20:
        return "Needs Strength Camp"
    return "Needs Development Year"

WEIGHT_CLASS_COLORS = {
    "At or above target":   "#4CAF82",
    "Standard Off-Season":  "#4CAF82",
    "Needs Strength Camp":  "#E2C188",
    "Needs Development Year": "#BA0C2F",
}

# ─── Position groups ──────────────────────────────────────────────────────────
PITCHERS   = {"SP", "RHP", "LHP", "RP", "TWP"}
CATCHERS   = {"C"}
INFIELDERS = {"SS", "3B", "2B", "1B"}
OUTFIELDERS= {"CF", "LF", "RF"}

def pos_group(pos):
    if pos in PITCHERS:    return "Pitcher"
    if pos in CATCHERS:    return "Catcher"
    if pos in INFIELDERS:  return "Infielder"
    if pos in OUTFIELDERS: return "Outfielder"
    return "Unknown"

def programming_category(ci, p1_ci):
    """Classify athlete into in-house programming tier."""
    try:
        ci    = float(ci)
        p1_ci = float(p1_ci)
    except (TypeError, ValueError):
        return "Unclassified"
    if ci >= 285:
        return "High-High" if p1_ci >= 195 else "High-Low"
    return "Low"

PROG_COLORS = {
    "High-High":    "#4CAF82",
    "High-Low":     "#E2C188",
    "Low":          "#BA0C2F",
    "Unclassified": "#9AAAC0",
}
PROG_DESC = {
    "High-High": "CI ≥ 285 & P1 CI ≥ 195 — advanced program",
    "High-Low":  "CI ≥ 285 & P1 CI < 195 — high base, develop P1",
    "Low":       "CI < 285 — foundational program",
}

# ─── Colors ───────────────────────────────────────────────────────────────────
NAV   = "#11225A"
RED   = "#BA0C2F"
GOLD  = "#E2C188"
SURF  = "#F7F8FA"
BORD  = "#E8ECF0"
GREEN = "#4CAF82"

ARCHETYPE_COLORS = {
    "Normal":              "#4CAF82",
    "Long Loader":         "#11225A",
    "Front-Loaded Driver": "#BA0C2F",
    "Shallow Late-Driver": "#6b7fa3",
    "Unclassified":        "#9AAAC0",
}

BASE_STRATEGY_FEATURES = [
    "Eccentric Duration", "Concentric Duration", "Braking Phase Duration",
    "Countermovement Depth", "Concentric Impulse-100ms",
]
FEATURE_DIRECTION = {
    "Eccentric Duration":       ("unusually long loading phase",            "unusually short loading phase"),
    "Concentric Duration":      ("unusually long push-off",                 "unusually short / explosive push-off"),
    "Braking Phase Duration":   ("unusually slow load-to-drive transition", "unusually fast load-to-drive transition"),
    "Countermovement Depth":    ("unusually deep dip",                      "unusually shallow dip"),
    "Concentric Impulse-100ms": ("unusually high early drive",              "unusually low early drive"),
    "CI100ms_to_TotalCI_Ratio": ("front-loaded impulse strategy",           "back-loaded impulse strategy"),
}
SHORT_NAMES = {
    "Eccentric Duration":        "Ecc Duration",
    "Concentric Duration":       "Conc Duration",
    "Braking Phase Duration":    "Braking Dur",
    "Countermovement Depth":     "CM Depth",
    "Concentric Impulse-100ms":  "CI 100ms",
    "CI100ms_to_TotalCI_Ratio":  "CI100:TotalCI",
}

_PLOT_BASE = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor=SURF,
    font=dict(family="Arial, sans-serif", color=NAV),
)

def _layout(**kw):
    m = dict(_PLOT_BASE); m.update(kw); return m

# ─── Helpers ──────────────────────────────────────────────────────────────────
def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a.div(b).replace([np.inf, -np.inf], np.nan)

def pct_rank(series):
    return series.rank(pct=True) * 100

def robust_z(series):
    series = pd.to_numeric(series, errors="coerce")
    med = series.median()
    mad = np.median(np.abs(series.dropna() - med)) if series.notna().any() else np.nan
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (series - med) / mad

def scaled_0_100(values):
    arr = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if np.isclose(lo, hi): return np.repeat(50.0, len(arr))
    return (arr - lo) / (hi - lo) * 100

def fmt(x, d=1, s=""):
    return "—" if pd.isna(x) else f"{x:.{d}f}{s}"

def fmt_height(cm):
    """Convert cm to feet-inches string, e.g. 185.4 to 6ft 1in"""
    if pd.isna(cm): return "—"
    try:
        total_in = float(cm) / 2.54
        feet     = int(total_in // 12)
        inches   = int(round(total_in % 12))
        if inches == 12:
            feet += 1; inches = 0
        return f"{feet}\'{inches}\""
    except Exception:
        return "—"

def fmt_mass(kg):
    """kg → lbs"""
    if pd.isna(kg): return "—"
    try: return f"{float(kg) * 2.20462:.1f} lbs"
    except: return "—"

def fmt_wingspan(cm):
    """cm → feet-inches (same as height)"""
    return fmt_height(cm)

def fmt_wingspan_adv(cm):
    """cm difference → inches"""
    if pd.isna(cm): return "—"
    try: return f'{float(cm) / 2.54:.1f}"'
    except: return "—"

def fmt_bwht(kg, cm):
    """kg/cm ratio → lbs/inch"""
    if pd.isna(kg) or pd.isna(cm): return "—"
    try: return f"{(float(kg) * 2.20462) / (float(cm) / 2.54):.3f}"
    except: return "—"

def delta_html(val, d=1, invert=False):
    if pd.isna(val): return '<span style="color:#9AAAC0">—</span>'
    good = (val < 0) if invert else (val > 0)
    col  = GREEN if good else RED
    sign = "▲ +" if val > 0 else "▼ "
    return f'<span style="color:{col};font-weight:600">{sign}{val:.{d}f}</span>'

# ─── Archetype helpers ────────────────────────────────────────────────────────
def label_archetype(row, all_rz_cols):
    z_ecc   = float(row.get("rz_Eccentric Duration", 0) or 0)
    z_dep   = float(row.get("rz_Countermovement Depth", 0) or 0)
    z_ci100 = float(row.get("rz_Concentric Impulse-100ms", 0) or 0)
    z_ratio = float(row.get("rz_CI100ms_to_TotalCI_Ratio", 0) or 0)
    dist    = float(row.get("strategy_distance_score", 50) or 50)
    all_z   = [abs(float(row.get(c, 0) or 0)) for c in all_rz_cols]
    if dist < 50 and sum(z < 0.7 for z in all_z) >= 5: return "Normal"
    if z_dep > 1.0 and z_ecc > 1.0:                    return "Long Loader"
    if z_ci100 > 0.6 and z_ratio > 0.5:                return "Front-Loaded Driver"
    if z_dep < -0.7 and z_ratio < -0.5:                return "Shallow Late-Driver"
    return "Unclassified"

def explain_strategy(rz_dict, feats):
    parts = []
    for f in feats:
        z = rz_dict.get(f"rz_{f}", 0)
        if abs(z) < 0.8: continue
        pos, neg = FEATURE_DIRECTION.get(f, (f"high {f}", f"low {f}"))
        q = "very" if abs(z) >= 1.5 else "notably"
        parts.append(f"{q} {pos if z > 0 else neg}")
    return "; ".join(parts) if parts else "moderate profile — worth follow-up"

# ─── Google Sheets loader ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading data from Google Sheets…")
def load_data(_v=3):  # bump _v to bust cache
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds_env = os.environ.get("GOOGLE_CREDENTIALS")
    if creds_env:
        creds = Credentials.from_service_account_info(json.loads(creds_env), scopes=scopes)
    else:
        creds = Credentials.from_service_account_file("service_account.json", scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ.get("GOOGLE_SHEET_ID", SHEET_ID))

    fp_raw     = pd.DataFrame(sh.worksheet("Force Plate").get_all_records())
    isak_raw   = pd.DataFrame(sh.worksheet("ISAK").get_all_records())
    sprint_raw = pd.DataFrame(sh.worksheet("Sprint").get_all_records())
    pos_raw    = pd.DataFrame(sh.worksheet("Positions").get_all_records())

    for df in [fp_raw, isak_raw, sprint_raw]:
        df["playerID"] = df["playerID"].astype(str).str.strip()
        df["Year"]     = pd.to_numeric(df["Year"], errors="coerce")

    num_fp = ["Eccentric Duration","Concentric Duration","Braking Phase Duration",
              "Countermovement Depth","Concentric Impulse","Concentric Impulse-100ms",
              "P1 Concentric Impulse","Jump Height (Flight Time) in Inches",
              "RSI-modified","Peak Power / BM"]
    for c in num_fp:
        if c in fp_raw.columns:
            fp_raw[c] = pd.to_numeric(fp_raw[c], errors="coerce")

    for c in ["Mass","Height","Seated Height","Wingspan"]:
        if c in isak_raw.columns:
            isak_raw[c] = pd.to_numeric(isak_raw[c], errors="coerce")

    for c in ["10yd Split","20yd Split","30yd Split"]:
        if c in sprint_raw.columns:
            sprint_raw[c] = pd.to_numeric(sprint_raw[c], errors="coerce")

    # Merge all on playerID + Year
    df = fp_raw.merge(isak_raw.drop(columns=["athleteName"], errors="ignore"),
                      on=["playerID","Year"], how="outer")
    df = df.merge(sprint_raw.drop(columns=["athleteName"], errors="ignore"),
                  on=["playerID","Year"], how="outer")

    # Merge positions
    if "playerID" in pos_raw.columns and "Position" in pos_raw.columns:
        pos_raw["playerID"] = pos_raw["playerID"].astype(str).str.strip()
        df = df.merge(pos_raw[["playerID","Position"]].drop_duplicates("playerID"),
                      on="playerID", how="left")
    else:
        df["Position"] = ""

    # Fill athlete name gaps
    name_map = (fp_raw[["playerID","athleteName"]].dropna()
                .drop_duplicates("playerID").set_index("playerID")["athleteName"].to_dict())
    name_map.update(isak_raw[["playerID","athleteName"]].dropna()
                    .drop_duplicates("playerID").set_index("playerID")["athleteName"].to_dict())
    name_map.update(sprint_raw[["playerID","athleteName"]].dropna()
                    .drop_duplicates("playerID").set_index("playerID")["athleteName"].to_dict())
    df["athleteName"] = df["athleteName"].where(
        df["athleteName"].notna(), df["playerID"].map(name_map))

    # Position lookup from FP School Type
    pos_map = (fp_raw[["playerID","School Type"]].dropna()
               .drop_duplicates("playerID").set_index("playerID")["School Type"].to_dict())
    if "School Type" not in df.columns:
        df["School Type"] = df["playerID"].map(pos_map)
    else:
        df["School Type"] = df["School Type"].where(
            df["School Type"].notna() & (df["School Type"] != ""),
            df["playerID"].map(pos_map))

    df = df[df["athleteName"].notna() &
            (df["athleteName"].str.strip() != "") &
            (df["athleteName"].str.lower() != "nan")].copy()
    df = df.sort_values(["athleteName","Year"]).reset_index(drop=True)

    return df

# ─── Model pipeline ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building scores…", hash_funcs={pd.DataFrame: lambda x: x.shape})
def build_scores(_df,
                 w_ci=0.35, w_sprint=0.30, w_rsi=0.15, w_pp=0.20,
                 w_ci_ns=0.45, w_rsi_ns=0.20, w_pp_ns=0.35,
                 wp_peakpow=0.25, wp_height=0.25, wp_bmi=0.20,
                 wp_school=0.15, wp_wingspan=0.15):
    df = _df.copy()

    # ── Sanity bounds — null out physiologically impossible values ─────────────
    BOUNDS = {
        "Concentric Impulse":                  (100, 400),
        "RSI-modified":                        (0.1, 1.5),
        "Peak Power / BM":                     (20,  120),
        "30yd Split":                          (3.3, 5.0),
        "10yd Split":                          (1.3, 2.2),
        "20yd Split":                          (2.4, 3.8),
        "Jump Height (Flight Time) in Inches": (8,   40),
        "Concentric Impulse-100ms":            (30,  250),
        "P1 Concentric Impulse":               (30,  250),
        "Eccentric Duration":                  (0.1, 1.5),
        "Concentric Duration":                 (0.05,0.8),
        "Braking Phase Duration":              (0.05,0.8),
        "Countermovement Depth":               (-120, -5),
        "Height":                              (155, 215),
        "Mass":                                (55,  175),
        "Wingspan":                            (155, 230),
    }
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            bad = (df[col] < lo) | (df[col] > hi)
            df.loc[bad, col] = np.nan

    # ── CMJ strategy features ──────────────────────────────────────────────────
    df["CI100ms_to_TotalCI_Ratio"] = safe_div(
        df["Concentric Impulse-100ms"], df["Concentric Impulse"])
    strategy_features = BASE_STRATEGY_FEATURES + ["CI100ms_to_TotalCI_Ratio"]

    for c in strategy_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    # Robust z-scores
    for c in strategy_features:
        df[f"rz_{c}"] = robust_z(df[c])
    all_rz_cols = [f"rz_{f}" for f in strategy_features]

    # Strategy distance — identical pipeline to international scouting app
    # StandardScaler → PCA (whitened) → MinCovDet Mahalanobis
    X_s      = df[strategy_features].fillna(0).values.astype(float)
    scaler_s = StandardScaler()
    X_sc     = scaler_s.fit_transform(X_s)
    ncomp    = min(len(strategy_features), max(3, min(6, len(df) - 1)))
    pca_s    = PCA(n_components=ncomp, whiten=True, random_state=42)
    X_pca    = pca_s.fit_transform(X_sc)
    try:
        mcd = MinCovDet(random_state=42, support_fraction=0.8).fit(X_pca)
        df["strategy_distance_raw"] = np.sqrt(mcd.mahalanobis(X_pca))
    except Exception:
        # Fallback to standard covariance if MinCovDet fails
        cov     = np.cov(X_pca, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        mu_pca  = X_pca.mean(axis=0)
        diff    = X_pca - mu_pca
        dist2   = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        df["strategy_distance_raw"] = np.sqrt(np.abs(dist2))
    df["strategy_distance_score"] = pct_rank(df["strategy_distance_raw"])

    # Archetypes
    df["archetype"]  = df.apply(lambda r: label_archetype(r.to_dict(), all_rz_cols), axis=1)
    df["why_flagged"] = df.apply(lambda r: explain_strategy(
        {f"rz_{f}": r.get(f"rz_{f}", 0) for f in strategy_features}, strategy_features), axis=1)

    # ── Athleticism Score Score ──────────────────────────────────────────────────
    # Position group
    df["pos_group"] = df["Position"].astype(str).map(pos_group)

    # ── Helper: compute percentiles within a given mask ────────────────────────
    def add_pct_group(df, col, out_col, invert=False, mask=None):
        df[out_col] = np.nan
        sub = df[mask] if mask is not None else df
        has = sub[col].notna()
        if has.any():
            r = pct_rank(sub.loc[has, col])
            df.loc[sub.index[has], out_col] = 100 - r if invert else r

    # ── All-time percentiles ───────────────────────────────────────────────────
    # CI percentile (higher = better)
    df["ci_pct_alltime"]     = pct_rank(df["Concentric Impulse"])
    # RSI percentile (higher = better)
    df["rsi_pct_alltime"]    = pct_rank(df["RSI-modified"])
    # Peak Power / BM percentile (higher = better) — always available
    df["pp_pct_alltime"]     = pct_rank(df["Peak Power / BM"])
    # Sprint percentile — ranked only among athletes who actually ran
    sprint_mask = df["30yd Split"].notna()
    df["sprint_pct_alltime"] = np.nan
    if sprint_mask.any():
        df.loc[sprint_mask, "sprint_pct_alltime"] = (
            100 - pct_rank(df.loc[sprint_mask, "30yd Split"])
        )

    # ── Position-group percentiles (Pitcher / Catcher / Infielder / Outfielder) ──
    for grp in ["Pitcher", "Catcher", "Infielder", "Outfielder"]:
        mask = df["pos_group"] == grp
        for col, suffix, inv in [
            ("Concentric Impulse", "ci",     False),
            ("RSI-modified",       "rsi",    False),
            ("Peak Power / BM",    "pp",     False),
            ("30yd Split",         "sprint", True),
        ]:
            out = f"{suffix}_pct_{grp.lower()}"
            df[out] = np.nan
            sub = df[mask]
            if col == "30yd Split":
                has = sub[col].notna()
                if has.any():
                    r = pct_rank(sub.loc[has, col])
                    df.loc[sub.index[has], out] = 100 - r
            else:
                has = sub[col].notna()
                if has.any():
                    df.loc[sub.index[has], out] = pct_rank(sub.loc[has, col])

    # Position-group quality score
    def pos_group_aq(row):
        grp = row.get("pos_group", "Unknown")
        if grp == "Unknown": return np.nan
        g = grp.lower()
        ci_p  = row.get(f"ci_pct_{g}", np.nan)
        rsi_p = row.get(f"rsi_pct_{g}", np.nan)
        pp_p  = row.get(f"pp_pct_{g}", np.nan)
        spr_p = row.get(f"sprint_pct_{g}", np.nan)
        if pd.notna(spr_p):
            return (w_ci * (ci_p or 50) + w_sprint * spr_p +
                    w_rsi * (rsi_p or 50) + w_pp * (pp_p or 50))
        else:
            return (w_ci_ns * (ci_p or 50) + w_rsi_ns * (rsi_p or 50) +
                    w_pp_ns * (pp_p or 50))

    df["aq_pos_raw"]   = df.apply(pos_group_aq, axis=1)
    df["aq_pos_score"] = np.nan
    for grp in ["Pitcher", "Catcher", "Infielder", "Outfielder"]:
        idx = df["pos_group"] == grp
        if idx.any():
            df.loc[idx, "aq_pos_score"] = scaled_0_100(df.loc[idx, "aq_pos_raw"].values)

    # Two scoring modes:
    # With sprint:    w_ci*CI + w_sprint*Sprint + w_rsi*RSI + w_pp*PeakPower
    # Without sprint: w_ci_ns*CI + w_rsi_ns*RSI + w_pp_ns*PeakPower
    def aq_raw(row):
        if pd.notna(row["sprint_pct_alltime"]):
            return (w_ci     * row["ci_pct_alltime"] +
                    w_sprint * row["sprint_pct_alltime"] +
                    w_rsi    * row["rsi_pct_alltime"] +
                    w_pp     * row["pp_pct_alltime"])
        else:
            return (w_ci_ns  * row["ci_pct_alltime"] +
                    w_rsi_ns * row["rsi_pct_alltime"] +
                    w_pp_ns  * row["pp_pct_alltime"])

    df["athlete_quality_raw"] = df.apply(aq_raw, axis=1)
    df["athlete_quality_score"] = scaled_0_100(df["athlete_quality_raw"].values)

    # Per-year versions
    for col, pct_col, invert in [
        ("Concentric Impulse",  "ci_pct_yr",     False),
        ("RSI-modified",        "rsi_pct_yr",     False),
        ("Peak Power / BM",     "pp_pct_yr",      False),
        ("30yd Split",          "sprint_pct_yr",  True),
    ]:
        df[pct_col] = np.nan
        for yr, idx in df.groupby("Year").groups.items():
            s = df.loc[idx, col]
            has = s.notna()
            if has.any():
                r = pct_rank(s[has])
                df.loc[idx[has], pct_col] = 100 - r if invert else r
        # Non-runners stay NaN for sprint — handled in scoring below

    def aq_yr_raw(row):
        if pd.notna(row["sprint_pct_yr"]):
            return (w_ci     * row["ci_pct_yr"] +
                    w_sprint * row["sprint_pct_yr"] +
                    w_rsi    * row["rsi_pct_yr"] +
                    w_pp     * row["pp_pct_yr"])
        else:
            return (w_ci_ns  * row["ci_pct_yr"] +
                    w_rsi_ns * row["rsi_pct_yr"] +
                    w_pp_ns  * row["pp_pct_yr"])

    df["aq_yr_raw"] = df.apply(aq_yr_raw, axis=1)
    # Scale per year
    df["aq_score_yr"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "aq_score_yr"] = scaled_0_100(df.loc[idx, "aq_yr_raw"].values)

    # ── Potential Score ────────────────────────────────────────────────────────
    # Peak Power / BM percentile (higher = better)
    df["pp_pct"]      = df["pp_pct_alltime"]  # reuse alltime percentile
    # Height percentile (taller = better)
    df["height_pct"]  = pct_rank(df["Height"])
    # BMI-style (kg/cm) — leaner = better → invert
    df["bmi_raw"]     = safe_div(df["Mass"] * 2.20462, df["Height"] / 2.54)  # lbs/inch
    df["bmi_pct"]     = pct_rank(df["bmi_raw"])  # higher ratio = higher pct
    # School type: HS > 4-Year College > Junior College
    school_score_map  = {"High School": 100, "4-Year College": 60, "Junior College": 40}
    df["school_score"] = df["School Type"].map(school_score_map).fillna(50)
    # Wingspan advantage (wingspan - height; positive = longer reach)
    df["wingspan_advantage"] = df["Wingspan"] - df["Height"]
    df["wingspan_pct"]       = pct_rank(df["wingspan_advantage"])

    # Build potential — include wingspan only for pitchers
    def pot_score(row):
        # Collect available components — skip missing ones and redistribute weight
        def sv(key, default=50):
            v = row.get(key)
            try:
                f = float(v)
                return f if not np.isnan(f) else None
            except (TypeError, ValueError):
                return None

        components = {
            "pp_pct":       (sv("pp_pct"),       wp_peakpow),
            "height_pct":   (sv("height_pct"),   wp_height),
            "bmi_pct":      (sv("bmi_pct"),      wp_bmi),
            "school_score": (sv("school_score"), wp_school),
            "wingspan_pct": (sv("wingspan_pct"), wp_wingspan),
        }
        total_w = sum(w for _, (v, w) in components.items() if v is not None)
        if total_w == 0:
            return np.nan
        # Weighted average of available components, rescaled to full weight
        pot = sum(v * w for _, (v, w) in components.items() if v is not None) / total_w * 100
        return pot

    df["potential_raw"]   = df.apply(pot_score, axis=1)
    df["potential_score"] = scaled_0_100(df["potential_raw"].values)

    # Per-year potential
    df["potential_score_yr"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "potential_score_yr"] = scaled_0_100(df.loc[idx, "potential_raw"].values)

    # Overall rank (athlete quality only)
    # Rank within each year
    # Programming category
    df["programming_category"] = df.apply(
        lambda r: programming_category(
            r.get("Concentric Impulse", np.nan),
            r.get("P1 Concentric Impulse", np.nan)
        ), axis=1)

    df["overall_rank"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "overall_rank"] = (
            df.loc[idx, "athlete_quality_score"]
            .rank(ascending=False, method="min")
        )
    df["overall_rank"] = df["overall_rank"].astype("Int64")

    # ── CI tier classification ───────────────────────────────────────────────
    df["ci_tier"]       = df["Concentric Impulse"].apply(ci_tier_label)
    df["ci_tier_idx"]   = df["Concentric Impulse"].apply(ci_tier_index)

    # Weight needed to hit 315 and next tier
    TARGET_300 = 315.0
    def ci_pathway(row):
        ci      = row.get("Concentric Impulse", np.nan)
        mass_kg = row.get("Mass", np.nan)
        result  = {}
        # Lbs to 315
        if pd.notna(ci) and ci < TARGET_300 and pd.notna(mass_kg):
            result["lbs_to_315"]       = lbs_to_target(ci, mass_kg, TARGET_300)
            result["weight_class_315"] = weight_gain_classification(result["lbs_to_315"])
        elif pd.notna(ci) and ci >= TARGET_300:
            result["lbs_to_315"]       = 0.0
            result["weight_class_315"] = "At or above target"
        else:
            result["lbs_to_315"]       = np.nan
            result["weight_class_315"] = "—"
        # Lbs to next tier
        next_tier = ci_next_tier_target(ci) if pd.notna(ci) else None
        if next_tier and pd.notna(mass_kg):
            next_label, next_target    = next_tier
            result["next_tier_label"]  = next_label
            result["lbs_to_next_tier"] = lbs_to_target(ci, mass_kg, next_target)
            result["weight_class_next"]= weight_gain_classification(result["lbs_to_next_tier"])
        elif ci_tier_index(ci) == len(CI_TIERS) - 1:
            result["next_tier_label"]  = "Top tier"
            result["lbs_to_next_tier"] = 0.0
            result["weight_class_next"]= "At or above target"
        else:
            result["next_tier_label"]  = "—"
            result["lbs_to_next_tier"] = np.nan
            result["weight_class_next"]= "—"
        return pd.Series(result)

    pathway_df = df.apply(ci_pathway, axis=1)
    for col in pathway_df.columns:
        df[col] = pathway_df[col]

    return df, strategy_features, all_rz_cols

# ─── CSS ──────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] {{
    font-family: 'Source Sans 3', sans-serif;
    background-color: {SURF}; color: {NAV};
}}
h1,h2,h3 {{ font-family: 'Playfair Display', serif; color: {NAV}; }}
.block-container {{ padding-top: 1.5rem; max-width: 1400px; }}
div[data-testid="metric-container"] {{
    background: white; border: 1px solid {BORD};
    border-top: 3px solid {RED}; border-radius: 10px;
    padding: 14px 18px; box-shadow: 0 2px 8px rgba(17,34,90,0.06);
}}
div[data-testid="metric-container"] label {{
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b7fa3;
}}
div[data-testid="metric-container"] div[data-testid="metric-value"] {{
    font-family: 'Playfair Display', serif; font-size: 28px; color: {RED};
}}
.stTabs [data-baseweb="tab-list"] {{
    background: white; border-bottom: 2px solid {BORD}; gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7fa3; padding: 12px 28px;
    border: none; border-bottom: 3px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {RED} !important; border-bottom: 3px solid {RED} !important;
    background: white !important;
}}
.card {{
    background: white; border: 1px solid {BORD}; border-radius: 10px;
    padding: 18px 22px; box-shadow: 0 2px 8px rgba(17,34,90,0.06);
    margin-bottom: 16px;
}}
.card-red   {{ border-top: 4px solid {RED}; }}
.card-navy  {{ border-top: 4px solid {NAV}; }}
.card-gold  {{ border-top: 4px solid {GOLD}; }}
.card-green {{ border-top: 4px solid {GREEN}; }}
.label {{
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b7fa3; margin-bottom: 6px;
}}
.stat-row {{ display: flex; align-items: baseline; margin-bottom: 7px; font-size: 13px; }}
.stat-label {{ color: #6b7fa3; min-width: 160px; font-size: 12px; }}
.stat-val {{ font-weight: 600; color: {NAV}; }}
.arch-badge {{
    display: inline-block; color: white; font-size: 11px; font-weight: 700;
    padding: 3px 12px; border-radius: 20px; letter-spacing: 0.04em; margin-bottom: 8px;
}}
.rank-badge {{
    display: inline-block; background: {RED}; color: white;
    font-family: 'Playfair Display', serif; font-size: 13px; font-weight: 700;
    padding: 2px 10px; border-radius: 20px; margin-right: 8px;
}}
.why-text {{
    font-size: 13px; line-height: 1.7; color: {NAV};
    background: {SURF}; border-radius: 8px; padding: 10px 14px; margin-top: 6px;
}}
.grad-bar {{
    height: 5px;
    background: linear-gradient(90deg, {RED} 0%, {NAV} 60%, {GOLD} 100%);
    border-radius: 2px; margin-bottom: 0;
}}
.score-big {{
    font-family: 'Playfair Display', serif; font-size: 42px;
    font-weight: 900; line-height: 1;
}}
</style>
"""

# ─── Gauge chart ──────────────────────────────────────────────────────────────
def make_gauge(value, title, color=RED):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value if pd.notna(value) else 0,
        title={"text": title, "font": {"size": 12, "color": NAV, "family": "Arial"}},
        number={"font": {"size": 26, "color": color, "family": "Playfair Display, serif"},
                "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 9, "color": "#9AAAC0"}},
            "bar":  {"color": color, "thickness": 0.6},
            "bgcolor": SURF,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  33], "color": "#f0f3f8"},
                {"range": [33, 66], "color": "#e4eaf3"},
                {"range": [66, 100], "color": "#d8e0ed"},
            ],

        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="white", font=dict(family="Arial"))
    return fig

# ─── Radar chart ─────────────────────────────────────────────────────────────
def make_radar(row, label="Athlete"):
    def safe_pct(row, key, default=50.0):
        v = row.get(key, default)
        try:
            return float(v) if v is not None and str(v) != "<NA>" else default
        except (TypeError, ValueError):
            return default

    has_sprint = pd.notna(row.get("30yd Split")) if hasattr(row, "get") else False

    if has_sprint:
        cats = ["CI", "Sprint", "RSI-mod", "Peak Pwr", "Height"]
        vals = [
            safe_pct(row, "ci_pct_alltime"),
            safe_pct(row, "sprint_pct_alltime"),
            safe_pct(row, "rsi_pct_alltime"),
            safe_pct(row, "pp_pct_alltime"),
            safe_pct(row, "height_pct"),
        ]
    else:
        cats = ["CI", "RSI-mod", "Peak Pwr", "Height"]
        vals = [
            safe_pct(row, "ci_pct_alltime"),
            safe_pct(row, "rsi_pct_alltime"),
            safe_pct(row, "pp_pct_alltime"),
            safe_pct(row, "height_pct"),
        ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", name=label,
        line=dict(color=RED, width=2),
        fillcolor=f"rgba(186,12,47,0.15)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, color="#9AAAC0"),
                            gridcolor=BORD),
            angularaxis=dict(tickfont=dict(size=10, color=NAV)),
            bgcolor="white",
        ),
        showlegend=False,
        height=300, margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="white",
    )
    return fig

# ─── Strategy profile bar chart ───────────────────────────────────────────────
def make_profile(row, strat_feats):
    if hasattr(row, "to_dict"): row = row.to_dict()
    z_vals = [float(row.get(f"rz_{f}") or 0.0) for f in strat_feats]
    labels = [SHORT_NAMES.get(f, f) for f in strat_feats]
    colors = [RED if z >= 0 else NAV for z in z_vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=z_vals, marker_color=colors,
        text=[f"{z:+.2f}" for z in z_vals],
        textposition="outside", textfont=dict(color=NAV, size=11),
    ))
    fig.add_hline(y=0,    line_width=1.5, line_color=NAV)
    fig.add_hline(y=0.8,  line_dash="dot", line_color=RED, line_width=1)
    fig.add_hline(y=-0.8, line_dash="dot", line_color=RED, line_width=1)
    fig.update_layout(**_layout(
        title=dict(text="CMJ Strategy Profile", font=dict(size=13, color=NAV), x=0),
        yaxis_title="Robust z-score", xaxis_title="",
        height=320, margin=dict(l=30, r=20, t=50, b=100),
    ))
    return fig

# ─── Year-over-year trend ─────────────────────────────────────────────────────
def make_trend(player_df, col, label, invert=False):
    g = player_df.sort_values("Year").dropna(subset=[col])
    if len(g) < 2: return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["Year"], y=g[col], mode="lines+markers",
        line=dict(color=RED, width=2.5),
        marker=dict(size=8, color=RED, line=dict(width=1.5, color="white")),
    ))
    fig.update_layout(**_layout(
        title=dict(text=label, font=dict(size=12, color=NAV), x=0),
        height=180, margin=dict(l=30, r=10, t=36, b=30),
        xaxis=dict(tickmode="array", tickvals=g["Year"].tolist(),
                   tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed" if invert else True),
        showlegend=False,
    ))
    return fig

# ─── Scatter map ──────────────────────────────────────────────────────────────
def make_scatter(dff):
    fig = px.scatter(
        dff, x="athlete_quality_score", y="strategy_distance_score",
        color="ci_tier", hover_name="athleteName",
        hover_data={"Year": True, "athlete_quality_score": ":.1f",
                    "ci_tier": True, "weight_class_next": True,
                    "lbs_to_next_tier": ":.1f"},
        color_discrete_map={"< 285": "#BA0C2F", "285–315": "#E2C188", "315+": "#11225A"},
        labels={"athlete_quality_score": "Athleticism Score",
                "strategy_distance_score": "Strategy Distance"},
        height=480,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(**_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text="Athleticism vs Potential", font=dict(size=14, color=NAV), x=0),
        legend=dict(title="Archetype", font=dict(color=NAV)),
    ))
    return fig

# ─── App ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NationalsDraft · Athlete Scorecard",
    page_icon="⚾", layout="wide",
    initial_sidebar_state="expanded",
)

def check_password():
    if st.session_state.get("auth"): return True
    st.markdown(f"""
    <div style="max-width:380px;margin:80px auto;background:white;border-radius:12px;
        padding:36px 32px;box-shadow:0 4px 24px rgba(17,34,90,0.10);border-top:4px solid {RED}">
        <p style="font-size:10px;font-weight:600;letter-spacing:0.2em;color:{RED};margin-bottom:6px">
            WASHINGTON NATIONALS · DRAFT SCOUTING</p>
        <h2 style="font-family:'Playfair Display',serif;color:{NAV};margin:0 0 20px 0;font-size:22px">
            Athlete Scorecard</h2>
    </div>
    """, unsafe_allow_html=True)
    col = st.columns([1, 2, 1])[1]
    with col:
        pwd = st.text_input("Password", type="password",
                            label_visibility="collapsed", placeholder="Enter password")
        if st.button("Enter", use_container_width=True):
            if pwd == os.environ.get("DASHBOARD_PASSWORD", "NationalsDraft"):
                st.session_state.auth = True; st.rerun()
            else:
                st.error("Incorrect password.")
    return False

if not check_password(): st.stop()

st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="grad-bar"></div>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<p style="font-size:10px;font-weight:600;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:#6b7fa3">Data</p>', unsafe_allow_html=True)
    if st.button("↻ Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.markdown("---")
    st.markdown(f'<p style="font-size:10px;font-weight:600;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:#6b7fa3">Quality Weights (With Sprint)</p>',
                unsafe_allow_html=True)
    w_ci     = st.slider("Concentric Impulse",  0, 100, 35, 5, key="w_ci") / 100
    w_sprint = st.slider("30yd Sprint",          0, 100, 30, 5, key="w_sprint") / 100
    w_rsi    = st.slider("RSI-modified",         0, 100, 15, 5, key="w_rsi") / 100
    w_pp     = st.slider("Peak Power / BM",      0, 100, 20, 5, key="w_pp") / 100
    total_aq = w_ci + w_sprint + w_rsi + w_pp
    if not (0.99 < total_aq < 1.01):
        st.warning(f"Weights sum to {total_aq*100:.0f}% — should be 100%")

    st.markdown("---")
    st.markdown(f'<p style="font-size:10px;font-weight:600;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:#6b7fa3">Quality Weights (No Sprint)</p>',
                unsafe_allow_html=True)
    st.caption("Applied when a player has no sprint data")
    w_ci_ns  = st.slider("Concentric Impulse (no sprint)", 0, 100, 45, 5, key="w_ci_ns") / 100
    w_rsi_ns = st.slider("RSI-modified (no sprint)",       0, 100, 20, 5, key="w_rsi_ns") / 100
    w_pp_ns  = st.slider("Peak Power / BM (no sprint)",    0, 100, 35, 5, key="w_pp_ns") / 100
    total_ns = w_ci_ns + w_rsi_ns + w_pp_ns
    if not (0.99 < total_ns < 1.01):
        st.warning(f"No-sprint weights sum to {total_ns*100:.0f}% — should be 100%")

    wp_pp = 0.25; wp_ht = 0.25; wp_bmi = 0.20; wp_school = 0.15; wp_wings = 0.15

# ─── Load & score ─────────────────────────────────────────────────────────────
try:
    raw = load_data(_v=3)
    load_err = None
except Exception as e:
    raw = None; load_err = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"

if load_err:
    st.error("Could not load data.")
    st.code(load_err); st.stop()

df, strat_feats, all_rz_cols = build_scores(
    raw, w_ci, w_sprint, w_rsi, w_pp,
    w_ci_ns, w_rsi_ns, w_pp_ns,
    wp_pp, wp_ht, wp_bmi, wp_school, wp_wings)

# ─── Header ───────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown(
        f'<p style="font-size:10px;font-weight:600;letter-spacing:0.2em;color:{RED};margin-bottom:4px">'
        f'WASHINGTON NATIONALS · DRAFT SCOUTING</p>'
        f'<h1 style="margin:0 0 2px 0;font-size:clamp(22px,3vw,38px);line-height:1.1">'
        f'Athlete Scorecard</h1>', unsafe_allow_html=True)
with hc2:
    st.markdown(
        f'<div style="text-align:right;padding-top:12px">'
        f'<span style="font-family:\'Playfair Display\',serif;font-size:28px;font-weight:700;color:{RED}">'
        f'{df["playerID"].nunique()}</span>'
        f'<span style="font-size:12px;color:#6b7fa3;margin-left:4px">athletes</span><br>'
        f'<span style="font-family:\'Playfair Display\',serif;font-size:28px;font-weight:700;color:{NAV}">'
        f'{int(df["Year"].nunique())}</span>'
        f'<span style="font-size:12px;color:#6b7fa3;margin-left:4px">years</span></div>',
        unsafe_allow_html=True)
st.markdown('<hr style="margin:8px 0 0 0;border-color:#E8ECF0">', unsafe_allow_html=True)

tab_board, tab_card = st.tabs(["Leaderboard", "Athlete Scorecard"])

# =============================================================================
# TAB 1 — LEADERBOARD
# =============================================================================
with tab_board:

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3, f4, f5 = st.columns([2.5, 1, 1.2, 1.2, 1.2])
    with f1: search      = st.text_input("Search", placeholder="Name…", key="lb_search")
    with f2:
        yr_opts = ["All"] + sorted(df["Year"].dropna().unique().astype(int).tolist(), reverse=True)
        yr_sel  = st.selectbox("Year", yr_opts, key="lb_year")
    with f3:
        pos_grp_sel = st.selectbox("Position Group",
                        ["All","Pitcher","Catcher","Infielder","Outfielder"], key="lb_pos")
    with f4:
        st_sel = st.selectbox("School Type",
                    ["All"] + sorted(df["School Type"].dropna().unique()), key="lb_st")
    with f5:
        sort_by = st.selectbox("Sort by",
                    ["Athleticism Score", "Pos. Athleticism", "CI", "30yd Sprint"], key="lb_sort")

    dff = df.copy()
    if search:              dff = dff[dff["athleteName"].str.contains(search, case=False, na=False)]
    if yr_sel != "All":     dff = dff[dff["Year"] == int(yr_sel)]
    if pos_grp_sel != "All": dff = dff[dff["pos_group"] == pos_grp_sel]
    if st_sel != "All":     dff = dff[dff["School Type"] == st_sel]

    sort_col = {"Athleticism Score": "athlete_quality_score",
                "Pos. Athleticism":  "aq_pos_score",
                "CI":                "Concentric Impulse",
                "30yd Sprint":       "30yd Split"}[sort_by]
    dff = dff.sort_values(sort_col, ascending=(sort_by == "30yd Sprint"),
                          na_position="last").reset_index(drop=True)

    # ── Summary strip ─────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Athletes",    str(len(dff)))
    k2.metric("Median CI",   fmt(dff["Concentric Impulse"].median(), 1))
    k3.metric("Median RSI",  fmt(dff["RSI-modified"].median(), 3))
    k4.metric("Median PkPwr/BM", fmt(dff["Peak Power / BM"].median(), 1))
    k5.metric("Median 30yd", fmt(dff["30yd Split"].median(), 3, "s")
              if dff["30yd Split"].notna().any() else "—")

    # ── Table ─────────────────────────────────────────────────────────────────
    tbl = dff[["athleteName","Year","Position",
               "athlete_quality_score",
               "ci_tier","weight_class_next","lbs_to_next_tier",
               "weight_class_315","lbs_to_315"]].copy()

    tbl["lbs_to_next_tier"] = tbl["lbs_to_next_tier"].apply(
        lambda x: f"+{x:.1f} lbs" if pd.notna(x) and x > 0 else ("Top tier" if x == 0 else "—"))
    tbl["lbs_to_315"] = tbl["lbs_to_315"].apply(
        lambda x: f"+{x:.1f} lbs" if pd.notna(x) and x > 0 else ("✓" if x == 0 else "—"))

    tbl = tbl.rename(columns={
        "athleteName":"Athlete",
        "athlete_quality_score":"Athleticism",
        "ci_tier":"CI Tier",
        "weight_class_next":"To Next Tier Classification",
        "lbs_to_next_tier":"Lbs to Next Tier",
        "weight_class_315":"To 315 Classification",
        "lbs_to_315":"Lbs to 315",
    })
    tbl["Athleticism"] = tbl["Athleticism"].round(1)

    sel = st.dataframe(tbl, use_container_width=True, hide_index=True,
                       on_select="rerun", selection_mode="single-row", key="lb_tbl")
    sel_rows = sel.selection.rows if sel.selection else []
    default_ath = dff.iloc[sel_rows[0]]["athleteName"] if sel_rows else dff.iloc[0]["athleteName"]


# =============================================================================
# TAB 2 — ATHLETE SCORECARD
# =============================================================================
with tab_card:
    athletes = sorted(df["athleteName"].dropna().unique().tolist())

    sc1, sc2, sc3 = st.columns([1.5, 2, 1])
    with sc1:
        search_name = st.text_input("Search athlete", placeholder="Type a name…", key="sc_search")
    with sc2:
        filtered_athletes = ([a for a in athletes if search_name.lower() in a.lower()]
                             if search_name else athletes)
        if not filtered_athletes: filtered_athletes = athletes
        default_idx = filtered_athletes.index(default_ath) if default_ath in filtered_athletes else 0
        sel_ath = st.selectbox("Select athlete", filtered_athletes, index=default_idx, key="sc_ath")
    with sc3:
        ath_years = sorted(df[df["athleteName"]==sel_ath]["Year"].dropna().unique().astype(int).tolist(),
                           reverse=True)
        yr_opts2   = ["All years"] + [str(y) for y in ath_years]
        sel_yr_str = st.selectbox("Year", yr_opts2, key="sc_yr")
        sel_yr     = None if sel_yr_str == "All years" else int(sel_yr_str)

    ath_all = df[df["athleteName"] == sel_ath].sort_values("Year")
    if sel_yr is None:
        row_data = ath_all.select_dtypes(include=[np.number]).mean()
        latest   = ath_all.iloc[-1]
        for c in ath_all.columns:
            if c not in row_data.index: row_data[c] = latest[c]
        row            = row_data
        sel_yr_display = f"{ath_years[-1]}–{ath_years[0]}" if len(ath_years)>1 else str(ath_years[0])
    else:
        sub = ath_all[ath_all["Year"]==sel_yr]
        if sub.empty: st.warning("No data for this year."); st.stop()
        row            = sub.iloc[0]
        sel_yr_display = str(sel_yr)

    # ── Helper ────────────────────────────────────────────────────────────────
    def sf(v):
        try:
            f = float(v)
            return f if not np.isnan(f) else np.nan
        except: return np.nan

    aq_val  = sf(row.get("athlete_quality_score"))
    pos_val = sf(row.get("aq_pos_score"))

    # Percentile helper
    def pct_sfx(p):
        if pd.isna(p): return "—"
        p = int(round(p))
        if 11<=p%100<=13: return f"{p}th"
        return f"{p}{({1:'st',2:'nd',3:'rd'}.get(p%10,'th'))}"

    def alltime_pct(val, col):
        try:
            pool = df[col].dropna().apply(float)
            v    = float(val)
            if len(pool)==0: return "—"
            return pct_sfx(int(round((pool<v).mean()*100)))
        except: return "—"

    aq_pct_str = alltime_pct(aq_val, "athlete_quality_score")

    # CI pathway
    ci_val          = sf(row.get("Concentric Impulse"))
    mass_kg         = sf(row.get("Mass"))
    ht_cm           = sf(row.get("Height"))
    ci_tier_val     = ci_tier_label(ci_val)
    tier_idx        = ci_tier_index(ci_val)
    tier_clrs       = ["#BA0C2F","#E2C188","#11225A"]
    tier_color      = tier_clrs[tier_idx] if tier_idx>=0 else "#9AAAC0"
    prog_cat        = str(row.get("programming_category","—"))
    prog_color      = PROG_COLORS.get(prog_cat, "#9AAAC0")
    prog_desc       = PROG_DESC.get(prog_cat, "")

    lbs_next   = sf(row.get("lbs_to_next_tier"))
    next_label = str(row.get("next_tier_label","—"))
    wc_next    = str(row.get("weight_class_next","—"))
    wc_col     = WEIGHT_CLASS_COLORS.get(wc_next,"#9AAAC0")
    lbs_to_315 = sf(row.get("lbs_to_315"))
    wc_315     = str(row.get("weight_class_315","—"))
    wc_315_col = WEIGHT_CLASS_COLORS.get(wc_315,"#9AAAC0")

    lbs_next_str = f"+{lbs_next:.1f} lbs" if (pd.notna(lbs_next) and lbs_next>0) else ("Top tier" if lbs_next==0 else "—")
    lbs_315_str  = f"+{lbs_to_315:.1f} lbs"  if (pd.notna(lbs_to_315)  and lbs_to_315>0)  else ("✓ Already ≥ 315" if lbs_to_315==0 else "—")

    def proj_bwht(lbs_gain):
        if pd.isna(mass_kg) or pd.isna(ht_cm) or pd.isna(lbs_gain) or lbs_gain<=0: return "—"
        new_kg = mass_kg + lbs_gain/2.20462
        ratio  = (new_kg*2.20462)/(ht_cm/2.54)
        pool_r = df["bmi_raw"].dropna()
        pct    = int(round(float((pool_r<ratio).mean()*100)))
        return f"{ratio:.2f} ({pct_sfx(pct)})"

    # ── Header card ───────────────────────────────────────────────────────────
    pos_str  = str(row.get("Position","")) if str(row.get("Position","")) not in ("nan","None","") else "—"
    sch_str  = str(row.get("School Type","")) if str(row.get("School Type","")) not in ("nan","None","") else "—"
    rnk_val  = int((df["athlete_quality_score"].dropna()>aq_val).sum()+1) if pd.notna(aq_val) else None
    pool_n   = int(df["athlete_quality_score"].notna().sum())

    st.markdown(f"""
    <div style="background:white;border:1px solid {BORD};border-top:4px solid {RED};
        border-radius:10px;padding:20px 28px;margin-bottom:16px;
        box-shadow:0 2px 8px rgba(17,34,90,0.06)">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px">
            <div>
                <p style="font-size:9px;font-weight:700;letter-spacing:0.2em;color:{RED};margin:0 0 4px 0">
                    WASHINGTON NATIONALS · ATHLETE SCORECARD</p>
                <h2 style="font-family:'Playfair Display',serif;font-size:28px;color:{NAV};margin:0 0 8px 0">
                    {sel_ath}</h2>
                <span style="font-size:12px;color:#6b7fa3">
                    {sel_yr_display} · {pos_str} · {sch_str}</span><br>
                <span style="display:inline-block;margin-top:8px;background:{prog_color};
                    color:white;font-size:11px;font-weight:700;padding:3px 14px;
                    border-radius:20px;letter-spacing:0.06em">⚙ {prog_cat}</span>
                <span style="font-size:11px;color:#6b7fa3;margin-left:8px">{prog_desc}</span>
            </div>
            <div style="text-align:right">
                <p style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:#6b7fa3;margin:0">
                    OVERALL RANK</p>
                <p style="font-family:'Playfair Display',serif;font-size:36px;font-weight:900;
                    color:{RED};margin:0">{"#"+str(rnk_val) if rnk_val else "—"}</p>
                <p style="font-size:11px;color:#9AAAC0;margin:0">of {pool_n} athletes (all years)</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero scores + CI tier ─────────────────────────────────────────────────
    h1, h2, h3 = st.columns([1, 1, 1.3])

    with h1:
        bar_w  = min(100, max(0, float(aq_val or 0)))
        bar_cl = GREEN if pd.notna(aq_val) and aq_val>=75 else GOLD if pd.notna(aq_val) and aq_val>=50 else RED
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORD};border-top:6px solid {RED};
            border-radius:12px;padding:24px 20px;text-align:center;
            box-shadow:0 4px 16px rgba(186,12,47,0.12)">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.18em;
                text-transform:uppercase;color:{RED};margin-bottom:8px">★ ATHLETICISM SCORE</div>
            <div style="font-family:'Playfair Display',serif;font-size:64px;
                font-weight:900;color:{RED};line-height:1">
                {str(int(round(aq_val))) if pd.notna(aq_val) else "—"}</div>
            <div style="font-size:13px;font-weight:700;color:{RED};margin-top:4px">{aq_pct_str}</div>
            <div style="font-size:11px;color:#6b7fa3;margin-top:2px">score out of 100 · all-time</div>
            <div style="margin-top:12px;background:#F0F3F8;border-radius:6px;height:8px">
                <div style="width:{bar_w:.0f}%;background:{bar_cl};border-radius:6px;height:8px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with h2:
        # Pos athleticism gauge
        _pg  = str(row.get("pos_group",""))
        _pos = str(row.get("Position",""))
        if _pg and _pg not in ("Unknown","nan",""): pos_lbl = f"Athleticism relative to {_pg}s"
        elif _pos and _pos not in ("nan","None",""): pos_lbl = f"Athleticism relative to {_pos}s"
        else: pos_lbl = "Position Data Unavailable"
        st.plotly_chart(make_gauge(pos_val if (pd.notna(pos_val) and pos_val>0) else None,
                                   pos_lbl, "#6b7fa3"), use_container_width=True, key="g_pos")

    with h3:
        # CI tier card
        parts = []
        parts.append(f'<div style="background:white;border:1px solid {BORD};border-top:4px solid ' + tier_color + f';border-radius:10px;padding:18px 20px;box-shadow:0 2px 8px rgba(17,34,90,0.06)">')
        parts.append(f'<p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#6b7fa3;margin:0 0 4px 0">CI TIER</p>')
        parts.append(f'<div style="font-family:Playfair Display,serif;font-size:32px;font-weight:900;color:' + tier_color + f'">{ci_tier_val}</div>')
        parts.append(f'<div style="font-size:12px;color:#6b7fa3;margin-top:2px">Current CI: <strong style="color:{NAV}">{fmt(ci_val,1)}</strong></div>')
        parts.append(f'<hr style="border-color:{BORD};margin:10px 0">')
        parts.append(f'<div style="display:flex;gap:16px;flex-wrap:wrap">')
        parts.append(f'<div><p style="font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#6b7fa3;margin:0 0 2px 0">To next tier ({next_label})</p>')
        parts.append(f'<div style="font-size:18px;font-weight:700;color:' + wc_col + f'">{lbs_next_str}</div>')
        parts.append(f'<div style="font-size:11px;color:#6b7fa3">BW/Ht at target: {proj_bwht(lbs_next)}</div>')
        parts.append(f'<span style="display:inline-block;background:' + wc_col + f';color:white;font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;margin-top:4px">{wc_next}</span></div>')
        parts.append(f'<div><p style="font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#6b7fa3;margin:0 0 2px 0">To 315 CI</p>')
        parts.append(f'<div style="font-size:18px;font-weight:700;color:' + wc_315_col + f'">{lbs_315_str}</div>')
        parts.append(f'<div style="font-size:11px;color:#6b7fa3">BW/Ht at target: {proj_bwht(lbs_300)}</div>')
        parts.append(f'<span style="display:inline-block;background:' + wc_315_col + f';color:white;font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;margin-top:4px">{wc_315}</span></div>')
        parts.append('</div></div>')
        st.markdown("".join(parts), unsafe_allow_html=True)

    # ── Score context legend ───────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 16px 0">
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{GREEN}"></span>
            <strong>75–100</strong>&nbsp;Elite · clear draft target
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{GOLD}"></span>
            <strong>50–74</strong>&nbsp;Above average · worth consideration
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#D0D7E6"></span>
            <strong>25–49</strong>&nbsp;Below average
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#9AAAC0"></span>
            <strong>0–24</strong>&nbsp;Well below average
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Percentile cards ──────────────────────────────────────────────────────
    grp_key   = str(row.get("pos_group","")).lower()
    grp_label = str(row.get("pos_group","")) if str(row.get("pos_group","")) not in ("Unknown","nan","") else "Pos."
    pc = st.columns(5)
    pct_items = [
        ("CI",          "ci_pct_alltime",     "ci_pct_yr",     False),
        ("30yd Sprint", "sprint_pct_alltime",  "sprint_pct_yr", True),
        ("RSI-mod",     "rsi_pct_alltime",     "rsi_pct_yr",    False),
        ("Pk Pwr/BM",   "pp_pct_alltime",      "pp_pct_yr",     False),
        ("Height",      "height_pct",          None,            False),
    ]
    pos_pct_map = {
        "CI":         f"ci_pct_{grp_key}"     if grp_key else None,
        "30yd Sprint":f"sprint_pct_{grp_key}" if grp_key else None,
        "RSI-mod":    f"rsi_pct_{grp_key}"    if grp_key else None,
        "Pk Pwr/BM":  f"pp_pct_{grp_key}"     if grp_key else None,
        "Height":     None,
    }
    for i, (col_lbl, pa, py, inv) in enumerate(pct_items):
        p_all = sf(row.get(pa))
        p_yr  = sf(row.get(py)) if py else np.nan
        p_pos = sf(row.get(pos_pct_map[col_lbl])) if pos_pct_map[col_lbl] else np.nan
        with pc[i]:
            st.markdown(f"""
            <div class="card card-red" style="text-align:center;padding:14px 10px">
                <div class="label">{col_lbl}</div>
                <div class="score-big" style="color:{RED};font-size:28px">{fmt(p_all,0)}</div>
                <div style="font-size:10px;color:#9AAAC0;margin-top:2px">All-time pct</div>
                <div style="font-size:13px;font-weight:600;color:{NAV};margin-top:4px">
                    {fmt(p_yr,0) if pd.notna(p_yr) else "—"}</div>
                <div style="font-size:10px;color:#9AAAC0">{sel_yr_display} pct</div>
                <div style="font-size:13px;font-weight:600;color:#6b7fa3;margin-top:4px">
                    {fmt(p_pos,0) if pd.notna(p_pos) else "—"}</div>
                <div style="font-size:10px;color:#9AAAC0">{grp_label} pct</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Raw metrics + visuals ─────────────────────────────────────────────────
    m1, m2, m3 = st.columns([1, 1.3, 1.3])

    with m1:
        st.markdown(f"""
        <div class="card card-navy">
            <p class="label">Force Plate</p>
            <div class="stat-row"><span class="stat-label">Concentric Impulse</span>
                <span class="stat-val">{fmt(sf(row.get("Concentric Impulse")),1)}</span></div>
            <div class="stat-row"><span class="stat-label">P1 Conc. Impulse</span>
                <span class="stat-val">{fmt(sf(row.get("P1 Concentric Impulse")),1)}</span></div>
            <div class="stat-row"><span class="stat-label">CI-100ms</span>
                <span class="stat-val">{fmt(sf(row.get("Concentric Impulse-100ms")),1)}</span></div>
            <div class="stat-row"><span class="stat-label">RSI-modified</span>
                <span class="stat-val">{fmt(sf(row.get("RSI-modified")),3)}</span></div>
            <div class="stat-row"><span class="stat-label">Peak Power / BM</span>
                <span class="stat-val">{fmt(sf(row.get("Peak Power / BM")),1)}</span></div>
            <div class="stat-row"><span class="stat-label">Jump Height</span>
                <span class="stat-val">{fmt(sf(row.get("Jump Height (Flight Time) in Inches")),2," in")}</span></div>
            <p class="label" style="margin-top:12px">Sprint</p>
            <div class="stat-row"><span class="stat-label">30yd</span>
                <span class="stat-val">{fmt(sf(row.get("30yd Split")),3,"s")}</span></div>
            <div class="stat-row"><span class="stat-label">10yd</span>
                <span class="stat-val">{fmt(sf(row.get("10yd Split")),3,"s")}</span></div>
            <div class="stat-row"><span class="stat-label">20yd</span>
                <span class="stat-val">{fmt(sf(row.get("20yd Split")),3,"s")}</span></div>
            <p class="label" style="margin-top:12px">Anthropometrics</p>
            <div class="stat-row"><span class="stat-label">Height</span>
                <span class="stat-val">{fmt_height(sf(row.get("Height")))}</span></div>
            <div class="stat-row"><span class="stat-label">Mass</span>
                <span class="stat-val">{fmt_mass(sf(row.get("Mass")))}</span></div>
            <div class="stat-row"><span class="stat-label">Wingspan</span>
                <span class="stat-val">{fmt_wingspan(sf(row.get("Wingspan")))}</span></div>
            <div class="stat-row"><span class="stat-label">Wingspan Adv.</span>
                <span class="stat-val">{fmt_wingspan_adv(sf(row.get("wingspan_advantage")))}</span></div>
            <div class="stat-row"><span class="stat-label">BW/Ht Ratio</span>
                <span class="stat-val">{fmt(sf(row.get("bmi_raw")),2)}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.plotly_chart(make_radar(row), use_container_width=True, key="g_radar")
        st.plotly_chart(make_profile(row, strat_feats), use_container_width=True, key="g_profile")

    with m3:
        st.markdown(f'<p class="label" style="margin-bottom:4px">Why Flagged</p>', unsafe_allow_html=True)
        why = str(row.get("why_flagged","—"))
        st.markdown(f'<div class="why-text">{why}</div>', unsafe_allow_html=True)

        # Year-over-year trends
        multi = ath_all[ath_all["Year"].notna()]
        if len(multi) >= 2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<p class="label">Year-over-year trends</p>', unsafe_allow_html=True)
            for tcol, tlbl, tinv in [
                ("Concentric Impulse", "CI", False),
                ("RSI-modified",       "RSI-mod", False),
                ("30yd Split",         "30yd Sprint", True),
                ("athlete_quality_score", "Athleticism Score", False),
            ]:
                fig_t = make_trend(multi, tcol, tlbl, tinv)
                if fig_t:
                    st.plotly_chart(fig_t, use_container_width=True,
                                    key=f"trend_{sel_ath}_{tcol}")

    # ── Development projection table ──────────────────────────────────────────
    if pd.notna(ci_val) and pd.notna(mass_kg) and ci_val>0 and mass_kg>0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="border-left:4px solid {GREEN};padding-left:12px;margin:0 0 14px 0">'
            f'<span style="font-size:10px;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:{GREEN}">Development Projection</span></div>',
            unsafe_allow_html=True)

        ci_per_kg = ci_val / mass_kg
        ci_per_kg_new = ci_per_kg * 0.97

        def sc_bwht(kg, cm):
            if pd.isna(kg) or pd.isna(cm) or cm==0: return np.nan
            return (kg*2.20462)/(cm/2.54)
        def sc_bwht_pct(val):
            pool_r = df["bmi_raw"].dropna()
            if pd.isna(val) or len(pool_r)==0: return np.nan
            return float((pool_r<val).mean()*100)
        def sc_ci_pct(val):
            pool_c = df["Concentric Impulse"].dropna()
            if pd.isna(val) or len(pool_c)==0: return np.nan
            return float((pool_c<val).mean()*100)

        bwht_cur = sc_bwht(mass_kg, ht_cm)

        rows_proj = []
        for label, gain_lbs in [("Current",0),("+10 lbs",10),("+15 lbs",15)]:
            new_kg   = mass_kg + gain_lbs/2.20462
            ci_p     = ci_per_kg_new * new_kg if gain_lbs>0 else ci_val
            bwht_v   = sc_bwht(new_kg, ht_cm)
            bwht_p   = sc_bwht_pct(bwht_v)
            ci_p_pct = sc_ci_pct(ci_p)
            delta    = ci_p - ci_val
            rows_proj.append({
                "Scenario": label,
                "CI (N·s)": f"{ci_p:.1f}" + (f"  ({'+' if delta>=0 else ''}{delta:.1f})" if gain_lbs>0 else ""),
                "CI Pct":   pct_sfx(int(round(ci_p_pct))) if pd.notna(ci_p_pct) else "—",
                "Body Mass":f"{new_kg*2.20462:.1f} lbs",
                "BW/Ht":    f"{bwht_v:.2f}" if pd.notna(bwht_v) else "—",
                "Leanness Pct": pct_sfx(int(round(bwht_p))) if pd.notna(bwht_p) else "—",
            })

        proj_tbl = pd.DataFrame(rows_proj)
        st.dataframe(proj_tbl, use_container_width=True, hide_index=True, key="sc_proj_tbl")
        st.markdown(
            f'<p style="font-size:11px;color:#9AAAC0;margin-top:4px">'
            f'Assumes 3% decrease in CI/kg with added mass. All-time percentiles. '
            f'Internal data has shown a trend towards pitchers having smaller penalties (0–3%) '
            f'than position players (3–5%).</p>', unsafe_allow_html=True)
