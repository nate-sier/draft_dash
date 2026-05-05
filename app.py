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

# ─── Config ───────────────────────────────────────────────────────────────────
SHEET_ID   = "1RKyeb4CfU4wACUKqpbiR_--PJJ9KKO-LH-Ov9e6zNQs"
PITCHER_POSITIONS = {"P", "SP", "RP", "Starting Pitcher", "Relief Pitcher",
                     "Right Hand Pitcher", "Left Hand Pitcher", "SC", "PC",
                     "Starting pitcher", "starting pitcher"}

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

    # Strategy distance (Mahalanobis, all-time pool) — pure numpy
    X_s = df[strategy_features].fillna(0).values.astype(float)
    # Standardize
    mu_s  = X_s.mean(axis=0)
    sd_s  = X_s.std(axis=0)
    sd_s[sd_s == 0] = 1.0
    X_sc  = (X_s - mu_s) / sd_s
    # PCA via SVD
    _, s_vals, Vt = np.linalg.svd(X_sc, full_matrices=False)
    ncomp  = min(len(strategy_features), max(3, min(6, len(df) - 1)))
    X_pca  = X_sc @ Vt[:ncomp].T
    # Mahalanobis via covariance of projected data
    try:
        cov    = np.cov(X_pca, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        mu_pca = X_pca.mean(axis=0)
        diff   = X_pca - mu_pca
        dist2  = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        df["strategy_distance_raw"] = np.sqrt(np.abs(dist2))
    except Exception:
        df["strategy_distance_raw"] = np.linalg.norm(X_pca, axis=1)
    df["strategy_distance_score"] = pct_rank(df["strategy_distance_raw"])

    # Archetypes
    df["archetype"]  = df.apply(lambda r: label_archetype(r.to_dict(), all_rz_cols), axis=1)
    df["why_flagged"] = df.apply(lambda r: explain_strategy(
        {f"rz_{f}": r.get(f"rz_{f}", 0) for f in strategy_features}, strategy_features), axis=1)

    # ── Athlete Quality Score ──────────────────────────────────────────────────
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
    df["bmi_raw"]     = safe_div(df["Mass"], df["Height"])
    df["bmi_pct"]     = 100 - pct_rank(df["bmi_raw"])
    # School type: HS > 4-Year College > Junior College
    school_score_map  = {"High School": 100, "4-Year College": 60, "Junior College": 40}
    df["school_score"] = df["School Type"].map(school_score_map).fillna(50)
    # Wingspan advantage (wingspan - height; positive = longer reach)
    df["wingspan_advantage"] = df["Wingspan"] - df["Height"]
    df["wingspan_pct"]       = pct_rank(df["wingspan_advantage"])

    # Build potential — include wingspan only for pitchers
    def pot_score(row):
        is_pitcher = False  # no position col; use school type as proxy if needed
        base = (wp_peakpow * (row.get("pp_pct") or 50) +
                wp_height  * (row.get("height_pct") or 50) +
                wp_bmi     * (row.get("bmi_pct") or 50) +
                wp_school  * (row.get("school_score") or 50))
        # Redistribute wingspan weight to others if not pitcher
        # (wingspan weight already baked into sliders; use as-is for all, note in UI)
        pot = base + wp_wingspan * (row.get("wingspan_pct") or 50)
        return pot

    df["potential_raw"]   = df.apply(pot_score, axis=1)
    df["potential_score"] = scaled_0_100(df["potential_raw"].values)

    # Per-year potential
    df["potential_score_yr"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "potential_score_yr"] = scaled_0_100(df.loc[idx, "potential_raw"].values)

    # Overall rank (athlete quality only)
    # Rank within each year
    df["overall_rank"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "overall_rank"] = (
            df.loc[idx, "athlete_quality_score"]
            .rank(ascending=False, method="min")
        )
    df["overall_rank"] = df["overall_rank"].astype("Int64")

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
        dff, x="athlete_quality_score", y="potential_score",
        color="archetype", hover_name="athleteName",
        hover_data={"Year": True, "athlete_quality_score": ":.1f",
                    "potential_score": ":.1f", "archetype": True},
        color_discrete_map=ARCHETYPE_COLORS,
        labels={"athlete_quality_score": "Athlete Quality Score",
                "potential_score": "Potential Score"},
        height=480,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(**_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text="Quality vs Potential", font=dict(size=14, color=NAV), x=0),
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

    st.markdown("---")
    st.markdown(f'<p style="font-size:10px;font-weight:600;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:#6b7fa3">Potential Weights</p>',
                unsafe_allow_html=True)
    wp_pp     = st.slider("Peak Power / BM",     0, 100, 25, 5, key="wp_pp") / 100
    wp_ht     = st.slider("Height",              0, 100, 25, 5, key="wp_ht") / 100
    wp_bmi    = st.slider("Leanness (BW/Ht)",    0, 100, 20, 5, key="wp_bmi") / 100
    wp_school = st.slider("School Type",         0, 100, 15, 5, key="wp_school") / 100
    wp_wings  = st.slider("Wingspan Advantage",  0, 100, 15, 5, key="wp_wings") / 100
    total_pot = wp_pp + wp_ht + wp_bmi + wp_school + wp_wings
    if not (0.99 < total_pot < 1.01):
        st.warning(f"Weights sum to {total_pot*100:.0f}% — should be 100%")

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

tab_board, tab_card, tab_pct, tab_guide, tab_ref = st.tabs(["Leaderboard", "Athlete Scorecard", "Distributions", "Guide", "Reference"])

# =============================================================================
# TAB 1 — LEADERBOARD
# =============================================================================
with tab_board:
    fc1, fc2, fc3, fc4, fc5, fc6 = st.columns([2, 1, 1.2, 1.2, 1.2, 1.2])
    with fc1: search  = st.text_input("Search athlete", placeholder="Name…", key="lb_search")
    with fc2:
        yr_opts = ["All"] + sorted(df["Year"].dropna().unique().astype(int).tolist(), reverse=True)
        yr_sel  = st.selectbox("Year", yr_opts, key="lb_year")
    with fc3:
        pos_grp_opts = ["All", "Pitcher", "Catcher", "Infielder", "Outfielder"]
        pos_grp_sel  = st.selectbox("Position Group", pos_grp_opts, key="lb_posgrp")
    with fc4:
        arch_opts = ["All"] + sorted(df["archetype"].dropna().unique())
        arch_sel  = st.selectbox("Archetype", arch_opts, key="lb_arch")
    with fc5:
        st_opts = ["All"] + sorted(df["School Type"].dropna().unique())
        st_sel  = st.selectbox("School Type", st_opts, key="lb_st")
    with fc6:
        sort_by = st.selectbox("Sort by", ["Athlete Quality", "Pos. Group Quality",
                                            "Potential", "CI", "30yd Sprint"],
                               key="lb_sort")

    dff = df.copy()
    if search:              dff = dff[dff["athleteName"].str.contains(search, case=False, na=False)]
    if yr_sel != "All":     dff = dff[dff["Year"] == int(yr_sel)]
    if pos_grp_sel != "All": dff = dff[dff["pos_group"] == pos_grp_sel]
    if arch_sel != "All":   dff = dff[dff["archetype"] == arch_sel]
    if st_sel != "All":     dff = dff[dff["School Type"] == st_sel]

    sort_col = {"Athlete Quality": "athlete_quality_score",
                "Pos. Group Quality": "aq_pos_score",
                "Potential": "potential_score",
                "CI": "Concentric Impulse",
                "30yd Sprint": "30yd Split"}[sort_by]
    asc = sort_by == "30yd Sprint"
    dff = dff.sort_values(sort_col, ascending=asc, na_position="last").reset_index(drop=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Athletes", str(len(dff)))
    k2.metric("Avg Quality", fmt(dff["athlete_quality_score"].mean()))
    k3.metric("Avg Potential", fmt(dff["potential_score"].mean()))
    k4.metric("Median CI", fmt(dff["Concentric Impulse"].median(), 0))
    k5.metric("Median 30yd", fmt(dff["30yd Split"].median(), 3, "s"))

    st.markdown("<br>", unsafe_allow_html=True)

    ch1, ch2 = st.columns([1.4, 1])
    with ch1:
        st.plotly_chart(make_scatter(dff), use_container_width=True, key="lb_scatter")
    with ch2:
        arch_vc = dff["archetype"].value_counts().reset_index()
        arch_vc.columns = ["archetype", "count"]
        fig_arch = px.bar(arch_vc, x="archetype", y="count", color="archetype",
                          color_discrete_map=ARCHETYPE_COLORS, height=240)
        fig_arch.update_layout(**_layout(
            title=dict(text="Archetypes", font=dict(size=13, color=NAV), x=0),
            margin=dict(l=20, r=10, t=45, b=80), showlegend=False,
            xaxis_title="", yaxis_title="Athletes"))
        st.plotly_chart(fig_arch, use_container_width=True, key="lb_arch_bar")

        st_vc = dff["School Type"].value_counts().reset_index()
        st_vc.columns = ["School Type","count"]
        fig_st = px.pie(st_vc, names="School Type", values="count",
                        color_discrete_sequence=[RED, NAV, GOLD, "#6b7fa3"],
                        height=220)
        fig_st.update_layout(**_layout(
            title=dict(text="School Type Mix", font=dict(size=13, color=NAV), x=0),
            margin=dict(l=10, r=10, t=45, b=10),
            legend=dict(font=dict(color=NAV, size=10))))
        st.plotly_chart(fig_st, use_container_width=True, key="lb_st_pie")

    st.markdown('<p class="label" style="margin-top:8px">Ranked Athletes</p>',
                unsafe_allow_html=True)

    tbl = dff[["athleteName","Year","Position","pos_group","School Type","archetype",
               "athlete_quality_score","aq_pos_score","potential_score",
               "Concentric Impulse","30yd Split","RSI-modified","Peak Power / BM"]].copy()
    tbl = tbl.rename(columns={
        "athleteName": "Athlete", "pos_group": "Group", "School Type": "School",
        "archetype": "Archetype", "athlete_quality_score": "Quality",
        "aq_pos_score": "Pos. Quality", "potential_score": "Potential",
        "Concentric Impulse": "CI", "30yd Split": "30yd (s)",
        "RSI-modified": "RSI-mod", "Peak Power / BM": "PkPwr/BM",
    })
    for c in ["Quality","Pos. Quality","Potential","CI","RSI-mod","PkPwr/BM"]:
        tbl[c] = tbl[c].round(1)
    tbl["30yd (s)"] = tbl["30yd (s)"].round(3)

    sel = st.dataframe(tbl, use_container_width=True, hide_index=True,
                       on_select="rerun", selection_mode="single-row", key="lb_table")
    sel_rows = sel.selection.rows if sel.selection else []
    if sel_rows:
        sel_pid = dff.iloc[sel_rows[0]]["playerID"]
        st.session_state["scorecard_pid"] = sel_pid
        st.session_state["scorecard_yr"]  = dff.iloc[sel_rows[0]]["Year"]
        st.info("Row selected — switch to the Athlete Scorecard tab to view the full profile.")

# =============================================================================
# TAB 2 — ATHLETE SCORECARD
# =============================================================================
with tab_card:
    athletes = sorted(df["athleteName"].dropna().unique())

    # Pre-select from leaderboard click
    default_ath = athletes[0]
    if "scorecard_pid" in st.session_state:
        pid_match = df[df["playerID"] == st.session_state["scorecard_pid"]]["athleteName"]
        if not pid_match.empty:
            default_ath = pid_match.iloc[0]

    sc1, sc2, sc3 = st.columns([1.5, 2, 1])
    with sc1:
        search_name = st.text_input("Search athlete", placeholder="Type a name…", key="sc_search")
    with sc2:
        filtered_athletes = (
            [a for a in athletes if search_name.lower() in a.lower()]
            if search_name else athletes
        )
        if not filtered_athletes:
            filtered_athletes = athletes
        default_idx = 0
        if default_ath in filtered_athletes:
            default_idx = filtered_athletes.index(default_ath)
        sel_ath = st.selectbox("Select athlete", filtered_athletes,
                               index=default_idx, key="sc_ath")
    with sc3:
        ath_years = sorted(df[df["athleteName"] == sel_ath]["Year"].dropna().unique().astype(int).tolist(),
                           reverse=True)
        yr_options = ["All years"] + [str(y) for y in ath_years]
        sel_yr_str = st.selectbox("Year", yr_options, key="sc_yr")
        sel_yr = None if sel_yr_str == "All years" else int(sel_yr_str)

    ath_all = df[df["athleteName"] == sel_ath].sort_values("Year")

    if sel_yr is None:
        # All years — average numeric columns across years
        row_data = ath_all.select_dtypes(include=[np.number]).mean()
        # Use most recent year for non-numeric fields
        latest = ath_all.iloc[-1]
        for col in ath_all.columns:
            if col not in row_data.index:
                row_data[col] = latest[col]
        row = row_data
        sel_yr_label = f"{ath_years[-1]}–{ath_years[0]}" if len(ath_years) > 1 else str(ath_years[0])
        sel_yr_display = sel_yr_label
    else:
        row = ath_all[ath_all["Year"] == sel_yr]
        if row.empty:
            st.warning("No data for this athlete/year combination.")
            st.stop()
        row = row.iloc[0]
        sel_yr_label   = str(sel_yr)
        sel_yr_display = str(sel_yr)

    arch_color = ARCHETYPE_COLORS.get(row.get("archetype","Unclassified"), "#9AAAC0")

    # ── Header banner ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:20px 28px;margin-bottom:20px;
        border:1px solid {BORD};border-top:4px solid {RED};
        box-shadow:0 2px 8px rgba(17,34,90,0.06);">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px">
            <div>
                <p style="font-size:9px;font-weight:700;letter-spacing:0.2em;color:{RED};margin:0 0 4px 0">
                    WASHINGTON NATIONALS · ATHLETE SCORECARD</p>
                <h2 style="font-family:'Playfair Display',serif;font-size:28px;color:{NAV};margin:0 0 8px 0">
                    {sel_ath}</h2>
                <span class="arch-badge" style="background:{arch_color}">{row.get('archetype','—')}</span>
                <span style="font-size:12px;color:#6b7fa3;margin-left:10px">
                    {sel_yr_display} · {row.get('Position','—')} · {row.get('School Type','—')}</span>
            </div>
            <div style="text-align:right">
                <p style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:#6b7fa3;margin:0">
                    OVERALL RANK</p>
                <p style="font-family:'Playfair Display',serif;font-size:36px;font-weight:900;
                    color:{RED};margin:0">#{int(row.get('overall_rank', 0))}</p>
                <p style="font-size:11px;color:#9AAAC0;margin:0">{"of " + str(int(df["overall_rank"].notna().sum())) + " athletes (all years)" if sel_yr is None else f"of {int(df[df['Year']==sel_yr]['overall_rank'].notna().sum())} athletes in {sel_yr}"}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score heroes — Quality + Potential front and center ──────────────────
    aq_val  = row.get("athlete_quality_score", np.nan)
    pot_val = row.get("potential_score", np.nan)
    pos_val = row.get("aq_pos_score", np.nan)

    # Rank-derived percentile (most accurate — direct count of athletes beaten)
    def pct_suffix(p):
        if pd.isna(p): return "—"
        p = int(round(p))
        if 11 <= p % 100 <= 13: return f"{p}th"
        return {1:"st",2:"nd",3:"rd"}.get(p % 10, "th")
    def rank_pct_str(rank_val, total):
        if pd.isna(rank_val) or total == 0: return "—"
        pct = int(round((1 - (rank_val - 1) / total) * 100))
        return f"{pct}{pct_suffix(pct)} percentile"

    overall_rank_val = row.get("overall_rank", np.nan)
    yr_pool_size = (int(df[df["Year"]==sel_yr]["overall_rank"].notna().sum())
                    if sel_yr else int(df["overall_rank"].notna().sum()))
    aq_pct_str  = rank_pct_str(overall_rank_val, yr_pool_size)

    # Potential rank percentile
    pot_rank = df["potential_score"].dropna().rank(ascending=False, method="min")
    pot_rank_val = np.nan
    try:
        if sel_yr:
            yr_df = df[df["Year"] == sel_yr]
            pot_ranks_yr = yr_df["potential_score"].dropna().rank(ascending=False, method="min")
            match = yr_df[yr_df["athleteName"] == sel_ath]["potential_score"]
            if not match.empty and pd.notna(match.iloc[0]):
                pot_rank_val = pot_ranks_yr[match.index[0]]
                pot_pool_size = yr_df["potential_score"].notna().sum()
            else:
                pot_rank_val, pot_pool_size = np.nan, 0
        else:
            match = df[df["athleteName"] == sel_ath]["potential_score"]
            if not match.empty:
                pot_rank_val = df["potential_score"].dropna().rank(ascending=False, method="min")[match.index[0]]
                pot_pool_size = df["potential_score"].notna().sum()
            else:
                pot_rank_val, pot_pool_size = np.nan, 0
    except Exception:
        pot_rank_val, pot_pool_size = np.nan, 0
    pot_pct_str = rank_pct_str(pot_rank_val, pot_pool_size)

    def score_color(v):
        if pd.isna(v): return "#9AAAC0"
        if v >= 75: return GREEN
        if v >= 50: return GOLD
        return RED

    hero1, hero2, hero3 = st.columns([1, 1, 1.2])
    with hero1:
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORD};border-radius:12px;
            padding:24px 20px;text-align:center;
            box-shadow:0 4px 16px rgba(186,12,47,0.12);
            border-top:6px solid {RED};margin-bottom:4px">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.18em;
                text-transform:uppercase;color:{RED};margin-bottom:8px">
                ★ ATHLETE QUALITY</div>
            <div style="font-family:'Playfair Display',serif;font-size:64px;
                font-weight:900;color:{RED};line-height:1">
                {f"{aq_val:.0f}" if pd.notna(aq_val) else "—"}</div>
            <div style="font-size:13px;font-weight:700;color:{RED};margin-top:4px">
                {aq_pct_str}</div>
            <div style="font-size:11px;color:#6b7fa3;margin-top:2px">score out of 100 · all-time</div>
            <div style="margin-top:12px;background:#F0F3F8;border-radius:6px;height:8px">
                <div style="width:{min(100,max(0,aq_val or 0)):.0f}%;
                    background:{"#4CAF82" if pd.notna(aq_val) and aq_val>=75 else "#E2C188" if pd.notna(aq_val) and aq_val>=50 else RED};
                    border-radius:6px;height:8px;transition:width 0.5s"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with hero2:
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORD};border-radius:12px;
            padding:24px 20px;text-align:center;
            box-shadow:0 4px 16px rgba(17,34,90,0.12);
            border-top:6px solid {NAV};margin-bottom:4px">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.18em;
                text-transform:uppercase;color:{NAV};margin-bottom:8px">
                ★ DEVELOPMENT POTENTIAL</div>
            <div style="font-family:'Playfair Display',serif;font-size:64px;
                font-weight:900;color:{NAV};line-height:1">
                {f"{pot_val:.0f}" if pd.notna(pot_val) else "—"}</div>
            <div style="font-size:13px;font-weight:700;color:{NAV};margin-top:4px">
                {pot_pct_str}</div>
            <div style="font-size:11px;color:#6b7fa3;margin-top:6px">out of 100 · all-time</div>
            <div style="margin-top:12px;background:#F0F3F8;border-radius:6px;height:8px">
                <div style="width:{min(100,max(0,pot_val or 0)):.0f}%;
                    background:{"#4CAF82" if pd.notna(pot_val) and pot_val>=75 else "#E2C188" if pd.notna(pot_val) and pot_val>=50 else NAV};
                    border-radius:6px;height:8px;transition:width 0.5s"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with hero3:
        # Pos group quality gauge + radar stacked
        pos_grp_label = f"{row.get('pos_group','Pos.')} Quality"
        st.plotly_chart(make_gauge(pos_val, pos_grp_label, "#6b7fa3"),
                        use_container_width=True, key="g_pos")
        st.plotly_chart(make_radar(row), use_container_width=True, key="g_radar")

    # ── Score context legend ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;margin-top:4px">
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                background:{GREEN}"></span>
            <strong>75–100</strong>&nbsp;Elite · clear draft target
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                background:{GOLD}"></span>
            <strong>50–74</strong>&nbsp;Above average · worth consideration
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                background:#D0D7E6"></span>
            <strong>25–49</strong>&nbsp;Below average · needs more development
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                background:#9AAAC0"></span>
            <strong>0–24</strong>&nbsp;Well below average
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Percentile cards ──────────────────────────────────────────────────────
    st.markdown('<p class="label" style="margin-top:4px">Percentiles</p>', unsafe_allow_html=True)
    pc = st.columns(5)
    grp       = row.get("pos_group", "Unknown")
    grp_label = grp if grp != "Unknown" else "Pos. Group"
    grp_key   = grp.lower() if grp != "Unknown" else None

    pct_items = [
        ("CI",         "ci_pct_alltime",     "ci_pct_yr",      False),
        ("30yd Sprint","sprint_pct_alltime",  "sprint_pct_yr",  True),
        ("RSI-mod",    "rsi_pct_alltime",     "rsi_pct_yr",     False),
        ("Pk Pwr/BM",  "pp_pct_alltime",      "pp_pct_yr",      False),
        ("Height",     "height_pct",          None,             False),
    ]
    pos_pct_map = {
        "CI":         f"ci_pct_{grp_key}"     if grp_key else None,
        "30yd Sprint":f"sprint_pct_{grp_key}" if grp_key else None,
        "RSI-mod":    f"rsi_pct_{grp_key}"    if grp_key else None,
        "Pk Pwr/BM":  f"pp_pct_{grp_key}"     if grp_key else None,
        "Height":     None,
    }
    for col, pct_all_col, pct_yr_col, inv in pct_items:
        p_all    = row.get(pct_all_col, np.nan)
        p_yr     = row.get(pct_yr_col, np.nan) if pct_yr_col else np.nan
        p_pos    = row.get(pos_pct_map[col], np.nan) if pos_pct_map[col] else np.nan
        with pc[pct_items.index((col, pct_all_col, pct_yr_col, inv))]:
            st.markdown(f"""
            <div class="card card-red" style="text-align:center;padding:14px 10px">
                <div class="label">{col}</div>
                <div class="score-big" style="color:{RED};font-size:30px">{fmt(p_all, 0)}</div>
                <div style="font-size:10px;color:#9AAAC0;margin-top:2px">All-time pct</div>
                <div style="font-size:13px;font-weight:600;color:{NAV};margin-top:4px">
                    {fmt(p_yr, 0) if pd.notna(p_yr) else '—'}</div>
                <div style="font-size:10px;color:#9AAAC0">{sel_yr_display} pct</div>
                <div style="font-size:13px;font-weight:600;color:#6b7fa3;margin-top:4px">
                    {fmt(p_pos, 0) if pd.notna(p_pos) else '—'}</div>
                <div style="font-size:10px;color:#9AAAC0">{grp_label} pct</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Detail columns ────────────────────────────────────────────────────────
    d1, d2, d3 = st.columns([1, 1.2, 1.2])

    with d1:
        st.markdown(f"""
        <div class="card card-navy">
            <p class="label">Raw Metrics</p>
            <div class="stat-row"><span class="stat-label">Concentric Impulse</span>
                <span class="stat-val">{fmt(row.get('Concentric Impulse'), 1)}</span></div>
            <div class="stat-row"><span class="stat-label">RSI-modified</span>
                <span class="stat-val">{fmt(row.get('RSI-modified'), 3)}</span></div>
            <div class="stat-row"><span class="stat-label">30yd Sprint</span>
                <span class="stat-val">{fmt(row.get('30yd Split'), 3, 's')}</span></div>
            <div class="stat-row"><span class="stat-label">10yd Split</span>
                <span class="stat-val">{fmt(row.get('10yd Split'), 3, 's')}</span></div>
            <div class="stat-row"><span class="stat-label">20yd Split</span>
                <span class="stat-val">{fmt(row.get('20yd Split'), 3, 's')}</span></div>
            <div class="stat-row"><span class="stat-label">Jump Height</span>
                <span class="stat-val">{fmt(row.get('Jump Height (Flight Time) in Inches'), 2, ' in')}</span></div>
            <div class="stat-row"><span class="stat-label">Peak Power / BM</span>
                <span class="stat-val">{fmt(row.get('Peak Power / BM'), 1)}</span></div>
            <hr style="border-color:{BORD};margin:10px 0">
            <p class="label">Anthropometrics</p>
            <div class="stat-row"><span class="stat-label">Height</span>
                <span class="stat-val">{fmt(row.get('Height'), 1, ' cm')}</span></div>
            <div class="stat-row"><span class="stat-label">Mass</span>
                <span class="stat-val">{fmt(row.get('Mass'), 1, ' kg')}</span></div>
            <div class="stat-row"><span class="stat-label">Wingspan</span>
                <span class="stat-val">{fmt(row.get('Wingspan'), 1, ' cm')}</span></div>
            <div class="stat-row"><span class="stat-label">Wingspan Adv.</span>
                <span class="stat-val">{fmt(row.get('wingspan_advantage'), 1, ' cm')}</span></div>
            <div class="stat-row"><span class="stat-label">BW/Ht Ratio</span>
                <span class="stat-val">{fmt(row.get('bmi_raw'), 3)}</span></div>
        </div>
        <div class="card card-gold">
            <p class="label" style="color:{RED}">CMJ Strategy — Why Flagged</p>
            <div class="why-text">{row.get('why_flagged','—')}</div>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.plotly_chart(make_profile(row, strat_feats),
                        use_container_width=True, key="sc_profile")

        # Potential breakdown bar
        pot_components = {
            "Peak Pwr/BM": row.get("pp_pct", np.nan),
            "Height":      row.get("height_pct", np.nan),
            "Leanness":    row.get("bmi_pct", np.nan),
            "School Type": row.get("school_score", np.nan),
            "Wingspan":    row.get("wingspan_pct", np.nan),
        }
        fig_pot = go.Figure(go.Bar(
            y=list(pot_components.keys()),
            x=[v if pd.notna(v) else 0 for v in pot_components.values()],
            orientation="h",
            marker_color=[RED, NAV, GOLD, GREEN, "#6b7fa3"],
            text=[f"{v:.0f}" if pd.notna(v) else "—" for v in pot_components.values()],
            textposition="outside",
        ))
        fig_pot.add_vline(x=50, line_dash="dot", line_color="#9AAAC0", line_width=1)
        fig_pot.update_layout(**_layout(
            title=dict(text="Potential Components (percentile)", font=dict(size=12, color=NAV), x=0),
            height=260, margin=dict(l=100, r=40, t=45, b=20),
            xaxis=dict(range=[0, 115], showgrid=False),
            yaxis=dict(tickfont=dict(size=10)),
            showlegend=False,
        ))
        st.plotly_chart(fig_pot, use_container_width=True, key="sc_pot_bar")

    with d3:
        if len(ath_all) >= 2:
            st.markdown('<p class="label">Year-over-Year Trends</p>', unsafe_allow_html=True)
            trend_items = [
                ("Concentric Impulse", "Concentric Impulse", False),
                ("RSI-modified",       "RSI-modified",       False),
                ("30yd Split",         "30yd Sprint",        True),
                ("Peak Power / BM",    "Peak Power / BM",    False),
                ("Mass",               "Body Mass (kg)",     False),
            ]
            for col, lbl, inv in trend_items:
                fig_t = make_trend(ath_all, col, lbl, invert=inv)
                if fig_t:
                    st.plotly_chart(fig_t, use_container_width=True,
                                    key=f"trend_{col}_{sel_ath}")
        else:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:40px 20px">
                <p style="color:#9AAAC0;font-size:13px;margin:0">
                    Trend charts appear once a second year of data is recorded.</p>
            </div>
            """, unsafe_allow_html=True)



# =============================================================================
# TAB 3 — PERCENTILE REFERENCE
# =============================================================================
with tab_pct:

    rf1, rf2, rf3 = st.columns([1, 1.2, 1.2])
    with rf1:
        yr_opts_ref = ["All years"] + sorted(df["Year"].dropna().unique().astype(int).tolist(), reverse=True)
        ref_yr      = st.selectbox("Year", yr_opts_ref, key="ref_yr")
    with rf2:
        ref_pos_opts = ["All positions", "Pitcher", "Catcher", "Infielder", "Outfielder"]
        ref_pos      = st.selectbox("Position Group", ref_pos_opts, key="ref_pos")
    with rf3:
        ref_chart    = st.selectbox("Chart type", ["Histogram", "Box plot"], key="ref_chart")

    # Filter population
    ref_df = df.copy()
    if ref_yr != "All years":
        ref_df = ref_df[ref_df["Year"] == int(ref_yr)]
    if ref_pos != "All positions":
        ref_df = ref_df[ref_df["pos_group"] == ref_pos]

    n_pop = len(ref_df)
    st.markdown(
        f'<p style="font-size:12px;color:#6b7fa3;margin-bottom:16px">'
        f'Showing distribution for <strong style="color:{NAV}">{n_pop} athlete-years</strong>'
        f'{" · " + ref_yr if ref_yr != "All years" else ""}'
        f'{" · " + ref_pos if ref_pos != "All positions" else ""}</p>',
        unsafe_allow_html=True)

    def dist_fig(col, label, nbins=25, invert=False, digits=2, suffix=""):
        data = ref_df[col].dropna()
        if len(data) < 3:
            return None

        q10 = data.quantile(0.10 if not invert else 0.90)
        q25 = data.quantile(0.25 if not invert else 0.75)
        q50 = data.median()
        q75 = data.quantile(0.75 if not invert else 0.25)
        q90 = data.quantile(0.90 if not invert else 0.10)

        if ref_chart == "Histogram":
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data, nbinsx=nbins,
                marker_color=NAV,
                marker_line=dict(color="white", width=0.5),
                opacity=0.9,
            ))
            # Clean vertical lines for key percentiles
            line_styles = [
                (q25, "p25", GOLD,  "dash"),
                (q50, "p50", RED,   "solid"),
                (q75, "p75", GOLD,  "dash"),
                (q90, "p90", GREEN, "dot"),
            ]
            for q, lbl, col_color, dash in line_styles:
                fig.add_vline(
                    x=q, line_dash=dash, line_color=col_color, line_width=2,
                    annotation=dict(
                        text=f"<b>{lbl}</b><br>{q:.{digits}f}{suffix}",
                        font=dict(color=col_color, size=10),
                        bgcolor="white",
                        bordercolor=col_color,
                        borderwidth=1,
                        borderpad=3,
                        yref="paper", y=0.98,
                        showarrow=False,
                    ),
                )
            fig.update_layout(**_layout(
                title=dict(text=f"<b>{label}</b>", font=dict(size=13, color=NAV), x=0),
                height=280,
                margin=dict(l=40, r=20, t=55, b=40),
                xaxis=dict(
                    tickfont=dict(size=10, color=NAV),
                    showgrid=True, gridcolor=BORD, gridwidth=1,
                    zeroline=False,
                ),
                yaxis=dict(
                    title="Count", tickfont=dict(size=10, color=NAV),
                    showgrid=True, gridcolor=BORD, gridwidth=1,
                ),
                showlegend=False,
                plot_bgcolor="white",
            ))
        else:
            fig = go.Figure()
            fig.add_trace(go.Box(
                x=data, name=label,
                marker_color=RED,
                line_color=NAV,
                line_width=2,
                fillcolor=f"rgba(186,12,47,0.15)",
                boxmean="sd",
                boxpoints="outliers",
                jitter=0.4,
                marker=dict(size=5, opacity=0.5, color=NAV),
            ))
            fig.update_layout(**_layout(
                title=dict(text=f"<b>{label}</b>", font=dict(size=13, color=NAV), x=0),
                height=280,
                margin=dict(l=40, r=20, t=55, b=40),
                xaxis=dict(
                    tickfont=dict(size=10, color=NAV),
                    showgrid=True, gridcolor=BORD,
                    autorange="reversed" if invert else True,
                ),
                yaxis=dict(showticklabels=False),
                showlegend=False,
                plot_bgcolor="white",
            ))
        return fig

    # ── Summary stats table ───────────────────────────────────────────────────
    metrics_summary = [
        ("Concentric Impulse",                 "CI (N·s)",        1, "",     False),
        ("RSI-modified",                       "RSI-mod",         3, "",     False),
        ("Peak Power / BM",                    "Peak Pwr/BM",     1, "",     False),
        ("30yd Split",                         "30yd (s)",        3, "s",    True),
        ("10yd Split",                         "10yd (s)",        3, "s",    True),
        ("Jump Height (Flight Time) in Inches","Jump Ht (in)",    2, " in",  False),
        ("Height",                             "Height (cm)",     1, " cm",  False),
        ("Mass",                               "Mass (kg)",       1, " kg",  False),
        ("Wingspan",                           "Wingspan (cm)",   1, " cm",  False),
        ("wingspan_advantage",                 "Wingspan Adv.",   1, " cm",  False),
    ]

    summary_rows = []
    for col, label, digits, suffix, inv in metrics_summary:
        data = ref_df[col].dropna()
        if len(data) < 2:
            continue
        summary_rows.append({
            "Metric":  label,
            "N":       int(len(data)),
            "Mean":    f"{data.mean():.{digits}f}{suffix}",
            "Median":  f"{data.median():.{digits}f}{suffix}",
            "p25":     f"{data.quantile(0.25):.{digits}f}{suffix}",
            "p75":     f"{data.quantile(0.75):.{digits}f}{suffix}",
            "p90":     f"{data.quantile(0.90 if not inv else 0.10):.{digits}f}{suffix}",
            "Min":     f"{data.min():.{digits}f}{suffix}",
            "Max":     f"{data.max():.{digits}f}{suffix}",
        })

    st.markdown('<p class="label" style="margin-top:4px">Summary Statistics</p>',
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True,
                 hide_index=True, key="ref_summary")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="label">Distributions</p>', unsafe_allow_html=True)

    # ── Metric groups ─────────────────────────────────────────────────────────
    ALL_METRICS = {
        "Athleticism": [
            ("Concentric Impulse",                  "CI (N·s)",       25, False, 1, ""),
            ("RSI-modified",                        "RSI-modified",   25, False, 3, ""),
            ("Peak Power / BM",                     "Peak Pwr/BM",    25, False, 1, ""),
            ("30yd Split",                          "30yd (s)",       25, True,  3, "s"),
            ("10yd Split",                          "10yd (s)",       20, True,  3, "s"),
            ("Jump Height (Flight Time) in Inches", "Jump Ht (in)",   20, False, 2, " in"),
        ],
        "Anthropometrics": [
            ("Height",             "Height (cm)",     20, False, 1, " cm"),
            ("Mass",               "Mass (kg)",        20, False, 1, " kg"),
            ("Wingspan",           "Wingspan (cm)",    20, False, 1, " cm"),
            ("wingspan_advantage", "Wingspan Adv.",    20, False, 1, " cm"),
            ("bmi_raw",            "BW/Ht Ratio",      20, False, 3, ""),
        ],

    }

    SECTION_COLORS = {
        "Athleticism":      RED,
        "Anthropometrics":  NAV,
    }

    # Population groups: (label, accent color, filter mask or None)
    POP_GROUPS = [
        ("All Players",      "#6b7fa3", None),
        ("Position Players", GREEN,     df["pos_group"].isin(["Catcher","Infielder","Outfielder"])),
        ("Pitchers",         RED,       df["pos_group"] == "Pitcher"),
        ("Catchers",         NAV,       df["pos_group"] == "Catcher"),
        ("Infielders",       GOLD,      df["pos_group"] == "Infielder"),
        ("Outfielders",      GREEN,     df["pos_group"] == "Outfielder"),
    ]

    # Apply the year filter from the top of the tab to each group
    def get_group_df(mask):
        base = df.copy() if ref_yr == "All years" else df[df["Year"] == int(ref_yr)].copy()
        return base[mask] if mask is not None else base

    # ── Render each metric section ─────────────────────────────────────────────
    for section_name, metrics in ALL_METRICS.items():
        sec_color = SECTION_COLORS[section_name]
        st.markdown(
            f'<div style="border-left:4px solid {sec_color};padding-left:10px;'
            f'margin:28px 0 12px 0">'
            f'<span style="font-size:11px;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:{sec_color}">{section_name}</span></div>',
            unsafe_allow_html=True)

        for metric_col, label, nbins, inv, digits, suffix in metrics:
            # Row header for this metric
            st.markdown(
                f'<p style="font-size:10px;font-weight:600;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:#6b7fa3;margin:12px 0 6px 0">'
                f'{label}</p>',
                unsafe_allow_html=True)

            # One column per population group
            group_cols = st.columns(len(POP_GROUPS))
            for col_widget, (grp_label, grp_color, grp_mask) in zip(group_cols, POP_GROUPS):
                grp_df  = get_group_df(grp_mask)
                data    = grp_df[metric_col].dropna() if metric_col in grp_df.columns else pd.Series([], dtype=float)
                n       = len(data)

                # Column group header
                col_widget.markdown(
                    f'<div style="text-align:center;padding:4px 0;margin-bottom:4px;'
                    f'border-bottom:2px solid {grp_color}">'
                    f'<span style="font-size:9px;font-weight:700;letter-spacing:0.1em;'
                    f'text-transform:uppercase;color:{grp_color}">{grp_label}</span>'
                    f'<span style="font-size:9px;color:#9AAAC0;margin-left:4px">n={n}</span>'
                    f'</div>',
                    unsafe_allow_html=True)

                if n < 4:
                    col_widget.markdown(
                        f'<div style="height:180px;display:flex;align-items:center;'
                        f'justify-content:center;color:#9AAAC0;font-size:11px;'
                        f'text-align:center">Insufficient<br>data</div>',
                        unsafe_allow_html=True)
                    continue

                q25 = data.quantile(0.25 if not inv else 0.75)
                q50 = data.median()
                q75 = data.quantile(0.75 if not inv else 0.25)

                if ref_chart == "Histogram":
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=data, nbinsx=nbins,
                        marker_color=grp_color,
                        marker_line=dict(color="white", width=0.4),
                        opacity=0.85,
                    ))
                    for q, lbl, dash, y_pos in [
                        (q25, "p25", "dash",   0.99),
                        (q50, "p50", "solid",  0.82),
                        (q75, "p75", "dash",   0.65),
                    ]:
                        fig.add_vline(
                            x=q, line_dash=dash,
                            line_color=NAV, line_width=1.5,
                            annotation=dict(
                                text=f"<b>{lbl}</b> {q:.{digits}f}{suffix}",
                                font=dict(color=NAV, size=8),
                                bgcolor="rgba(255,255,255,0.85)",
                                borderpad=2,
                                yref="paper", y=y_pos,
                                showarrow=False,
                                xanchor="left",
                            ),
                        )
                    fig.update_layout(
                        height=200,
                        margin=dict(l=20, r=8, t=30, b=30),
                        paper_bgcolor="white", plot_bgcolor="white",
                        font=dict(family="Arial", color=NAV, size=9),
                        xaxis=dict(tickfont=dict(size=8), showgrid=True,
                                   gridcolor=BORD, zeroline=False),
                        yaxis=dict(tickfont=dict(size=8), showgrid=True,
                                   gridcolor=BORD, title=""),
                        showlegend=False, template="plotly_white",
                    )
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        x=data, name="",
                        marker_color=grp_color,
                        line_color=NAV, line_width=1.5,
                        fillcolor=f"rgba(17,34,90,0.08)",
                        boxmean="sd",
                        boxpoints="outliers",
                        marker=dict(size=4, opacity=0.4, color=grp_color),
                    ))
                    fig.update_layout(
                        height=200,
                        margin=dict(l=20, r=8, t=30, b=30),
                        paper_bgcolor="white", plot_bgcolor="white",
                        font=dict(family="Arial", color=NAV, size=9),
                        xaxis=dict(tickfont=dict(size=8), showgrid=True,
                                   gridcolor=BORD,
                                   autorange="reversed" if inv else True),
                        yaxis=dict(showticklabels=False),
                        showlegend=False, template="plotly_white",
                    )

                col_widget.plotly_chart(
                    fig, use_container_width=True,
                    key=f"ref_{section_name}_{metric_col}_{grp_label}")

        st.markdown('<hr style="border-color:#E8ECF0;margin:8px 0">', unsafe_allow_html=True)

# =============================================================================
# TAB 4 — GUIDE
# =============================================================================
with tab_guide:
    def gs(title, accent=RED):
        st.markdown(
            f'<h3 style="border-left:4px solid {accent};padding-left:12px;'
            f'color:{NAV};margin:24px 0 10px 0">{title}</h3>',
            unsafe_allow_html=True)

    def score_block(name, formula, desc, accent=NAV):
        st.markdown(f"""
        <div style="padding:14px 16px;border-bottom:1px solid {BORD}">
            <span style="font-family:'Playfair Display',serif;font-size:16px;font-weight:700;
                color:{accent};margin-right:10px">{name}</span>
            <span style="font-size:12px;color:#6b7fa3;background:{SURF};padding:2px 8px;
                border-radius:4px;border:1px solid {BORD}">{formula}</span>
            <p style="font-size:13px;line-height:1.7;color:#2a3a5a;margin:8px 0 0 0">{desc}</p>
        </div>""", unsafe_allow_html=True)

    def arch_block(name, color, desc):
        st.markdown(f"""
        <div style="padding:14px 16px;border-bottom:1px solid {BORD}">
            <span style="background:{color};color:white;font-size:11px;font-weight:700;
                padding:3px 12px;border-radius:20px;letter-spacing:0.04em">{name}</span>
            <p style="font-size:13px;line-height:1.7;color:#2a3a5a;margin:8px 0 0 0">{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="background:white;border:1px solid {BORD};border-top:4px solid {NAV};'
        f'border-radius:10px;padding:20px 24px;margin-bottom:20px;'
        f'box-shadow:0 2px 8px rgba(17,34,90,0.06)">'
        f'<p style="font-size:10px;font-weight:700;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:#6b7fa3;margin:0 0 6px 0">HOW TO READ THIS DASHBOARD</p>'
        f'<h2 style="margin:0 0 6px 0;font-family:\'Playfair Display\',serif">Guide</h2>'
        f'<p style="font-size:14px;color:#6b7fa3;margin:0">'
        f'What the scores mean, how they\'re built, and what the archetypes tell you.</p></div>',
        unsafe_allow_html=True)

    gs("The Two Scores That Matter", RED)
    st.markdown(
        f'<p style="font-size:14px;line-height:1.75;color:#2a3a5a">'
        f'Every athlete is evaluated on two headline numbers — <strong>Athlete Quality</strong> '
        f'and <strong>Development Potential</strong>. Both are scaled 0–100 and shown prominently '
        f'at the top of every scorecard. These are the numbers to focus on.</p>',
        unsafe_allow_html=True)

    score_block("Athlete Quality", "CI × 0.35 + Sprint × 0.30 + RSI × 0.15 + Peak Power × 0.20",
        "How good an athlete they are right now, based on force plate and sprint performance. "
        "Weights are adjustable in the sidebar. Players without sprint data use a separate CI/RSI/Peak Power formula.", accent=RED)
    score_block("Development Potential", "Peak Power + Height + Leanness + School Type + Wingspan",
        "How much room they have to grow. Rewards tall, lean high schoolers with long wingspans and high relative power output. "
        "Weights are adjustable in the sidebar.", accent=NAV)
    score_block("Position Group Quality", "Same formula as Athlete Quality, ranked within position group only",
        "How the athlete compares to players at their position specifically — Pitchers vs Pitchers, Infielders vs Infielders, etc.")

    gs("Athlete Quality — Component Metrics", RED)
    for label, desc in [
        ("Concentric Impulse (CI)", "Total force applied during the upward phase of the jump (N·s). The primary measure of lower body power output. Higher = better."),
        ("30yd Sprint", "Time to cover 30 yards from a standing start (seconds). Lower = faster = better. 10yd and 20yd splits also recorded."),
        ("RSI-modified", "Reactive Strength Index — jump height divided by ground contact time (m/s). Measures explosive efficiency. Higher = better."),
        ("Peak Power / BM", "Peak power output during the jump normalized to body mass (W/kg). Higher = more powerful relative to size."),
    ]:
        st.markdown(f"""
        <div style="padding:10px 14px;border-bottom:1px solid {BORD}">
            <div style="font-size:13px;font-weight:600;color:{NAV};margin-bottom:3px">{label}</div>
            <div style="font-size:12px;color:#6b7fa3;line-height:1.5">{desc}</div>
        </div>""", unsafe_allow_html=True)

    gs("Development Potential — Component Factors", NAV)
    for label, desc in [
        ("Peak Power / BM", "Higher relative power = more athletic ceiling. Already used in quality score but a strong predictor of future output."),
        ("Height", "Taller athletes have more projectability across all positions. No position adjustment — taller is always scored higher."),
        ("Leanness (BW/Ht)", "Body weight divided by height. Lower ratio = leaner athlete = more room to add functional mass without losing athleticism."),
        ("School Type", "High school athletes score highest — more developmental runway ahead. College athletes are more proven but less projectable."),
        ("Wingspan Advantage", "Wingspan minus height. Positive = longer reach than height would predict. Particularly valuable for pitchers. Scaled as a percentile."),
    ]:
        st.markdown(f"""
        <div style="padding:10px 14px;border-bottom:1px solid {BORD}">
            <div style="font-size:13px;font-weight:600;color:{NAV};margin-bottom:3px">{label}</div>
            <div style="font-size:12px;color:#6b7fa3;line-height:1.5">{desc}</div>
        </div>""", unsafe_allow_html=True)

    gs("CMJ Archetypes", GOLD)
    st.markdown(
        f'<p style="font-size:13px;line-height:1.7;color:#2a3a5a;margin-bottom:10px">'
        f'Archetypes describe <em>how</em> an athlete jumps, not how well. '
        f'They are derived from six CMJ strategy features using robust z-scores and Mahalanobis distance.</p>',
        unsafe_allow_html=True)
    arch_block("Normal", "#4CAF82",
        "All strategy features within 0.7 SD of pool median. Typical, unremarkable mechanics.")
    arch_block("Long Loader", NAV,
        "Unusually deep countermovement combined with a long eccentric phase. More time and displacement in the loading phase.")
    arch_block("Front-Loaded Driver", RED,
        "High CI-100ms and front-loaded CI ratio — disproportionate drive force in the first 100ms of the push-off.")
    arch_block("Shallow Late-Driver", "#6b7fa3",
        "Shallow countermovement depth combined with a back-loaded impulse ratio — force production ramps up later in the drive.")
    arch_block("Unclassified", "#9AAAC0",
        "Unusual strategy profile that doesn't fit a named pattern. Check the CMJ strategy bar chart for what's driving the flag.")

    gs("CMJ Strategy Features", GOLD)
    for feat, desc in [
        ("Eccentric Duration",       "Time spent loading (moving downward). Longer = more deliberate, slower dip."),
        ("Concentric Duration",      "Time spent driving upward. Shorter = more explosive push-off."),
        ("Braking Phase Duration",   "Time between peak downward velocity and start of upward drive. Short = faster reversal."),
        ("Countermovement Depth",    "How far the center of mass drops. Deeper = more elastic energy stored."),
        ("Concentric Impulse–100ms", "Raw impulse in the first 100ms of upward drive. Proxy for early explosiveness."),
        ("CI100 : Total CI Ratio",   "Share of total concentric impulse produced in first 100ms. High = front-loaded strategy."),
    ]:
        st.markdown(f"""
        <div style="padding:10px 14px;border-bottom:1px solid {BORD}">
            <div style="font-size:13px;font-weight:600;color:{NAV};margin-bottom:3px">{feat}</div>
            <div style="font-size:12px;color:#6b7fa3;line-height:1.5">{desc}</div>
        </div>""", unsafe_allow_html=True)

    gs("Position Groups", "#6b7fa3")
    for grp, positions, desc in [
        ("Pitchers",    "SP, RHP, LHP, RP, TWP", "Percentiles calculated within pitcher pool only. Wingspan included in potential weighting."),
        ("Catchers",    "C",                      "Smallest group — percentiles less stable with fewer athletes."),
        ("Infielders",  "SS, 3B, 2B, 1B",         "Largest position player group."),
        ("Outfielders", "CF, LF, RF",              "Speed and power profile tends to skew higher than infielders."),
    ]:
        st.markdown(f"""
        <div style="padding:10px 14px;border-bottom:1px solid {BORD}">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:3px">
                <span style="font-size:13px;font-weight:600;color:{NAV}">{grp}</span>
                <span style="font-size:11px;color:#9AAAC0;background:{SURF};padding:1px 8px;
                    border-radius:10px;border:1px solid {BORD}">{positions}</span>
            </div>
            <div style="font-size:12px;color:#6b7fa3;line-height:1.5">{desc}</div>
        </div>""", unsafe_allow_html=True)

    gs("What This Is Not", "#9AAAC0")
    for para in [
        "These scores are <strong>not predictions of major league success</strong>. They describe current physical output and development room — one input among many.",
        "The potential score does not account for pitchability, bat-to-ball skill, makeup, or injury history.",
        "Scores become more meaningful as data accumulates. Single-year athletes are informative but tentative.",
        "Players with missing sprint data are scored on CI/RSI/Peak Power only — this is noted in the scorecard. Missing data is not penalized.",
    ]:
        st.markdown(
            f'<p style="font-size:14px;line-height:1.75;color:#2a3a5a;margin-bottom:10px">{para}</p>',
            unsafe_allow_html=True)


# =============================================================================
# TAB 5 — REFERENCE
# =============================================================================
with tab_ref:
    st.markdown(
        f'<div style="background:white;border:1px solid {BORD};border-top:4px solid {NAV};'
        f'border-radius:10px;padding:20px 24px;margin-bottom:20px;'
        f'box-shadow:0 2px 8px rgba(17,34,90,0.06)">'
        f'<p style="font-size:10px;font-weight:700;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:#6b7fa3;margin:0 0 6px 0">POPULATION NORMS</p>'
        f'<h2 style="margin:0 0 6px 0;font-family:\'Playfair Display\',serif">Reference Tables</h2>'
        f'<p style="font-size:14px;color:#6b7fa3;margin:0">'
        f'Mean ± 1 SD for each metric, broken out by position group and year.</p></div>',
        unsafe_allow_html=True)

    rr1, rr2 = st.columns([1, 1])
    with rr1:
        ref_yr_opts = ["All years"] + sorted(df["Year"].dropna().unique().astype(int).tolist(), reverse=True)
        ref_yr_sel  = st.selectbox("Year", ref_yr_opts, key="ref_tbl_yr")
    with rr2:
        ref_pos_opts = ["All positions", "Pitcher", "Catcher", "Infielder", "Outfielder"]
        ref_pos_sel  = st.selectbox("Position Group", ref_pos_opts, key="ref_tbl_pos")

    ref_base = df.copy()
    if ref_yr_sel != "All years":
        ref_base = ref_base[ref_base["Year"] == int(ref_yr_sel)]
    if ref_pos_sel != "All positions":
        ref_base = ref_base[ref_base["pos_group"] == ref_pos_sel]

    METRIC_GROUPS = {
        "Athleticism": [
            ("Concentric Impulse",                  "CI (N·s)",       1, ""),
            ("RSI-modified",                        "RSI-modified",   3, ""),
            ("Peak Power / BM",                     "Peak Pwr/BM",    1, ""),
            ("30yd Split",                          "30yd (s)",       3, "s"),
            ("10yd Split",                          "10yd (s)",       3, "s"),
            ("Jump Height (Flight Time) in Inches", "Jump Ht (in)",   2, " in"),
        ],
        "Anthropometrics": [
            ("Height",             "Height (cm)",      1, " cm"),
            ("Mass",               "Mass (kg)",         1, " kg"),
            ("Wingspan",           "Wingspan (cm)",     1, " cm"),
            ("wingspan_advantage", "Wingspan Adv. (cm)",1, " cm"),
        ],

    }

    for section, metrics in METRIC_GROUPS.items():
        st.markdown(
            f'<div style="border-left:4px solid {RED};padding-left:10px;margin:20px 0 10px 0">'
            f'<span style="font-size:10px;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:{RED}">{section}</span></div>',
            unsafe_allow_html=True)

        rows = []
        for col, label, digits, suffix in metrics:
            if col not in ref_base.columns:
                continue
            data = ref_base[col].dropna()
            if len(data) < 2:
                continue
            rows.append({
                "Metric":  label,
                "N":       int(len(data)),
                "Mean":    f"{data.mean():.{digits}f}{suffix}",
                "SD":      f"{data.std(ddof=1):.{digits}f}{suffix}",
                "p10":     f"{data.quantile(0.10):.{digits}f}{suffix}",
                "p25":     f"{data.quantile(0.25):.{digits}f}{suffix}",
                "Median":  f"{data.median():.{digits}f}{suffix}",
                "p75":     f"{data.quantile(0.75):.{digits}f}{suffix}",
                "p90":     f"{data.quantile(0.90):.{digits}f}{suffix}",
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, key=f"ref_tbl_{section}")

    # ── School level comparison ───────────────────────────────────────────────
    st.markdown(
        f'<div style="border-left:4px solid {NAV};padding-left:10px;margin:28px 0 10px 0">'
        f'<span style="font-size:10px;font-weight:700;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:{NAV}">By School Level</span></div>',
        unsafe_allow_html=True)

    school_metrics = [
        ("Concentric Impulse",                  "CI (N·s)",     1),
        ("RSI-modified",                        "RSI-mod",      3),
        ("Peak Power / BM",                     "Pk Pwr/BM",    1),
        ("30yd Split",                          "30yd (s)",     3),
        ("Jump Height (Flight Time) in Inches", "Jump Ht (in)", 2),
        ("Height",                              "Height (cm)",  1),
        ("Mass",                                "Mass (kg)",    1),
        ("Wingspan",                            "Wingspan (cm)",1),
        ("wingspan_advantage",                  "Wing Adv.",    1),
    ]

    base_yr = df.copy()
    if ref_yr_sel != "All years":
        base_yr = base_yr[base_yr["Year"] == int(ref_yr_sel)]

    # Filter by position group if selected
    if ref_pos_sel != "All positions":
        base_yr = base_yr[base_yr["pos_group"] == ref_pos_sel]

    school_levels = ["High School", "Junior College", "4-Year College"]
    school_rows = []
    for level in school_levels:
        lvl_df = base_yr[base_yr["School Type"] == level]
        if len(lvl_df) == 0:
            continue
        row = {"School Level": level, "N": len(lvl_df)}
        for col, label, digits in school_metrics:
            if col in lvl_df.columns:
                data = lvl_df[col].dropna()
                if len(data) > 0:
                    row[label] = f"{data.median():.{digits}f}"
                    row[f"{label} (p25–p75)"] = f"{data.quantile(0.25):.{digits}f} – {data.quantile(0.75):.{digits}f}"
                else:
                    row[label] = "—"
                    row[f"{label} (p25–p75)"] = "—"
            else:
                row[label] = "—"
                row[f"{label} (p25–p75)"] = "—"
        school_rows.append(row)

    if school_rows:
        # Show median table
        median_cols = ["School Level", "N"] + [lbl for _, lbl, _ in school_metrics]
        iqr_cols    = ["School Level", "N"] + [f"{lbl} (p25–p75)" for _, lbl, _ in school_metrics]
        median_df   = pd.DataFrame(school_rows)[median_cols]
        iqr_df      = pd.DataFrame(school_rows)[iqr_cols]

        st.markdown('<p style="font-size:11px;font-weight:600;color:#6b7fa3;margin:8px 0 4px 0">Medians</p>',
                    unsafe_allow_html=True)
        st.dataframe(median_df, use_container_width=True, hide_index=True, key="ref_school_med")

        st.markdown('<p style="font-size:11px;font-weight:600;color:#6b7fa3;margin:12px 0 4px 0">Interquartile Range (p25 – p75)</p>',
                    unsafe_allow_html=True)
        st.dataframe(iqr_df.rename(columns={f"{lbl} (p25–p75)": lbl for _, lbl, _ in school_metrics}),
                     use_container_width=True, hide_index=True, key="ref_school_iqr")

    st.caption("Values are medians / IQR within each school level for the selected filters.")
