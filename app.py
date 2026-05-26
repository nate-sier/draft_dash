# VERSION: athlete_scorecard_profile_bars_v7 -- radar replaced with horizontal percentile profile bars
# VERSION: sidebar_compact_filters_v6 -- leaderboard filters moved to sidebar; compact min/max inputs; seated height removed
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
SHEET_ID = "1RKyeb4CfU4wACUKqpbiR_--PJJ9KKO-LH-Ov9e6zNQs"

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
    for i, (lo, hi, _) in enumerate(CI_TIERS):
        if (lo is None or ci >= lo) and (hi is None or ci < hi):
            return i
    return len(CI_TIERS) - 1

def ci_next_tier_target(ci):
    idx = ci_tier_index(ci)
    if idx < 0 or idx >= len(CI_TIERS) - 1:
        return None
    lo, hi, label = CI_TIERS[idx + 1]
    return label, lo

def lbs_to_target(ci, mass_kg, target_ci, penalty=0.03):
    if pd.isna(ci) or pd.isna(mass_kg) or mass_kg == 0 or ci == 0:
        return np.nan
    ci_per_kg_adj = (ci / mass_kg) * (1 - penalty)
    if ci_per_kg_adj <= 0:
        return np.nan
    new_mass_kg = target_ci / ci_per_kg_adj
    gain_kg = new_mass_kg - mass_kg
    return gain_kg * 2.20462

def weight_gain_classification(lbs):
    if pd.isna(lbs) or lbs <= 0:
        return "At or above target"
    if lbs < 10:
        return "Standard Off-Season"
    if lbs < 20:
        return "Needs Strength Camp"
    return "Needs Development Year"

WEIGHT_CLASS_COLORS = {
    "At or above target":     "#4CAF82",
    "Standard Off-Season":    "#4CAF82",
    "Needs Strength Camp":    "#E2C188",
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
SLATE = "#6b7fa3"

# ─── CMJ Strategy ─────────────────────────────────────────────────────────────
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

ARCHETYPE_COLORS = {
    "Normal":              "#4CAF82",
    "Long Loader":         "#11225A",
    "Front-Loaded Driver": "#BA0C2F",
    "Shallow Late-Driver": "#6b7fa3",
    "Unclassified":        "#9AAAC0",
}

_PLOT_BASE = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor=SURF,
    font=dict(family="'Source Sans 3', Arial, sans-serif", color=NAV),
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
    if pd.isna(cm): return "—"
    try:
        total_in = float(cm) / 2.54
        feet = int(total_in // 12)
        inches = int(round(total_in % 12))
        if inches == 12:
            feet += 1; inches = 0
        return f"{feet}'{inches}\""
    except Exception:
        return "—"

def fmt_mass(kg):
    if pd.isna(kg): return "—"
    try: return f"{float(kg) * 2.20462:.1f} lbs"
    except: return "—"

def fmt_wingspan(cm):
    return fmt_height(cm)

def fmt_wingspan_adv(cm):
    if pd.isna(cm): return "—"
    try:
        val = float(cm) / 2.54
        sign = "+" if val >= 0 else ""
        return f'{sign}{val:.1f}"'
    except: return "—"

def fmt_bwht(kg, cm):
    if pd.isna(kg) or pd.isna(cm): return "—"
    try: return f"{(float(kg) * 2.20462) / (float(cm) / 2.54):.3f}"
    except: return "—"

def sf(v):
    try:
        f = float(v)
        return f if not np.isnan(f) else np.nan
    except: return np.nan

def pct_sfx(p):
    if pd.isna(p): return "—"
    p = int(round(p))
    if 11 <= p % 100 <= 13: return f"{p}th"
    return f"{p}{({1:'st',2:'nd',3:'rd'}.get(p%10,'th'))}"

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

# ─── Google Sheets loader ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading data…")
def load_data(_v=3):
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

    df = fp_raw.merge(isak_raw.drop(columns=["athleteName"], errors="ignore"),
                      on=["playerID","Year"], how="outer")
    df = df.merge(sprint_raw.drop(columns=["athleteName"], errors="ignore"),
                  on=["playerID","Year"], how="outer")

    if "playerID" in pos_raw.columns and "Position" in pos_raw.columns:
        pos_raw["playerID"] = pos_raw["playerID"].astype(str).str.strip()
        df = df.merge(pos_raw[["playerID","Position"]].drop_duplicates("playerID"),
                      on="playerID", how="left")
    else:
        df["Position"] = ""

    name_map = {}
    for src in [fp_raw, isak_raw, sprint_raw]:
        name_map.update(src[["playerID","athleteName"]].dropna()
                        .drop_duplicates("playerID").set_index("playerID")["athleteName"].to_dict())
    df["athleteName"] = df["athleteName"].where(
        df["athleteName"].notna(), df["playerID"].map(name_map))

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
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    df["CI100ms_to_TotalCI_Ratio"] = safe_div(
        df["Concentric Impulse-100ms"], df["Concentric Impulse"])
    strategy_features = BASE_STRATEGY_FEATURES + ["CI100ms_to_TotalCI_Ratio"]

    for c in strategy_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    for c in strategy_features:
        df[f"rz_{c}"] = robust_z(df[c])
    all_rz_cols = [f"rz_{f}" for f in strategy_features]

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
        cov     = np.cov(X_pca, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        mu_pca  = X_pca.mean(axis=0)
        diff    = X_pca - mu_pca
        dist2   = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        df["strategy_distance_raw"] = np.sqrt(np.abs(dist2))
    df["strategy_distance_score"] = pct_rank(df["strategy_distance_raw"])

    df["archetype"] = df.apply(lambda r: label_archetype(r.to_dict(), all_rz_cols), axis=1)

    df["pos_group"] = df["Position"].astype(str).map(pos_group)

    # ── Percentiles ───────────────────────────────────────────────────────────
    df["ci_pct_alltime"]  = pct_rank(df["Concentric Impulse"])
    df["p1_ci_pct_alltime"] = pct_rank(df["P1 Concentric Impulse"])
    df["ci100_pct_alltime"] = pct_rank(df["Concentric Impulse-100ms"])
    df["rsi_pct_alltime"] = pct_rank(df["RSI-modified"])
    df["pp_pct_alltime"]  = pct_rank(df["Peak Power / BM"])
    df["jump_height_pct_alltime"] = pct_rank(df["Jump Height (Flight Time) in Inches"])
    sprint_mask = df["30yd Split"].notna()
    df["sprint_pct_alltime"] = np.nan
    if sprint_mask.any():
        df.loc[sprint_mask, "sprint_pct_alltime"] = (
            100 - pct_rank(df.loc[sprint_mask, "30yd Split"]))

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
            has = sub[col].notna()
            if has.any():
                r = pct_rank(sub.loc[has, col])
                df.loc[sub.index[has], out] = 100 - r if inv else r

    def pos_group_aq(row):
        grp = row.get("pos_group", "Unknown")
        if grp == "Unknown": return np.nan
        g = grp.lower()
        ci_p  = row.get(f"ci_pct_{g}", np.nan)
        rsi_p = row.get(f"rsi_pct_{g}", np.nan)
        pp_p  = row.get(f"pp_pct_{g}", np.nan)
        spr_p = row.get(f"sprint_pct_{g}", np.nan)
        if pd.notna(spr_p):
            return (w_ci*(ci_p or 50)+w_sprint*spr_p+w_rsi*(rsi_p or 50)+w_pp*(pp_p or 50))
        else:
            return (w_ci_ns*(ci_p or 50)+w_rsi_ns*(rsi_p or 50)+w_pp_ns*(pp_p or 50))

    df["aq_pos_raw"] = df.apply(pos_group_aq, axis=1)
    df["aq_pos_score"] = np.nan
    for grp in ["Pitcher", "Catcher", "Infielder", "Outfielder"]:
        idx = df["pos_group"] == grp
        if idx.any():
            df.loc[idx, "aq_pos_score"] = scaled_0_100(df.loc[idx, "aq_pos_raw"].values)

    def aq_raw(row):
        if pd.notna(row["sprint_pct_alltime"]):
            return (w_ci*row["ci_pct_alltime"]+w_sprint*row["sprint_pct_alltime"]+
                    w_rsi*row["rsi_pct_alltime"]+w_pp*row["pp_pct_alltime"])
        else:
            return (w_ci_ns*row["ci_pct_alltime"]+w_rsi_ns*row["rsi_pct_alltime"]+
                    w_pp_ns*row["pp_pct_alltime"])

    df["athlete_quality_raw"]   = df.apply(aq_raw, axis=1)
    df["athlete_quality_score"] = scaled_0_100(df["athlete_quality_raw"].values)

    for col, pct_col, inv in [
        ("Concentric Impulse", "ci_pct_yr",    False),
        ("RSI-modified",       "rsi_pct_yr",   False),
        ("Peak Power / BM",    "pp_pct_yr",    False),
        ("30yd Split",         "sprint_pct_yr",True),
    ]:
        df[pct_col] = np.nan
        for yr, idx in df.groupby("Year").groups.items():
            s = df.loc[idx, col]
            has = s.notna()
            if has.any():
                r = pct_rank(s[has])
                df.loc[idx[has], pct_col] = 100 - r if inv else r

    def aq_yr_raw(row):
        if pd.notna(row["sprint_pct_yr"]):
            return (w_ci*row["ci_pct_yr"]+w_sprint*row["sprint_pct_yr"]+
                    w_rsi*row["rsi_pct_yr"]+w_pp*row["pp_pct_yr"])
        else:
            return (w_ci_ns*row["ci_pct_yr"]+w_rsi_ns*row["rsi_pct_yr"]+
                    w_pp_ns*row["pp_pct_yr"])

    df["aq_yr_raw"]   = df.apply(aq_yr_raw, axis=1)
    df["aq_score_yr"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "aq_score_yr"] = scaled_0_100(df.loc[idx, "aq_yr_raw"].values)

    # ── Potential Score ───────────────────────────────────────────────────────
    df["pp_pct"]      = df["pp_pct_alltime"]
    df["height_pct"]  = pct_rank(df["Height"])
    df["bmi_raw"]     = safe_div(df["Mass"] * 2.20462, df["Height"] / 2.54)
    df["bmi_pct"]     = pct_rank(df["bmi_raw"])
    school_score_map  = {"High School": 100, "4-Year College": 60, "Junior College": 40}
    df["school_score"] = df["School Type"].map(school_score_map).fillna(50)
    df["wingspan_advantage"] = df["Wingspan"] - df["Height"]
    df["wingspan_pct"]       = pct_rank(df["wingspan_advantage"])

    def pot_score(row):
        def sv(key):
            v = row.get(key)
            try:
                f = float(v)
                return f if not np.isnan(f) else None
            except: return None
        components = {
            "pp_pct":       (sv("pp_pct"),       wp_peakpow),
            "height_pct":   (sv("height_pct"),   wp_height),
            "bmi_pct":      (sv("bmi_pct"),      wp_bmi),
            "school_score": (sv("school_score"), wp_school),
            "wingspan_pct": (sv("wingspan_pct"), wp_wingspan),
        }
        total_w = sum(w for _, (v, w) in components.items() if v is not None)
        if total_w == 0: return np.nan
        return sum(v*w for _, (v, w) in components.items() if v is not None) / total_w * 100

    df["potential_raw"]   = df.apply(pot_score, axis=1)
    df["potential_score"] = scaled_0_100(df["potential_raw"].values)
    df["potential_score_yr"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "potential_score_yr"] = scaled_0_100(df.loc[idx, "potential_raw"].values)

    df["programming_category"] = df.apply(
        lambda r: programming_category(
            r.get("Concentric Impulse", np.nan),
            r.get("P1 Concentric Impulse", np.nan)), axis=1)

    df["overall_rank"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "overall_rank"] = (
            df.loc[idx, "athlete_quality_score"].rank(ascending=False, method="min"))
    df["overall_rank"] = df["overall_rank"].astype("Int64")

    df["ci_tier"]     = df["Concentric Impulse"].apply(ci_tier_label)
    df["ci_tier_idx"] = df["Concentric Impulse"].apply(ci_tier_index)

    TARGET_315 = 315.0

    def ci_pathway(row):
        ci      = row.get("Concentric Impulse", np.nan)
        mass_kg = row.get("Mass", np.nan)
        result  = {}
        if pd.notna(ci) and ci < TARGET_315 and pd.notna(mass_kg):
            result["lbs_to_315"]       = lbs_to_target(ci, mass_kg, TARGET_315)
            result["weight_class_315"] = weight_gain_classification(result["lbs_to_315"])
        elif pd.notna(ci) and ci >= TARGET_315:
            result["lbs_to_315"]       = 0.0
            result["weight_class_315"] = "At or above target"
        else:
            result["lbs_to_315"]       = np.nan
            result["weight_class_315"] = "—"
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
    text-transform: uppercase; color: {SLATE};
}}
div[data-testid="metric-container"] div[data-testid="metric-value"] {{
    font-family: 'Playfair Display', serif; font-size: 28px; color: {RED};
}}

.stTabs [data-baseweb="tab-list"] {{
    background: white; border-bottom: 2px solid {BORD}; gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: {SLATE}; padding: 12px 28px;
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
    text-transform: uppercase; color: {SLATE}; margin-bottom: 6px;
}}
.stat-row {{ display: flex; align-items: baseline; margin-bottom: 7px; font-size: 13px; }}
.stat-label {{ color: {SLATE}; min-width: 160px; font-size: 12px; }}
.stat-val {{ font-weight: 600; color: {NAV}; }}

.badge {{
    display: inline-block; color: white; font-size: 11px; font-weight: 700;
    padding: 3px 12px; border-radius: 20px; letter-spacing: 0.04em;
}}
.score-big {{
    font-family: 'Playfair Display', serif; font-size: 42px;
    font-weight: 900; line-height: 1;
}}
.grad-bar {{
    height: 4px;
    background: linear-gradient(90deg, {RED} 0%, {NAV} 60%, {GOLD} 100%);
    border-radius: 2px; margin-bottom: 0;
}}

/* Wingspan panel */
.wing-panel {{
    background: linear-gradient(135deg, {NAV} 0%, #1a3275 100%);
    border-radius: 12px; padding: 22px 24px; color: white; margin-bottom: 16px;
    box-shadow: 0 6px 24px rgba(17,34,90,0.20);
}}
.wing-stat {{
    background: rgba(255,255,255,0.10); border-radius: 8px;
    padding: 12px 16px; text-align: center;
}}
.wing-stat-val {{
    font-family: 'Playfair Display', serif; font-size: 26px;
    font-weight: 900; color: white; line-height: 1.1;
}}
.wing-stat-lbl {{
    font-size: 9px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: rgba(255,255,255,0.55); margin-top: 3px;
}}
.wing-adv-pos {{ color: #4CEFA0; }}
.wing-adv-neg {{ color: #FF6B6B; }}
.wing-adv-neu {{ color: {GOLD}; }}
</style>
"""

# ─── Chart builders ───────────────────────────────────────────────────────────
def make_gauge(value, title, color=RED):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value if pd.notna(value) else 0,
        title={"text": title, "font": {"size": 12, "color": NAV}},
        number={"font": {"size": 26, "color": color, "family": "Playfair Display, serif"}},
        gauge={
            "axis": {"range": [0,100], "tickfont": {"size": 9, "color": "#9AAAC0"}},
            "bar":  {"color": color, "thickness": 0.6},
            "bgcolor": SURF, "borderwidth": 0,
            "steps": [
                {"range": [0,  33], "color": "#f0f3f8"},
                {"range": [33, 66], "color": "#e4eaf3"},
                {"range": [66,100], "color": "#d8e0ed"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor="white", font=dict(family="Arial"))
    return fig

def make_radar(row, label="Athlete", is_pitcher=False):
    """Horizontal percentile profile used in place of the old radar chart."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    def val(key):
        try:
            v = row.get(key, np.nan)
            f = float(v)
            return f if not np.isnan(f) else np.nan
        except Exception:
            return np.nan

    def pct(key, default=np.nan):
        v = val(key)
        if pd.isna(v):
            return default
        return max(0.0, min(100.0, v))

    def raw_fmt(key, digits=1, suffix=""):
        v = val(key)
        return "—" if pd.isna(v) else f"{v:.{digits}f}{suffix}"

    sections = []
    sections.append(("Force Plate", [
        ("CI", pct("ci_pct_alltime"), raw_fmt("Concentric Impulse", 1)),
        ("P1 Conc. Impulse", pct("p1_ci_pct_alltime"), raw_fmt("P1 Concentric Impulse", 1)),
        ("CI-100ms", pct("ci100_pct_alltime"), raw_fmt("Concentric Impulse-100ms", 1)),
        ("RSI-modified", pct("rsi_pct_alltime"), raw_fmt("RSI-modified", 3)),
        ("Peak Power / BM", pct("pp_pct_alltime"), raw_fmt("Peak Power / BM", 1)),
        ("Jump Height", pct("jump_height_pct_alltime"), raw_fmt("Jump Height (Flight Time) in Inches", 2, " in")),
    ]))

    sprint_rows = []
    if pd.notna(val("30yd Split")):
        sprint_rows.append(("30yd Sprint", pct("sprint_pct_alltime"), raw_fmt("30yd Split", 3, "s")))
    # 10yd and 20yd splits are shown as raw context if present, but not percentiled unless columns exist later.
    if pd.notna(val("10yd Split")):
        sprint_rows.append(("10yd Split", pct("ten_yd_pct_alltime", 50), raw_fmt("10yd Split", 3, "s")))
    if pd.notna(val("20yd Split")):
        sprint_rows.append(("20yd Split", pct("twenty_yd_pct_alltime", 50), raw_fmt("20yd Split", 3, "s")))
    if sprint_rows:
        sections.append(("Sprint", sprint_rows))

    anthro_rows = [
        ("Height", pct("height_pct"), fmt_height(val("Height"))),
        ("Mass", pct("mass_pct_alltime", pct("bmi_pct", 50)), fmt_mass(val("Mass"))),
        ("Wingspan", pct("wingspan_pct"), fmt_wingspan(val("Wingspan"))),
        ("BW/Ht", pct("bmi_pct"), pct_sfx(pct("bmi_pct")) if pd.notna(pct("bmi_pct")) else "—"),
    ]
    if is_pitcher:
        anthro_rows.insert(3, ("Wing Adv.", pct("wingspan_pct"), fmt_wingspan_adv(val("wingspan_advantage"))))
    sections.append(("Anthropometrics", anthro_rows))

    labels, vals, raw_vals, section_for_row = [], [], [], []
    y = []
    section_title_y = []
    cur_y = 0
    for section, rows in sections:
        section_title_y.append((section, cur_y))
        cur_y -= 0.75
        for lab, pc, rv in rows:
            labels.append(lab)
            vals.append(0 if pd.isna(pc) else pc)
            raw_vals.append(rv)
            section_for_row.append(section)
            y.append(cur_y)
            cur_y -= 1
        cur_y -= 0.45

    def color_for(v):
        if pd.isna(v):
            return "#9AAAC0"
        if v >= 75:
            return RED
        if v >= 50:
            return "#AFC4C9"
        return "#6B83B6"

    bar_colors = [color_for(v) for v in vals]

    fig = go.Figure()

    # Gray full scale rails, then colored percentile bars.
    fig.add_trace(go.Bar(
        x=[100] * len(y), y=y, orientation="h", marker_color="rgba(210,213,218,0.70)",
        width=0.72, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=vals, y=y, orientation="h", marker_color=bar_colors,
        width=0.72, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=vals, y=y, mode="markers+text",
        marker=dict(size=34, color=bar_colors, line=dict(color="white", width=2)),
        text=["—" if pd.isna(v) else f"{int(round(v))}" for v in vals],
        textfont=dict(color="white", size=13, family="Arial Black"),
        textposition="middle center", hoverinfo="skip", showlegend=False,
    ))

    # Dashed row separators.
    for yy in y:
        fig.add_shape(type="line", x0=-25, x1=118, y0=yy - 0.5, y1=yy - 0.5,
                      line=dict(color="rgba(150,150,150,0.35)", width=1, dash="dash"), layer="below")

    # Vertical percentile guide lines.
    for xg in [0, 25, 50, 75, 100]:
        fig.add_vline(x=xg, line_width=1, line_dash="dash" if xg in [25,50,75] else "solid",
                      line_color="rgba(170,170,170,0.55)")

    # Metric labels and raw values.
    for lab, yy, rv in zip(labels, y, raw_vals):
        fig.add_annotation(x=-4, y=yy, text=lab, showarrow=False, xanchor="right",
                           font=dict(size=12, color="#20232A"))
        fig.add_annotation(x=110, y=yy, text=rv, showarrow=False, xanchor="center",
                           font=dict(size=12, color="#20232A"))

    # Section titles.
    for section, yy in section_title_y:
        fig.add_annotation(x=-25, y=yy, text=f"<b>{section}</b>", showarrow=False,
                           xanchor="left", font=dict(size=16, color="#20232A"))
        fig.add_shape(type="line", x0=-25, x1=118, y0=yy - 0.35, y1=yy - 0.35,
                      line=dict(color="rgba(120,120,120,0.45)", width=1), layer="below")

    fig.update_layout(
        barmode="overlay",
        height=max(440, 62 * len(y) + 70 * len(sections)),
        margin=dict(l=10, r=10, t=8, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Source Sans 3, Arial, sans-serif", color=NAV),
        xaxis=dict(range=[-26, 122], tickmode="array", tickvals=[0,25,50,75,100],
                   tickfont=dict(size=10, color="#555"), showgrid=False, zeroline=False),
        yaxis=dict(visible=False, range=[min(y)-0.7, max(t for _, t in section_title_y)+0.5]),
        showlegend=False,
    )
    return fig

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
        height=320, margin=dict(l=30,r=20,t=50,b=100),
    ))
    return fig

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
        height=180, margin=dict(l=30,r=10,t=36,b=30),
        xaxis=dict(tickmode="array", tickvals=g["Year"].tolist(), tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed" if invert else True),
        showlegend=False,
    ))
    return fig

def make_wingspan_bar(row, df_all):
    """Horizontal bar showing athlete wingspan vs pool distribution."""
    pool = df_all["wingspan_advantage"].dropna()
    if pd.isna(row.get("wingspan_advantage")) or len(pool) == 0:
        return None
    val = float(row["wingspan_advantage"])
    p10 = float(pool.quantile(0.10))
    p25 = float(pool.quantile(0.25))
    p50 = float(pool.quantile(0.50))
    p75 = float(pool.quantile(0.75))
    p90 = float(pool.quantile(0.90))
    lo  = float(pool.min()) - 1
    hi  = float(pool.max()) + 1

    fig = go.Figure()
    # Background zone fills
    for zone_lo, zone_hi, zone_col, zone_lbl in [
        (lo,  p25,  "rgba(186,12,47,0.08)",    "Bottom 25%"),
        (p25, p75,  "rgba(226,193,136,0.12)",   "Middle 50%"),
        (p75, hi,   "rgba(76,175,130,0.12)",    "Top 25%"),
    ]:
        fig.add_shape(type="rect", x0=zone_lo, x1=zone_hi, y0=-0.5, y1=0.5,
                      fillcolor=zone_col, line_width=0, layer="below")
    # Percentile lines
    for pval, plbl, pcol in [
        (p25,"25th","#E2C188"),(p50,"50th","#9AAAC0"),(p75,"75th","#4CAF82")
    ]:
        fig.add_vline(x=pval, line_width=1.5, line_dash="dot", line_color=pcol,
                      annotation_text=plbl, annotation_position="top",
                      annotation_font=dict(size=9, color=pcol))
    # Athlete marker
    fig.add_trace(go.Scatter(
        x=[val], y=[0], mode="markers+text",
        marker=dict(size=18, color=RED, symbol="diamond",
                    line=dict(width=2, color="white")),
        text=[f"{val/2.54:+.1f}\""], textposition="top center",
        textfont=dict(size=11, color=RED, family="Playfair Display"),
        showlegend=False,
    ))
    fig.update_layout(**_layout(
        height=130,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(range=[lo, hi], showgrid=False, zeroline=False,
                   tickvals=[p10,p25,p50,p75,p90],
                   ticktext=[f"{v/2.54:+.1f}\"" for v in [p10,p25,p50,p75,p90]],
                   tickfont=dict(size=9)),
        yaxis=dict(visible=False),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text="Wingspan Advantage vs Pool", font=dict(size=12, color=NAV), x=0),
    ))
    return fig

# ─── Auth ─────────────────────────────────────────────────────────────────────
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
    col = st.columns([1,2,1])[1]
    with col:
        pwd = st.text_input("Password", type="password",
                            label_visibility="collapsed", placeholder="Enter password")
        if st.button("Enter", use_container_width=True):
            if pwd == os.environ.get("DASHBOARD_PASSWORD", "NationalsDraft"):
                st.session_state.auth = True; st.rerun()
            else:
                st.error("Incorrect password.")
    return False

# ─── App entry ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NationalsDraft · Athlete Scorecard",
    page_icon="⚾", layout="wide",
    initial_sidebar_state="expanded",
)

if not check_password(): st.stop()

st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="grad-bar"></div>', unsafe_allow_html=True)

# ─── Fixed weights / refresh ──────────────────────────────────────────────────
# Sidebar sliders removed. These keep the original default weights.
w_ci     = 0.35
w_sprint = 0.30
w_rsi    = 0.15
w_pp     = 0.20

w_ci_ns  = 0.45
w_rsi_ns = 0.20
w_pp_ns  = 0.35

with st.sidebar:
    st.markdown(f'<p class="label">Data</p>', unsafe_allow_html=True)
    if st.button("↻ Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.rerun()

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

# ─── App header ───────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([3,1])
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
        f'<span style="font-size:12px;color:{SLATE};margin-left:4px">athletes</span><br>'
        f'<span style="font-family:\'Playfair Display\',serif;font-size:28px;font-weight:700;color:{NAV}">'
        f'{int(df["Year"].nunique())}</span>'
        f'<span style="font-size:12px;color:{SLATE};margin-left:4px">years</span></div>',
        unsafe_allow_html=True)
st.markdown('<hr style="margin:8px 0 0 0;border-color:#E8ECF0">', unsafe_allow_html=True)

tab_board, tab_card = st.tabs(["Leaderboard", "Athlete Scorecard"])

# =============================================================================
# TAB 1 — LEADERBOARD
# =============================================================================
with tab_board:

    # -------------------------------------------------------------------------
    # Leaderboard setup
    # -------------------------------------------------------------------------
    df_lb = df.copy()

    # Create display/filter columns in the units scouts actually want to see.
    df_lb["Height_in"] = pd.to_numeric(df_lb.get("Height"), errors="coerce") / 2.54
    df_lb["Mass_lbs"] = pd.to_numeric(df_lb.get("Mass"), errors="coerce") * 2.20462
    df_lb["SeatedHeight_in"] = pd.to_numeric(df_lb.get("Seated Height"), errors="coerce") / 2.54
    df_lb["Wingspan_in"] = pd.to_numeric(df_lb.get("Wingspan"), errors="coerce") / 2.54
    df_lb["Wing_Adv_in"] = pd.to_numeric(df_lb.get("wingspan_advantage"), errors="coerce") / 2.54
    df_lb["BW_Ht_Ratio"] = pd.to_numeric(df_lb.get("bmi_raw"), errors="coerce")
    df_lb["BW_Ht_Pct"] = pct_rank(df_lb["BW_Ht_Ratio"])

    LEADERBOARD_METRICS = [
        # Force plate
        ("Concentric Impulse", "CI", 1, "", "Force Plate"),
        ("P1 Concentric Impulse", "P1 Conc. Impulse", 1, "", "Force Plate"),
        ("Concentric Impulse-100ms", "CI-100ms", 1, "", "Force Plate"),
        ("RSI-modified", "RSI-modified", 3, "", "Force Plate"),
        ("Peak Power / BM", "Peak Power / BM", 1, "", "Force Plate"),
        ("Jump Height (Flight Time) in Inches", "Jump Height", 2, " in", "Force Plate"),
        # Anthropometrics shown in the leaderboard
        ("Height_in", "Height", 1, " in", "Anthropometrics"),
        ("Mass_lbs", "Mass", 1, " lbs", "Anthropometrics"),
        ("Wingspan_in", "Wingspan", 1, " in", "Anthropometrics"),
        ("Wing_Adv_in", "Wingspan Adv.", 1, " in", "Anthropometrics"),
        ("BW_Ht_Pct", "BW/Ht Pct", 0, "th", "Anthropometrics"),
    ]

    FILTERABLE_METRICS = LEADERBOARD_METRICS

    # Make sure every expected column exists so the app will not silently drop
    # a leaderboard metric if a sheet is missing a column.
    for _col, _, _, _, _ in LEADERBOARD_METRICS:
        if _col not in df_lb.columns:
            df_lb[_col] = np.nan

    def _num_from_text(x):
        x = "" if x is None else str(x).strip()
        if x == "":
            return None
        try:
            return float(x.replace(",", ""))
        except ValueError:
            return "INVALID"

    # -------------------------------------------------------------------------
    # Leaderboard filters live in the Streamlit sidebar, where the old sliders were.
    # This keeps the leaderboard itself full-width and prevents the page from
    # getting clustered horizontally.
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Leaderboard")

        search = st.text_input("Search", placeholder="Name…", key="lb_search")

        yr_opts = ["All"] + sorted(df_lb["Year"].dropna().unique().astype(int).tolist(), reverse=True)
        yr_sel = st.selectbox("Year", yr_opts, key="lb_year")

        pos_grp_sel = st.selectbox(
            "Position Group",
            ["All", "Pitcher", "Catcher", "Infielder", "Outfielder"],
            key="lb_pos",
        )

        sort_options = [
            "Athleticism Score", "Pos. Athleticism",
            "CI", "P1 Conc. Impulse", "CI-100ms", "RSI-modified", "Peak Power / BM",
            "Jump Height",
            "Height", "Mass", "Wingspan", "Wingspan Adv.", "BW/Ht Pct",
        ]
        sort_by = st.selectbox("Sort by", sort_options, key="lb_sort")

        st.markdown("---")
        st.markdown("#### Numeric cutoffs")
        st.caption("Blank boxes ignored. Filters stack.")

        active_filter_notes = []
        filter_values = []

        # Use a base dataframe filtered only by the primary controls to display ranges.
        range_base = df_lb.copy()
        if search:
            range_base = range_base[range_base["athleteName"].str.contains(search, case=False, na=False)]
        if yr_sel != "All":
            range_base = range_base[range_base["Year"] == int(yr_sel)]
        if pos_grp_sel != "All":
            range_base = range_base[range_base["pos_group"] == pos_grp_sel]

        filter_groups = ["Force Plate", "Anthropometrics"]
        for group_name in filter_groups:
            with st.expander(group_name, expanded=False):
                group_metrics = [m for m in FILTERABLE_METRICS if m[4] == group_name]
                for i, (col, label, digits, suffix, _) in enumerate(group_metrics):
                    vals = pd.to_numeric(range_base[col], errors="coerce").dropna()
                    if vals.empty:
                        range_txt = "No data"
                    elif col == "BW_Ht_Pct":
                        range_txt = f"{vals.min():.0f}–{vals.max():.0f} pct"
                    else:
                        range_txt = f"{vals.min():.{digits}f}–{vals.max():.{digits}f}{suffix}"

                    st.caption(f"**{label}** · {range_txt}")
                    c_min, c_max = st.columns(2)
                    with c_min:
                        min_txt = st.text_input(
                            "Min",
                            value="",
                            placeholder="min",
                            key=f"lb_min_{group_name}_{i}_{col}",
                            label_visibility="collapsed",
                        )
                    with c_max:
                        max_txt = st.text_input(
                            "Max",
                            value="",
                            placeholder="max",
                            key=f"lb_max_{group_name}_{i}_{col}",
                            label_visibility="collapsed",
                        )

                    min_val = _num_from_text(min_txt)
                    max_val = _num_from_text(max_txt)

                    if min_val == "INVALID":
                        st.warning(f"{label} min must be numeric.")
                        min_val = None
                    if max_val == "INVALID":
                        st.warning(f"{label} max must be numeric.")
                        max_val = None

                    if min_val is not None:
                        active_filter_notes.append(f"{label} ≥ {min_val:g}")
                    if max_val is not None:
                        active_filter_notes.append(f"{label} ≤ {max_val:g}")
                    filter_values.append((col, label, min_val, max_val))

        if active_filter_notes:
            st.success("Active: " + "; ".join(active_filter_notes))
    dff = df_lb.copy()
    if search:
        dff = dff[dff["athleteName"].str.contains(search, case=False, na=False)]
    if yr_sel != "All":
        dff = dff[dff["Year"] == int(yr_sel)]
    if pos_grp_sel != "All":
        dff = dff[dff["pos_group"] == pos_grp_sel]

    for col, label, min_val, max_val in filter_values:
        numeric_values = pd.to_numeric(dff[col], errors="coerce")
        if min_val is not None:
            dff = dff[numeric_values >= min_val]
            numeric_values = pd.to_numeric(dff[col], errors="coerce")
        if max_val is not None:
            dff = dff[numeric_values <= max_val]

    # ---------------------------------------------------------------------
    # Sort
    # ---------------------------------------------------------------------
    sort_map = {
        "Athleticism Score": "athlete_quality_score",
        "Pos. Athleticism": "aq_pos_score",
        "CI": "Concentric Impulse",
        "P1 Conc. Impulse": "P1 Concentric Impulse",
        "CI-100ms": "Concentric Impulse-100ms",
        "RSI-modified": "RSI-modified",
        "Peak Power / BM": "Peak Power / BM",
        "Jump Height": "Jump Height (Flight Time) in Inches",
        "Height": "Height_in",
        "Mass": "Mass_lbs",
        "Wingspan": "Wingspan_in",
        "Wingspan Adv.": "Wing_Adv_in",
        "BW/Ht Pct": "BW_Ht_Pct",
    }
    sort_col = sort_map[sort_by]
    dff = dff.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

    if dff.empty:
        st.warning("No athletes match the selected filters.")
        default_ath = df.sort_values("athleteName").iloc[0]["athleteName"]
    else:
        # -----------------------------------------------------------------
        # Median cards
        # -----------------------------------------------------------------
        st.markdown(f'<p class="label">Force Plate Medians</p>', unsafe_allow_html=True)
        fp_metrics = [m for m in LEADERBOARD_METRICS if m[4] == "Force Plate"]
        fp_cols = st.columns(len(fp_metrics) + 1)
        fp_cols[0].metric("Athletes", str(len(dff)))
        for i, (col, label, digits, suffix, _) in enumerate(fp_metrics, start=1):
            fp_cols[i].metric(
                f"Median {label}",
                fmt(pd.to_numeric(dff[col], errors="coerce").median(), digits, suffix),
            )

        st.markdown(f'<p class="label">Anthropometric Medians</p>', unsafe_allow_html=True)
        anthro_metrics = [m for m in LEADERBOARD_METRICS if m[4] == "Anthropometrics"]
        an_cols = st.columns(len(anthro_metrics))
        for i, (col, label, digits, suffix, _) in enumerate(anthro_metrics):
            med_val = pd.to_numeric(dff[col], errors="coerce").median()
            if col == "BW_Ht_Pct":
                an_cols[i].metric("Median BW/Ht Pct", pct_sfx(med_val))
            else:
                an_cols[i].metric(f"Median {label}", fmt(med_val, digits, suffix))

        # -----------------------------------------------------------------
        # Leaderboard table: only text columns are Athlete and Position.
        # BW/Ht is shown as a percentile, not the raw pounds-per-inch value.
        # -----------------------------------------------------------------
        tbl_cols = [
            "athleteName", "Position", "Year",
            "athlete_quality_score", "aq_pos_score",
            "Concentric Impulse", "P1 Concentric Impulse", "Concentric Impulse-100ms",
            "RSI-modified", "Peak Power / BM", "Jump Height (Flight Time) in Inches",
            "Height_in", "Mass_lbs", "Wingspan_in", "Wing_Adv_in", "BW_Ht_Pct",
        ]
        tbl = dff[tbl_cols].copy()
        tbl = tbl.rename(columns={
            "athleteName": "Athlete",
            "athlete_quality_score": "Athleticism",
            "aq_pos_score": "Pos. Athleticism",
            "Concentric Impulse": "CI",
            "P1 Concentric Impulse": "P1 Conc. Impulse",
            "Concentric Impulse-100ms": "CI-100ms",
            "Jump Height (Flight Time) in Inches": "Jump Height",
            "Height_in": "Height",
            "Mass_lbs": "Mass",
            "Wingspan_in": "Wingspan",
            "Wing_Adv_in": "Wingspan Adv.",
            "BW_Ht_Pct": "BW/Ht Pct",
        })

        round_map = {
            "Athleticism": 1,
            "Pos. Athleticism": 1,
            "CI": 1,
            "P1 Conc. Impulse": 1,
            "CI-100ms": 1,
            "RSI-modified": 3,
            "Peak Power / BM": 1,
            "Jump Height": 2,
            "Height": 1,
            "Mass": 1,
            "Wingspan": 1,
            "Wingspan Adv.": 1,
            "BW/Ht Pct": 0,
        }
        for col, digits in round_map.items():
            if col in tbl.columns:
                tbl[col] = pd.to_numeric(tbl[col], errors="coerce").round(digits)

        st.caption(
            "Units: Height, Wingspan, and Wingspan Adv. are inches; "
            "Mass is pounds; BW/Ht Pct is the percentile rank of pounds per inch."
        )
        sel = st.dataframe(
            tbl,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="lb_tbl",
        )
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

    # ── Core values ───────────────────────────────────────────────────────────
    aq_val  = sf(row.get("athlete_quality_score"))
    pos_val = sf(row.get("aq_pos_score"))
    ci_val  = sf(row.get("Concentric Impulse"))
    mass_kg = sf(row.get("Mass"))
    ht_cm   = sf(row.get("Height"))

    # Wingspan values
    wing_cm     = sf(row.get("Wingspan"))
    wing_adv_cm = sf(row.get("wingspan_advantage"))
    wing_pct    = sf(row.get("wingspan_pct"))
    wing_pct_str= pct_sfx(int(round(wing_pct))) if pd.notna(wing_pct) else "—"
    wing_adv_in = fmt_wingspan_adv(wing_adv_cm)

    _wing_pool  = df["wingspan_advantage"].dropna()
    if pd.notna(wing_adv_cm) and len(_wing_pool) > 0:
        _p85 = float(_wing_pool.quantile(0.85))
        _p20 = float(_wing_pool.quantile(0.20))
        if wing_adv_cm >= _p85:
            wing_tier_label = "Notable Reach Advantage"
            wing_tier_color = GREEN
            wing_adv_css    = "wing-adv-pos"
        elif wing_adv_cm <= _p20:
            wing_tier_label = "Below-Average Reach"
            wing_tier_color = RED
            wing_adv_css    = "wing-adv-neg"
        else:
            wing_tier_label = "Average Reach"
            wing_tier_color = GOLD
            wing_adv_css    = "wing-adv-neu"
    else:
        wing_tier_label = "No Data"
        wing_tier_color = "#9AAAC0"
        wing_adv_css    = ""

    # CI tier
    ci_tier_val  = ci_tier_label(ci_val)
    tier_idx     = ci_tier_index(ci_val)
    tier_clrs    = ["#BA0C2F","#E2C188","#11225A"]
    tier_color   = tier_clrs[tier_idx] if tier_idx >= 0 else "#9AAAC0"
    prog_cat     = str(row.get("programming_category","—"))
    prog_color   = PROG_COLORS.get(prog_cat, "#9AAAC0")
    prog_desc    = PROG_DESC.get(prog_cat, "")

    lbs_next   = sf(row.get("lbs_to_next_tier"))
    next_label = str(row.get("next_tier_label","—"))
    wc_next    = str(row.get("weight_class_next","—"))
    wc_col     = WEIGHT_CLASS_COLORS.get(wc_next,"#9AAAC0")
    lbs_to_315 = sf(row.get("lbs_to_315"))
    wc_315     = str(row.get("weight_class_315","—"))
    wc_315_col = WEIGHT_CLASS_COLORS.get(wc_315,"#9AAAC0")
    lbs_next_str = f"+{lbs_next:.1f} lbs" if (pd.notna(lbs_next) and lbs_next>0) else ("Top tier" if lbs_next==0 else "—")
    lbs_315_str  = f"+{lbs_to_315:.1f} lbs" if (pd.notna(lbs_to_315) and lbs_to_315>0) else ("✓ Already ≥ 315" if lbs_to_315==0 else "—")

    def alltime_pct(val, col):
        try:
            pool = df[col].dropna().apply(float)
            v    = float(val)
            return pct_sfx(int(round((pool<v).mean()*100)))
        except: return "—"

    aq_pct_str = alltime_pct(aq_val, "athlete_quality_score")

    def proj_bwht(lbs_gain):
        if pd.isna(mass_kg) or pd.isna(ht_cm) or pd.isna(lbs_gain) or lbs_gain<=0: return "—"
        new_kg = mass_kg + lbs_gain/2.20462
        ratio  = (new_kg*2.20462)/(ht_cm/2.54)
        pool_r = df["bmi_raw"].dropna()
        pct    = int(round(float((pool_r<ratio).mean()*100)))
        return f"{ratio:.2f} ({pct_sfx(pct)})"

    pos_str = str(row.get("Position",""))
    pos_str = pos_str if pos_str not in ("nan","None","") else "—"
    sch_str = str(row.get("School Type",""))
    sch_str = sch_str if sch_str not in ("nan","None","") else "—"
    rnk_val = int((df["athlete_quality_score"].dropna() > aq_val).sum()+1) if pd.notna(aq_val) else None
    pool_n  = int(df["athlete_quality_score"].notna().sum())

    # ── Header ────────────────────────────────────────────────────────────────
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
                <span style="font-size:12px;color:{SLATE}">{sel_yr_display} · {pos_str} · {sch_str}</span><br>
                <span style="display:inline-block;margin-top:8px;background:{prog_color};
                    color:white;font-size:11px;font-weight:700;padding:3px 14px;
                    border-radius:20px;letter-spacing:0.06em">⚙ {prog_cat}</span>
                <span style="font-size:11px;color:{SLATE};margin-left:8px">{prog_desc}</span>
            </div>
            <div style="text-align:right">
                <p style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:{SLATE};margin:0">OVERALL RANK</p>
                <p style="font-family:'Playfair Display',serif;font-size:36px;font-weight:900;
                    color:{RED};margin:0">{"#"+str(rnk_val) if rnk_val else "—"}</p>
                <p style="font-size:11px;color:#9AAAC0;margin:0">of {pool_n} athletes (all years)</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero row: Athleticism | Pos. gauge | CI Tier ──────────────────────────
    h1, h2, h3 = st.columns([1,1,1.3])

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
            <div style="font-size:11px;color:{SLATE};margin-top:2px">score out of 100 · all-time</div>
            <div style="margin-top:12px;background:#F0F3F8;border-radius:6px;height:8px">
                <div style="width:{bar_w:.0f}%;background:{bar_cl};border-radius:6px;height:8px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with h2:
        _pg  = str(row.get("pos_group",""))
        _pos = str(row.get("Position",""))
        if _pg and _pg not in ("Unknown","nan",""):
            pos_lbl = f"Athleticism vs {_pg}s"
        elif _pos and _pos not in ("nan","None",""):
            pos_lbl = f"Athleticism vs {_pos}s"
        else:
            pos_lbl = "Position Data Unavailable"
        st.plotly_chart(
            make_gauge(pos_val if (pd.notna(pos_val) and pos_val>0) else None, pos_lbl, SLATE),
            use_container_width=True, key="g_pos")

    with h3:
        parts = [
            f'<div style="background:white;border:1px solid {BORD};border-top:4px solid {tier_color};'
            f'border-radius:10px;padding:18px 20px;box-shadow:0 2px 8px rgba(17,34,90,0.06)">',
            f'<p style="font-size:10px;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:{SLATE};margin:0 0 4px 0">CI TIER</p>',
            f'<div style="font-family:Playfair Display,serif;font-size:32px;font-weight:900;'
            f'color:{tier_color}">{ci_tier_val}</div>',
            f'<div style="font-size:12px;color:{SLATE};margin-top:2px">Current CI: '
            f'<strong style="color:{NAV}">{fmt(ci_val,1)}</strong></div>',
            f'<hr style="border-color:{BORD};margin:10px 0">',
            f'<div style="display:flex;gap:16px;flex-wrap:wrap">',
            f'<div><p style="font-size:9px;font-weight:700;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:{SLATE};margin:0 0 2px 0">To next tier ({next_label})</p>',
            f'<div style="font-size:18px;font-weight:700;color:{wc_col}">{lbs_next_str}</div>',
            f'<div style="font-size:11px;color:{SLATE}">BW/Ht at target: {proj_bwht(lbs_next)}</div>',
            f'<span style="display:inline-block;background:{wc_col};color:white;font-size:10px;'
            f'font-weight:700;padding:2px 8px;border-radius:10px;margin-top:4px">{wc_next}</span></div>',
            f'<div><p style="font-size:9px;font-weight:700;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:{SLATE};margin:0 0 2px 0">To 315 CI</p>',
            f'<div style="font-size:18px;font-weight:700;color:{wc_315_col}">{lbs_315_str}</div>',
            f'<div style="font-size:11px;color:{SLATE}">BW/Ht at target: {proj_bwht(lbs_to_315)}</div>',
            f'<span style="display:inline-block;background:{wc_315_col};color:white;font-size:10px;'
            f'font-weight:700;padding:2px 8px;border-radius:10px;margin-top:4px">{wc_315}</span></div>',
            '</div></div>',
        ]
        st.markdown("".join(parts), unsafe_allow_html=True)

    # ── Score legend ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 16px 0">
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="width:10px;height:10px;border-radius:50%;background:{GREEN};display:inline-block"></span>
            <strong>75–100</strong>&nbsp;Elite · clear draft target
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="width:10px;height:10px;border-radius:50%;background:{GOLD};display:inline-block"></span>
            <strong>50–74</strong>&nbsp;Above average · worth consideration
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="width:10px;height:10px;border-radius:50%;background:#D0D7E6;display:inline-block"></span>
            <strong>25–49</strong>&nbsp;Below average
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
            border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
            <span style="width:10px;height:10px;border-radius:50%;background:#9AAAC0;display:inline-block"></span>
            <strong>0–24</strong>&nbsp;Well below average
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pitcher flag ─────────────────────────────────────────────────────────
    is_pitcher = str(row.get("pos_group","")).strip() == "Pitcher"

    # ── Percentile cards ──────────────────────────────────────────────────────
    grp_key   = str(row.get("pos_group","")).lower()
    grp_label = str(row.get("pos_group",""))
    grp_label = grp_label if grp_label not in ("Unknown","nan","") else "Pos."

    # Show wingspan card only for pitchers
    base_pct_items = [
        ("CI",          "ci_pct_alltime",    "ci_pct_yr",    False),
        ("30yd Sprint", "sprint_pct_alltime", "sprint_pct_yr",True),
        ("RSI-mod",     "rsi_pct_alltime",   "rsi_pct_yr",   False),
        ("Pk Pwr/BM",   "pp_pct_alltime",    "pp_pct_yr",    False),
        ("Height",      "height_pct",        None,           False),
    ]
    if is_pitcher:
        pct_items = base_pct_items + [("Wingspan", "wingspan_pct", None, False)]
        pc = st.columns(6)
    else:
        pct_items = base_pct_items
        pc = st.columns(5)

    pos_pct_map = {
        "CI":          f"ci_pct_{grp_key}" if grp_key else None,
        "30yd Sprint": f"sprint_pct_{grp_key}" if grp_key else None,
        "RSI-mod":     f"rsi_pct_{grp_key}" if grp_key else None,
        "Pk Pwr/BM":   f"pp_pct_{grp_key}" if grp_key else None,
        "Height":      None,
        "Wingspan":    None,
    }
    for i, (col_lbl, pa, py, inv) in enumerate(pct_items):
        p_all = sf(row.get(pa))
        p_yr  = sf(row.get(py)) if py else np.nan
        p_pos = sf(row.get(pos_pct_map.get(col_lbl))) if pos_pct_map.get(col_lbl) else np.nan
        is_wing_card = col_lbl == "Wingspan"
        top_col = GOLD if is_wing_card else RED
        with pc[i]:
            st.markdown(
                f'<div style="background:white;border:1px solid {BORD};border-top:4px solid {top_col};'
                f'border-radius:10px;padding:14px 10px;text-align:center;margin-bottom:16px;'
                f'box-shadow:0 2px 8px rgba(17,34,90,0.06)">'
                f'<div style="font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;'
                f'color:{SLATE};margin-bottom:6px">{col_lbl}</div>'
                f'<div style="font-family:\'Playfair Display\',serif;font-size:28px;font-weight:900;'
                f'color:{top_col};line-height:1">{fmt(p_all,0) if pd.notna(p_all) else "—"}</div>'
                f'<div style="font-size:10px;color:#9AAAC0;margin-top:2px">All-time pct</div>'
                f'<div style="font-size:13px;font-weight:600;color:{NAV};margin-top:4px">'
                f'{fmt(p_yr,0) if pd.notna(p_yr) else "—"}</div>'
                f'<div style="font-size:10px;color:#9AAAC0">{sel_yr_display} pct</div>'
                f'<div style="font-size:13px;font-weight:600;color:{SLATE};margin-top:4px">'
                f'{fmt(p_pos,0) if pd.notna(p_pos) else "—"}</div>'
                f'<div style="font-size:10px;color:#9AAAC0">{grp_label} pct</div>'
                f'</div>',
                unsafe_allow_html=True)

    # ── Main body: metrics | charts | right panel ─────────────────────────────
    m1, m2, m3 = st.columns([1, 1.3, 1.3])

    with m1:
        # Wingspan row in anthropometrics — always show raw numbers, but
        # only add the advantage flag line for pitchers
        wing_adv_row = ""
        if is_pitcher:
            adv_color = (GREEN if wing_tier_color == GREEN
                         else RED if wing_tier_color == RED else GOLD)
            wing_adv_row = (
                f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
                f'<span style="color:{SLATE};min-width:160px;font-size:12px">Wingspan Adv.</span>'
                f'<span style="font-weight:600;color:{adv_color}">{wing_adv_in}</span>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:white;border:1px solid {BORD};border-top:4px solid {NAV};'
            f'border-radius:10px;padding:18px 22px;box-shadow:0 2px 8px rgba(17,34,90,0.06);margin-bottom:16px">'
            f'<div style="font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;'
            f'color:{SLATE};margin-bottom:6px">Force Plate</div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">Concentric Impulse</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("Concentric Impulse")),1)}</span></div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">P1 Conc. Impulse</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("P1 Concentric Impulse")),1)}</span></div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">CI-100ms</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("Concentric Impulse-100ms")),1)}</span></div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">RSI-modified</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("RSI-modified")),3)}</span></div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">Peak Power / BM</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("Peak Power / BM")),1)}</span></div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">Jump Height</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("Jump Height (Flight Time) in Inches")),2," in")}</span></div>'
            f'<div style="font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;'
            f'color:{SLATE};margin:12px 0 6px 0">Sprint</div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">30yd</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("30yd Split")),3,"s")}</span></div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">10yd</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("10yd Split")),3,"s")}</span></div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">20yd</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("20yd Split")),3,"s")}</span></div>'
            f'<div style="font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;'
            f'color:{SLATE};margin:12px 0 6px 0">Anthropometrics</div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">Height</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt_height(sf(row.get("Height")))}</span></div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">Mass</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt_mass(sf(row.get("Mass")))}</span></div>'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">Wingspan</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt_wingspan(wing_cm)}</span></div>'
            f'{wing_adv_row}'
            f'<div style="display:flex;align-items:baseline;margin-bottom:7px;font-size:13px">'
            f'<span style="color:{SLATE};min-width:160px;font-size:12px">BW/Ht Ratio</span>'
            f'<span style="font-weight:600;color:{NAV}">{fmt(sf(row.get("bmi_raw")),2)}</span></div>'
            f'</div>',
            unsafe_allow_html=True)

    with m2:
        st.plotly_chart(make_radar(row, is_pitcher=is_pitcher),
                        use_container_width=True, key="g_radar")
        st.plotly_chart(make_profile(row, strat_feats),
                        use_container_width=True, key="g_profile")

    with m3:
        if is_pitcher:
            # ── Wingspan Feature Panel (pitchers only) ─────────────────────────
            wing_pct_bar = min(100, max(0, float(wing_pct or 0)))
            wing_icon = "✓" if wing_tier_color == GREEN else ("⚠" if wing_tier_color == RED else "~")

            st.markdown(
                f'<div style="background:linear-gradient(135deg,{NAV} 0%,#1a3275 100%);'
                f'border-radius:12px;padding:22px 24px;color:white;margin-bottom:16px;'
                f'box-shadow:0 6px 24px rgba(17,34,90,0.20)">'
                f'<p style="font-size:10px;font-weight:700;letter-spacing:0.18em;'
                f'text-transform:uppercase;color:rgba(255,255,255,0.55);margin:0 0 14px 0">'
                f'✦ WINGSPAN ANALYSIS</p>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px">'
                f'<div style="background:rgba(255,255,255,0.10);border-radius:8px;padding:12px 16px;text-align:center">'
                f'<div style="font-family:\'Playfair Display\',serif;font-size:26px;font-weight:900;'
                f'color:white;line-height:1.1">{fmt_wingspan(wing_cm)}</div>'
                f'<div style="font-size:9px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
                f'color:rgba(255,255,255,0.55);margin-top:3px">Wingspan</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.10);border-radius:8px;padding:12px 16px;text-align:center">'
                f'<div style="font-family:\'Playfair Display\',serif;font-size:26px;font-weight:900;'
                f'color:white;line-height:1.1">{fmt_height(sf(row.get("Height")))}</div>'
                f'<div style="font-size:9px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
                f'color:rgba(255,255,255,0.55);margin-top:3px">Height</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.10);border-radius:8px;padding:12px 16px;text-align:center">'
                f'<div style="font-family:\'Playfair Display\',serif;font-size:26px;font-weight:900;'
                f'color:{wing_tier_color};line-height:1.1">{wing_adv_in}</div>'
                f'<div style="font-size:9px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
                f'color:rgba(255,255,255,0.55);margin-top:3px">Advantage</div>'
                f'</div>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
                f'<span style="font-size:11px;font-weight:600;color:rgba(255,255,255,0.75)">Wingspan Percentile</span>'
                f'<span style="font-family:\'Playfair Display\',serif;font-size:20px;font-weight:900;'
                f'color:white">{wing_pct_str}</span>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.15);border-radius:6px;height:10px;margin-bottom:14px">'
                f'<div style="width:{wing_pct_bar:.0f}%;background:{wing_tier_color};'
                f'border-radius:6px;height:10px"></div>'
                f'</div>'
                f'<span style="display:inline-block;background:{wing_tier_color};color:white;'
                f'font-size:11px;font-weight:700;padding:4px 14px;border-radius:20px;'
                f'letter-spacing:0.06em">{wing_icon} {wing_tier_label}</span>'
                f'</div>',
                unsafe_allow_html=True)

            fig_wing = make_wingspan_bar(row, df)
            if fig_wing:
                st.plotly_chart(fig_wing, use_container_width=True, key="g_wing_dist")

        # Year-over-year trends
        multi = ath_all[ath_all["Year"].notna()]
        if len(multi) >= 2:
            st.markdown(
                f'<p style="font-size:10px;font-weight:600;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:{SLATE};margin-top:8px;margin-bottom:6px">'
                f'Year-over-year trends</p>',
                unsafe_allow_html=True)
            trend_cols = [
                ("Concentric Impulse",    "CI",                False),
                ("RSI-modified",          "RSI-mod",           False),
                ("30yd Split",            "30yd Sprint",       True),
                ("athlete_quality_score", "Athleticism Score", False),
            ]
            if is_pitcher:
                trend_cols.append(("wingspan_advantage", "Wingspan Advantage", False))
            for tcol, tlbl, tinv in trend_cols:
                fig_t = make_trend(multi, tcol, tlbl, tinv)
                if fig_t:
                    st.plotly_chart(fig_t, use_container_width=True,
                                    key=f"trend_{sel_ath}_{tcol}")

    # ── Development Projection ────────────────────────────────────────────────
    if pd.notna(ci_val) and pd.notna(mass_kg) and ci_val > 0 and mass_kg > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="border-left:4px solid {GREEN};padding-left:12px;margin:0 0 14px 0">'
            f'<span style="font-size:10px;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:{GREEN}">Development Projection</span></div>',
            unsafe_allow_html=True)

        ci_per_kg     = ci_val / mass_kg
        ci_per_kg_new = ci_per_kg * 0.97

        def sc_bwht(kg, cm):
            if pd.isna(kg) or pd.isna(cm) or cm==0: return np.nan
            return (kg*2.20462)/(cm/2.54)

        def sc_bwht_pct(val):
            pool_r = df["bmi_raw"].dropna()
            if pd.isna(val) or len(pool_r)==0: return np.nan
            return float((pool_r < val).mean()*100)

        def sc_ci_pct(val):
            pool_c = df["Concentric Impulse"].dropna()
            if pd.isna(val) or len(pool_c)==0: return np.nan
            return float((pool_c < val).mean()*100)

        rows_proj = []
        for label, gain_lbs in [("Current",0),("+10 lbs",10),("+15 lbs",15)]:
            new_kg   = mass_kg + gain_lbs/2.20462
            ci_p     = ci_per_kg_new * new_kg if gain_lbs > 0 else ci_val
            bwht_v   = sc_bwht(new_kg, ht_cm)
            bwht_p   = sc_bwht_pct(bwht_v)
            ci_p_pct = sc_ci_pct(ci_p)
            delta    = ci_p - ci_val
            rows_proj.append({
                "Scenario":  label,
                "CI (N·s)":  f"{ci_p:.1f}" + (f"  ({'+' if delta>=0 else ''}{delta:.1f})" if gain_lbs>0 else ""),
                "CI Pct":    pct_sfx(int(round(ci_p_pct))) if pd.notna(ci_p_pct) else "—",
                "Body Mass": f"{new_kg*2.20462:.1f} lbs",
                "BW/Ht":     f"{bwht_v:.2f}" if pd.notna(bwht_v) else "—",
                "BW/Ht Pct": pct_sfx(int(round(bwht_p))) if pd.notna(bwht_p) else "—",
            })

        st.dataframe(pd.DataFrame(rows_proj), use_container_width=True,
                     hide_index=True, key="sc_proj_tbl")
        st.markdown(
            f'<p style="font-size:11px;color:#9AAAC0;margin-top:4px">'
            f'Assumes 3% decrease in CI/kg with added mass. All-time percentiles. '
            f'Internal data has shown a trend towards pitchers having smaller penalties (0–3%) '
            f'than position players (3–5%).</p>',
            unsafe_allow_html=True)
