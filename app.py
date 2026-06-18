# VERSION: ci100_missing_display_v64 -- CI-100ms missing values remain missing in scorecard charts/PDF
# VERSION: scorecard_future_capacity_card_v62 -- PDF adds Future Capacity range card below Potential to Gain
# VERSION: option1_methodology_tab_v51 -- added Methods / Definitions tab with exact stakeholder language
# VERSION: force_plate_2026_scorecards_v56 -- handle X/missing wingspan in Force Plate 2026
# VERSION: option1_capacity_raw_physical_attributes_v50 -- Capacity raw weighted percentile; Anthropometrics renamed Physical Attributes
# VERSION: option1_original_card_grid_v49 -- bottom card text 6.6
# VERSION: option1_bottom_cards_wide_v44 -- skins-based, wider bottom summary cards so Program Focus fits
# VERSION: option1_program_focus_fit_v43 -- uses updated scorecard_skins with compact Athlete Group and Program Focus cards
# VERSION: option1_skins_force_tiny_v42 -- uses fixed scorecard_skins.py card text sizing
# VERSION: option1_skins_fixed_v41 -- uses scorecard_skins.py; fixed Athlete Group/Program Focus sizing in skin
# VERSION: option1_wording_smaller_cards_v39 -- reload scorecard_skins and force smaller PDF card values
# VERSION: option1_wording_v37 -- smaller Athlete Group and Program Focus card text in PDF renderer
# VERSION: option1_wording_v36 -- smaller PDF Athlete Group and Program Focus card values
# VERSION: option1_pdf_old_dashboard_fonts_v33 -- dashboard fonts reverted; Option 1 PDF retained
# VERSION: stakeholder_feedback_v27 -- potential to gain, athlete group/program focus, cleaner capacity language
# VERSION: bodyweight_labels_v26 -- display label Mass changed to Bodyweight
# VERSION: happy_medium_v21 -- development projection moved into right scorecard column
# VERSION: compact_medians_v9 -- leaderboard medians compact; removed BW/Ht percentile median
# VERSION: athlete_scorecard_profile_bars_less_cramped_v8 -- profile bars moved full-width and spacing fixed
# VERSION: sidebar_compact_filters_v6 -- leaderboard filters moved to sidebar; compact min/max inputs; seated height removed
# VERSION: athlete_scorecard_wingspan_tiers_v24 -- updated wingspan reach tier labels by percentile
import warnings
warnings.filterwarnings("ignore")

import os
import json
import traceback
from io import BytesIO
from pathlib import Path
import re
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
INFIELDERS = {"SS", "3B", "2B", "1B", "INF", "IF"}
OUTFIELDERS= {"CF", "LF", "RF", "OF"}

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


def athlete_group_label(category):
    """User-facing athlete group based on CI/P1 CI buckets."""
    return {
        "High-High": "High CI - High P1",
        "High-Low": "High CI - Low P1",
        "Low": "Low CI / Foundational",
        "Unclassified": "Unclassified",
    }.get(str(category), "Unclassified")


def program_focus_label(category):
    """User-facing training/development focus tied to athlete group."""
    return {
        "High-High": "Advanced",
        "High-Low": "P1 Development",
        "Low": "Foundational Strength/Capacity",
        "Unclassified": "Unclassified",
    }.get(str(category), "Unclassified")

PROG_COLORS = {
    "High-High":    "#4CAF82",
    "High-Low":     "#E2C188",
    "Low":          "#BA0C2F",
    "Unclassified": "#9AAAC0",
}
PROG_DESC = {
    "High-High": "High CI and high P1 — advanced focus",
    "High-Low":  "High CI with lower P1 — prioritize P1 development",
    "Low":       "Lower CI — foundational strength/capacity focus",
    "Unclassified": "Insufficient CI/P1 data",
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




def worksheet_to_dataframe(sh, worksheet_name):
    """Read a Google Sheet tab with raw values so late-added columns are not missed.

    gspread.get_all_records() can be brittle when a sheet has duplicate/blank
    headers or manually-added columns at the far right. This reader keeps every
    non-empty header, uniquifies duplicate names, and pads short rows.
    """
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    raw_headers = [str(h).strip() for h in values[0]]
    max_len = max(len(r) for r in values)
    if len(raw_headers) < max_len:
        raw_headers += [""] * (max_len - len(raw_headers))

    seen = {}
    headers = []
    for i, h in enumerate(raw_headers):
        base = h if h else f"__blank_{i}"
        key = base
        if key in seen:
            seen[key] += 1
            key = f"{base}__{seen[base]}"
        else:
            seen[key] = 1
        headers.append(key)

    rows = []
    for r in values[1:]:
        rr = list(r) + [""] * (max_len - len(r))
        rows.append(rr[:max_len])

    df = pd.DataFrame(rows, columns=headers)
    # Drop columns that truly have no header and no values, but keep any blank-header
    # column with data because it may be a manually-added field.
    drop_cols = []
    for c in df.columns:
        if str(c).startswith("__blank_") and df[c].astype(str).str.strip().eq("").all():
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def clean_forcedecks_force_plate(raw_df):
    """Convert a raw ForceDecks CMJ export into the app's Force Plate tab format.

    Expected source tab: Force Plate 2026
    The raw ForceDecks export uses column names with units and sometimes trailing spaces.
    The dashboard scoring pipeline expects the simplified historical Google Sheet column names.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    fd = raw_df.copy()
    fd.columns = [str(c).strip() for c in fd.columns]

    def first_existing(*names):
        """Find a column by exact or case-insensitive stripped header name."""
        col_map = {str(c).strip().lower(): c for c in fd.columns}
        for name in names:
            if name in fd.columns:
                return name
            key = str(name).strip().lower()
            if key in col_map:
                return col_map[key]
        return None

    def num_col(*names):
        col = first_existing(*names)
        if col is None:
            return pd.Series(np.nan, index=fd.index)
        return pd.to_numeric(fd[col], errors="coerce")

    def text_col(*names):
        col = first_existing(*names)
        if col is None:
            return pd.Series("", index=fd.index, dtype="object")
        return fd[col].astype(str).replace({"nan": "", "None": ""}).str.strip()

    def length_to_cm_value(v):
        """Accept cm, inches, feet-inches text, blanks, or X/NA markers and return cm."""
        if pd.isna(v):
            return np.nan
        txt = str(v).strip()
        if not txt:
            return np.nan
        if txt.upper() in {"X", "NA", "N/A", "NONE", "NULL", "-", "--"}:
            return np.nan
        # Handles values like 6'4, 6'4", 6-4, or 6 4.
        m = re.match(r"^\s*(\d+)\s*(?:'|-|ft| )\s*(\d+(?:\.\d+)?)", txt, flags=re.I)
        if m:
            ft = float(m.group(1)); inch = float(m.group(2))
            return (ft * 12 + inch) * 2.54
        try:
            x = float(txt.replace(",", ""))
        except Exception:
            return np.nan
        # Most manual entries will be inches (55-90). ISAK style could be cm (155-230).
        if 55 <= x <= 90:
            return x * 2.54
        return x

    def height_to_cm(*names):
        """Accept height as cm, inches, feet-inches text, or X/NA and return cm."""
        col = first_existing(*names)
        if col is None:
            return pd.Series(np.nan, index=fd.index)
        return fd[col].apply(length_to_cm_value)

    def wingspan_to_cm(*names):
        """Accept wingspan as cm, inches, feet-inches text, or X/NA and return cm."""
        col = first_existing(*names)
        if col is None:
            return pd.Series(np.nan, index=fd.index)
        return fd[col].apply(length_to_cm_value)

    def position_col():
        """Read position from a named header or, if needed, from a trailing manual column."""
        named = first_existing("Position", "position", "POSITION", "Pos", "Primary Position", "PrimaryPosition")
        if named is not None:
            return fd[named].astype(str).replace({"nan": "", "None": ""}).str.strip()

        # Fallback: user is manually adding Position as the last column. If a blank
        # or oddly-named trailing column contains position-like values, use it.
        valid_aliases = {
            "SP", "RHP", "LHP", "RP", "TWP", "P", "PITCHER",
            "C", "CATCHER",
            "SS", "3B", "2B", "1B", "INF", "IF", "INFIELD", "INFIELDER",
            "CF", "LF", "RF", "OF", "OUTFIELD", "OUTFIELDER",
        }
        for c in list(fd.columns)[::-1]:
            s = fd[c].astype(str).replace({"nan": "", "None": ""}).str.strip()
            nonempty = s[s.ne("")]
            if nonempty.empty:
                continue
            norm = nonempty.str.upper()
            if norm.isin(valid_aliases).mean() >= 0.75:
                return s
        return pd.Series("", index=fd.index, dtype="object")

    name = text_col("athleteName", "Name")
    external_id = text_col("playerID", "ExternalId", "External ID")

    # ExternalId is ideal. If it is blank, fall back to the athlete name so rows
    # still load into the dashboard instead of being dropped. Add ExternalId to
    # ForceDecks long-term if you want perfect merging with ISAK/Sprint/Positions.
    player_id = external_id.where(external_id.ne(""), name)

    date_col = first_existing("Date", "Test Date")
    if date_col:
        parsed_date = pd.to_datetime(fd[date_col], errors="coerce")
        year = parsed_date.dt.year
    else:
        year = pd.Series(np.nan, index=fd.index)

    if "Year" in fd.columns:
        year = pd.to_numeric(fd["Year"], errors="coerce").fillna(year)
    year = year.fillna(2026)

    out = pd.DataFrame({
        "playerID": player_id,
        "athleteName": name,
        "Year": year,
        "Mass": num_col("Mass", "BW [KG]", "Bodyweight [kg]", "Bodyweight [KG]"),
        "Peak Power / BM": num_col("Peak Power / BM [W/kg]", "Peak Power / BM", "Relative Peak Power [W/kg]"),
        "Concentric Impulse": num_col("Concentric Impulse [N s]", "Concentric Impulse"),
        "Force at Zero Velocity": num_col("Force at Zero Velocity [N]", "Force at Zero Velocity"),
        "Peak Power": num_col("Peak Power [W]", "Peak Power"),
        "P1 Concentric Impulse": num_col("P1 Concentric Impulse [N s]", "P1 Concentric Impulse"),
        "P2 Concentric Impulse": num_col("P2 Concentric Impulse [N s]", "P2 Concentric Impulse"),
        "RSI-modified": num_col("RSI-modified (Imp-Mom) [m/s]", "RSI-modified"),
        "Jump Height (Flight Time) in Inches": num_col("Jump Height (Imp-Mom) in Inches [in]", "Jump Height (Flight Time) in Inches"),
        "Concentric Impulse-100ms": num_col("Concentric Impulse-100ms [N s]", "Concentric Impulse-100ms"),
        "Vertical Velocity at Takeoff": num_col("Vertical Velocity at Takeoff [m/s]", "Vertical Velocity at Takeoff"),
        "Eccentric Braking Impulse": num_col("Eccentric Braking Impulse [N s]", "Eccentric Braking Impulse"),
        "Test Type": text_col("Test Type"),
        "Tags": text_col("Tags"),
        "Height": height_to_cm("Height", "Height [cm]", "Height [in]", "Height (in)", "Height Inches"),
        "Wingspan": wingspan_to_cm("Wingspan", "Wingspan [cm]", "Wingspan [in]", "Wingspan (in)", "Wingspan Inches"),
        "Position": position_col(),
    })

    # ForceDecks exports durations in milliseconds. The dashboard's bounds/scoring
    # expect seconds, so convert ms -> s.
    out["Eccentric Duration"] = num_col("Eccentric Duration [ms]", "Eccentric Duration") / 1000.0
    out["Concentric Duration"] = num_col("Concentric Duration [ms]", "Concentric Duration") / 1000.0
    out["Braking Phase Duration"] = num_col("Braking Phase Duration [ms]", "Braking Phase Duration") / 1000.0

    # Current ForceDecks exports already use negative CM depth. If a future export
    # comes in as positive depth, flip it to match the dashboard's expected convention.
    depth = num_col("Countermovement Depth [cm]", "Countermovement Depth")
    out["Countermovement Depth"] = np.where(depth > 0, -depth, depth)

    # School Type is used in Potential to Gain. The 2026 tab can include it as
    # a manual column at the end of the raw ForceDecks export.
    out["School Type"] = text_col("School Type")
    out["Data Source"] = "Force Plate 2026"

    out = out[out["athleteName"].astype(str).str.strip().ne("")].copy()
    out["playerID"] = out["playerID"].astype(str).str.strip()
    out["athleteName"] = out["athleteName"].astype(str).str.strip()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")

    return out


def clean_sprint_2026(raw_df):
    """Convert the optional Sprint 2026 tab into the app's Sprint tab format.

    Expected source tab: Sprint 2026
    Required columns: Name, 10, 20, 30
    Optional columns: playerID/ExternalId, Year, Date
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    sp = raw_df.copy()
    sp.columns = [str(c).strip() for c in sp.columns]

    def first_existing(*names):
        col_map = {str(c).strip().lower(): c for c in sp.columns}
        for name in names:
            if name in sp.columns:
                return name
            key = str(name).strip().lower()
            if key in col_map:
                return col_map[key]
        return None

    def text_col(*names):
        col = first_existing(*names)
        if col is None:
            return pd.Series("", index=sp.index, dtype="object")
        return sp[col].astype(str).replace({"nan": "", "None": ""}).str.strip()

    def num_col(*names):
        col = first_existing(*names)
        if col is None:
            return pd.Series(np.nan, index=sp.index)
        return pd.to_numeric(sp[col], errors="coerce")

    name = text_col("athleteName", "Name", "Athlete", "Athlete Name")
    external_id = text_col("playerID", "ExternalId", "External ID")
    player_id = external_id.where(external_id.ne(""), name)

    date_col = first_existing("Date", "Test Date")
    if date_col:
        year = pd.to_datetime(sp[date_col], errors="coerce").dt.year
    else:
        year = pd.Series(np.nan, index=sp.index)
    if "Year" in sp.columns:
        year = pd.to_numeric(sp["Year"], errors="coerce").fillna(year)
    year = year.fillna(2026)

    out = pd.DataFrame({
        "playerID": player_id,
        "athleteName": name,
        "Year": year,
        "10yd Split": num_col("10yd Split", "10", "10yd", "10 yd", "10 Yard", "10 Yard Split"),
        "20yd Split": num_col("20yd Split", "20", "20yd", "20 yd", "20 Yard", "20 Yard Split"),
        "30yd Split": num_col("30yd Split", "30", "30yd", "30 yd", "30 Yard", "30 Yard Split"),
        "Data Source": "Sprint 2026",
    })

    out = out[out["athleteName"].astype(str).str.strip().ne("")].copy()
    out["playerID"] = out["playerID"].astype(str).str.strip()
    out["athleteName"] = out["athleteName"].astype(str).str.strip()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    for c in ["10yd Split", "20yd Split", "30yd Split"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# ─── Google Sheets loader ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading data…")
def load_data(_v=8):
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

    fp_raw     = worksheet_to_dataframe(sh, "Force Plate")
    if "Data Source" not in fp_raw.columns:
        fp_raw["Data Source"] = "Force Plate"

    # Optional raw ForceDecks 2026 tab. This tab can be pasted directly from the
    # ForceDecks CSV export; the app converts names/units and appends it to the
    # historical Force Plate dataset before scoring.
    try:
        fp_2026_raw = worksheet_to_dataframe(sh, "Force Plate 2026")
        fp_2026_clean = clean_forcedecks_force_plate(fp_2026_raw)
        if not fp_2026_clean.empty:
            fp_raw = pd.concat([fp_raw, fp_2026_clean], ignore_index=True, sort=False)
            fp_raw = fp_raw.drop_duplicates(subset=["playerID", "athleteName", "Year"], keep="last")
    except Exception:
        # Keep the app working if the optional tab has not been created yet.
        pass

    isak_raw   = worksheet_to_dataframe(sh, "ISAK")
    sprint_raw = worksheet_to_dataframe(sh, "Sprint")

    # Optional simple 2026 sprint tab. Expected columns: Name, 10, 20, 30.
    # These are converted to the dashboard's standard Sprint fields and appended
    # before the sprint percentiles / Capacity Score are calculated.
    try:
        sprint_2026_raw = worksheet_to_dataframe(sh, "Sprint 2026")
        sprint_2026_clean = clean_sprint_2026(sprint_2026_raw)
        if not sprint_2026_clean.empty:
            sprint_raw = pd.concat([sprint_raw, sprint_2026_clean], ignore_index=True, sort=False)
            sprint_raw = sprint_raw.drop_duplicates(subset=["playerID", "athleteName", "Year"], keep="last")
    except Exception:
        # Keep the app working if the optional tab has not been created yet.
        pass

    pos_raw    = worksheet_to_dataframe(sh, "Positions")

    for df in [fp_raw, isak_raw, sprint_raw]:
        if "playerID" not in df.columns:
            df["playerID"] = ""
        if "Year" not in df.columns:
            df["Year"] = np.nan
        df["playerID"] = df["playerID"].astype(str).str.strip()
        df["Year"]     = pd.to_numeric(df["Year"], errors="coerce")

    num_fp = ["Eccentric Duration","Concentric Duration","Braking Phase Duration",
              "Countermovement Depth","Concentric Impulse","Concentric Impulse-100ms",
              "P1 Concentric Impulse","P2 Concentric Impulse",
              "Jump Height (Flight Time) in Inches", "Force at Zero Velocity",
              "Peak Power", "Mass", "Height", "RSI-modified","Peak Power / BM"]
    for c in num_fp:
        if c in fp_raw.columns:
            fp_raw[c] = pd.to_numeric(fp_raw[c], errors="coerce")

    for c in ["Mass","Height","Seated Height","Wingspan"]:
        if c in isak_raw.columns:
            isak_raw[c] = pd.to_numeric(isak_raw[c], errors="coerce")

    for c in ["10yd Split","20yd Split","30yd Split"]:
        if c in sprint_raw.columns:
            sprint_raw[c] = pd.to_numeric(sprint_raw[c], errors="coerce")

    df = fp_raw.merge(
        isak_raw.drop(columns=["athleteName"], errors="ignore"),
        on=["playerID", "Year"], how="outer", suffixes=("_fp", "_isak")
    )

    # If the new Force Plate 2026 tab includes Height/Mass/School Type, coalesce
    # those values with the ISAK sheet so the scoring columns remain named exactly
    # Height, Mass, Seated Height, Wingspan, and School Type. ISAK wins when present;
    # Force Plate 2026 is the fallback for players who do not exist in ISAK yet.
    def coalesce_pair(base, preferred_suffix="_isak", fallback_suffix="_fp"):
        preferred = f"{base}{preferred_suffix}"
        fallback = f"{base}{fallback_suffix}"
        if preferred in df.columns and fallback in df.columns:
            df[base] = df[preferred].where(df[preferred].notna() & (df[preferred] != ""), df[fallback])
            df.drop(columns=[preferred, fallback], inplace=True)
        elif preferred in df.columns:
            df[base] = df[preferred]
            df.drop(columns=[preferred], inplace=True)
        elif fallback in df.columns:
            df[base] = df[fallback]
            df.drop(columns=[fallback], inplace=True)

    for _base in ["Mass", "Height", "Seated Height", "Wingspan", "School Type"]:
        coalesce_pair(_base)

    df = df.merge(sprint_raw.drop(columns=["athleteName"], errors="ignore"),
                  on=["playerID","Year"], how="outer")

    # Extra name-based fallback for Sprint 2026. This protects new players whose
    # sprint row uses Name as playerID while another tab later receives a formal
    # ExternalId/playerID. Official playerID/year matches still win first.
    if "athleteName" in df.columns and not sprint_raw.empty:
        sp_lookup = sprint_raw.copy()
        for _c in ["athleteName", "Year", "10yd Split", "20yd Split", "30yd Split"]:
            if _c not in sp_lookup.columns:
                sp_lookup[_c] = np.nan if _c != "athleteName" else ""
        sp_lookup["athleteName"] = sp_lookup["athleteName"].astype(str).str.strip()
        sp_lookup["Year"] = pd.to_numeric(sp_lookup["Year"], errors="coerce")
        sp_lookup = sp_lookup.dropna(subset=["athleteName", "Year"], how="any")
        if not sp_lookup.empty:
            sp_lookup = sp_lookup.drop_duplicates(["athleteName", "Year"], keep="last")
            sp_lookup = sp_lookup.set_index(["athleteName", "Year"])[["10yd Split", "20yd Split", "30yd Split"]]
            idx = pd.MultiIndex.from_arrays([df["athleteName"].astype(str).str.strip(), pd.to_numeric(df["Year"], errors="coerce")])
            for _split in ["10yd Split", "20yd Split", "30yd Split"]:
                if _split not in df.columns:
                    df[_split] = np.nan
                fallback_vals = pd.Series(idx.map(sp_lookup[_split].to_dict()), index=df.index)
                df[_split] = pd.to_numeric(df[_split], errors="coerce").fillna(fallback_vals)

    # Position can now come from either the regular Positions tab or the raw
    # Force Plate 2026 tab. Prefer the official Positions tab when available,
    # but use the 2026 tab as the fallback for new players.
    if "Position" not in df.columns:
        df["Position"] = ""
    df["Position"] = df["Position"].astype(str).replace({"nan": "", "None": ""}).str.strip()

    if "playerID" in pos_raw.columns and "Position" in pos_raw.columns:
        pos_raw["playerID"] = pos_raw["playerID"].astype(str).str.strip()
        pos_lookup = (pos_raw[["playerID", "Position"]]
                      .dropna()
                      .drop_duplicates("playerID")
                      .set_index("playerID")["Position"]
                      .astype(str)
                      .str.strip()
                      .to_dict())
        official_pos = df["playerID"].map(pos_lookup).fillna("")
        df["Position"] = official_pos.where(official_pos.ne(""), df["Position"])

    # Extra fallback for new Force Plate 2026 players: if the official Positions
    # tab does not contain the player yet, use the Position column that was added
    # to the end of Force Plate 2026. Do this by playerID first, then by athleteName.
    if "Position" in fp_raw.columns:
        fp_pos = fp_raw.copy()
        for _c in ["playerID", "athleteName", "Position"]:
            if _c not in fp_pos.columns:
                fp_pos[_c] = ""
        fp_pos["playerID"] = fp_pos["playerID"].astype(str).str.strip()
        fp_pos["athleteName"] = fp_pos["athleteName"].astype(str).str.strip()
        fp_pos["Position"] = fp_pos["Position"].astype(str).replace({"nan": "", "None": ""}).str.strip()
        fp_pos = fp_pos[fp_pos["Position"].ne("")].copy()

        if not fp_pos.empty:
            fp_pos_by_id = (fp_pos.drop_duplicates("playerID", keep="last")
                                  .set_index("playerID")["Position"].to_dict())
            fp_pos_by_name = (fp_pos.drop_duplicates("athleteName", keep="last")
                                    .set_index("athleteName")["Position"].to_dict())
            fallback_by_id = df["playerID"].astype(str).str.strip().map(fp_pos_by_id).fillna("")
            if "athleteName" in df.columns:
                fallback_by_name = df["athleteName"].astype(str).str.strip().map(fp_pos_by_name).fillna("")
            else:
                fallback_by_name = pd.Series("", index=df.index)
            current_pos = df["Position"].astype(str).replace({"nan": "", "None": ""}).str.strip()
            df["Position"] = current_pos.where(current_pos.ne(""), fallback_by_id)
            current_pos = df["Position"].astype(str).replace({"nan": "", "None": ""}).str.strip()
            df["Position"] = current_pos.where(current_pos.ne(""), fallback_by_name)

    # Normalize common position entries before grouping/PDF output.
    df["Position"] = (df["Position"].astype(str)
                      .replace({"nan": "", "None": ""})
                      .str.strip()
                      .str.upper())
    df["Position"] = df["Position"].replace({
        "OF": "OF", "OUTFIELD": "OF", "OUTFIELDER": "OF",
        "INF": "INF", "IF": "INF", "INFIELD": "INF", "INFIELDER": "INF",
        "P": "RHP", "PITCHER": "RHP",
        "CATCHER": "C",
        "FIRST BASE": "1B", "SECOND BASE": "2B", "THIRD BASE": "3B", "SHORTSTOP": "SS",
        "LEFT FIELD": "LF", "CENTER FIELD": "CF", "RIGHT FIELD": "RF",
    })

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
                 wp_peakpow=0.20, wp_height=0.25, wp_bmi=0.30,
                 wp_school=0.15, wp_wingspan=0.10):
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

    # Keep the source metrics untouched for scorecard/scatter display. The strategy
    # model can use median-imputed values internally, but a missing CI-100ms value
    # must remain missing everywhere the athlete's measured force-plate data are shown.
    strategy_input = df[strategy_features].copy()
    for c in strategy_features:
        strategy_input[c] = pd.to_numeric(strategy_input[c], errors="coerce")
        strategy_input[c] = strategy_input[c].fillna(strategy_input[c].median())

    for c in strategy_features:
        df[f"rz_{c}"] = robust_z(strategy_input[c])
    all_rz_cols = [f"rz_{f}" for f in strategy_features]

    X_s      = strategy_input.fillna(0).values.astype(float)
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
    # Capacity Score is now the raw weighted-percentile composite.
    # This avoids min-max scaling, so scores are more stable across loaded pools/filters.
    df["athlete_quality_score"] = df["athlete_quality_raw"]

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

    # ── Potential to Gain Score ───────────────────────────────────────────────────────
    df["pp_pct"]      = df["pp_pct_alltime"]
    df["height_pct"]  = pct_rank(df["Height"])
    df["bmi_raw"]     = safe_div(df["Mass"] * 2.20462, df["Height"] / 2.54)
    df["bmi_pct"]     = pct_rank(df["bmi_raw"])
    # For potential, lower BW/Ht percentile = more projectable frame.
    # Keep bmi_pct as the displayed BW/Ht percentile, but invert it for Potential to Gain Score only.
    df["bwht_potential_pct"] = 100 - df["bmi_pct"]
    school_score_map  = {"High School": 100, "Junior College": 75, "4-Year College": 60}
    df["school_score"] = df["School Type"].map(school_score_map).fillna(50)
    df["wingspan_advantage"] = df["Wingspan"] - df["Height"]
    df["wingspan_pct"]       = pct_rank(df["wingspan_advantage"])
    # 2026 ForceDecks entries do not currently include wingspan. Treat missing
    # wingspan as neutral/50th percentile for Potential to Gain scoring only.
    # Raw wingspan still displays as missing on the scorecard.
    df["wingspan_pct_for_scoring"] = df["wingspan_pct"].fillna(50)

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
            "bwht_potential_pct": (sv("bwht_potential_pct"), wp_bmi),
            "school_score": (sv("school_score"), wp_school),
            "wingspan_pct_for_scoring": (sv("wingspan_pct_for_scoring"), wp_wingspan),
        }
        total_w = sum(w for _, (v, w) in components.items() if v is not None)
        if total_w == 0: return np.nan
        return sum(v*w for _, (v, w) in components.items() if v is not None) / total_w * 100

    df["potential_raw"]   = df.apply(pot_score, axis=1)
    # Display Potential to Gain as a percentile-like 0-100 score so interpretation is clearer:
    # 50 is roughly the middle of the loaded player pool; higher = more room/profile to add capacity.
    df["potential_score"] = pct_rank(df["potential_raw"])
    df["potential_score_yr"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "potential_score_yr"] = pct_rank(df.loc[idx, "potential_raw"])

    df["programming_category"] = df.apply(
        lambda r: programming_category(
            r.get("Concentric Impulse", np.nan),
            r.get("P1 Concentric Impulse", np.nan)), axis=1)
    df["athlete_group"] = df["programming_category"].apply(athlete_group_label)
    df["program_focus"] = df["programming_category"].apply(program_focus_label)

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

/* Compact leaderboard median rows */
.median-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(105px, 1fr));
    gap: 8px;
    margin: 4px 0 14px 0;
}}
.median-card {{
    background: white;
    border: 1px solid #E8ECF0;
    border-radius: 8px;
    padding: 8px 10px;
    min-height: 58px;
    box-shadow: 0 1px 4px rgba(17,34,90,0.04);
}}
.median-card-label {{
    font-size: 10px;
    line-height: 1.1;
    color: #6b7fa3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.median-card-value {{
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 800;
    line-height: 1.05;
    color: #11225A;
    margin-top: 6px;
}}

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

    def raw_fmt(key, digits=1, suffix="", missing_label="—"):
        v = val(key)
        return missing_label if pd.isna(v) else f"{v:.{digits}f}{suffix}"

    sections = []
    sections.append(("Force Plate", [
        ("CI", pct("ci_pct_alltime"), raw_fmt("Concentric Impulse", 1)),
        ("P1 Conc. Impulse", pct("p1_ci_pct_alltime"), raw_fmt("P1 Concentric Impulse", 1)),
        # Do not assign a neutral/50th-percentile value when CI-100ms was not measured.
        ("CI-100ms", pct("ci100_pct_alltime"), raw_fmt("Concentric Impulse-100ms", 1, missing_label="Missing")),
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
        ("Bodyweight", pct("mass_pct_alltime", pct("bmi_pct", 50)), fmt_mass(val("Mass"))),
        ("Wingspan", pct("wingspan_pct"), fmt_wingspan(val("Wingspan"))),
        ("BW/Ht", pct("bmi_pct"), pct_sfx(pct("bmi_pct")) if pd.notna(pct("bmi_pct")) else "—"),
    ]
    if is_pitcher:
        anthro_rows.insert(3, ("Wing Adv.", pct("wingspan_pct"), fmt_wingspan_adv(val("wingspan_advantage"))))
    sections.append(("Physical Attributes", anthro_rows))

    labels, vals, raw_vals, section_for_row = [], [], [], []
    y = []
    section_title_y = []
    cur_y = 0
    for section, rows in sections:
        section_title_y.append((section, cur_y))
        cur_y -= 0.75
        for lab, pc, rv in rows:
            labels.append(lab)
            # Preserve missing percentiles as NaN so Plotly does not draw a
            # zero/neutral proxy bar or marker for an unmeasured metric.
            vals.append(np.nan if pd.isna(pc) else pc)
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
        marker=dict(size=28, color=bar_colors, line=dict(color="white", width=2)),
        text=["—" if pd.isna(v) else f"{int(round(v))}" for v in vals],
        textfont=dict(color="white", size=11, family="Arial Black"),
        textposition="middle center", hoverinfo="skip", showlegend=False,
    ))

    # Dashed row separators.
    for yy in y:
        fig.add_shape(type="line", x0=-42, x1=132, y0=yy - 0.5, y1=yy - 0.5,
                      line=dict(color="rgba(150,150,150,0.35)", width=1, dash="dash"), layer="below")

    # Vertical percentile guide lines.
    for xg in [0, 25, 50, 75, 100]:
        fig.add_vline(x=xg, line_width=1, line_dash="dash" if xg in [25,50,75] else "solid",
                      line_color="rgba(170,170,170,0.55)")

    # Metric labels and raw values.
    for lab, yy, rv in zip(labels, y, raw_vals):
        fig.add_annotation(x=-10, y=yy, text=lab, showarrow=False, xanchor="right",
                           font=dict(size=13, color="#20232A"))
        fig.add_annotation(x=124, y=yy, text=rv, showarrow=False, xanchor="center",
                           font=dict(size=13, color="#20232A"))

    # Section titles.
    for section, yy in section_title_y:
        fig.add_annotation(x=-42, y=yy, text=f"<b>{section}</b>", showarrow=False,
                           xanchor="left", font=dict(size=18, color="#20232A"))
        fig.add_shape(type="line", x0=-42, x1=132, y0=yy - 0.35, y1=yy - 0.35,
                      line=dict(color="rgba(120,120,120,0.45)", width=1), layer="below")

    fig.update_layout(
        barmode="overlay",
        height=max(520, 70 * len(y) + 85 * len(sections)),
        margin=dict(l=20, r=20, t=16, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Source Sans 3, Arial, sans-serif", color=NAV),
        xaxis=dict(range=[-44, 136], tickmode="array", tickvals=[0,25,50,75,100],
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




def make_force_plate_metric_scatter(
    plot_df,
    x_col,
    x_label,
    y_col,
    y_label,
    percentile_basis="All loaded data",
    x_benchmark_value=None,
    x_benchmark_label=None,
    y_benchmark_value=None,
    y_benchmark_label=None,
    show_quadrants=True,
):
    """Interactive force-plate scatter with independent X/Y metric selection and adjustable benchmarks."""
    data = plot_df.copy()

    # Percentiles shown in hover. By default this uses the all-time/dashboard
    # percentiles created in build_scores; optionally the tab can overwrite these
    # with percentiles based only on the currently filtered chart cohort.
    if percentile_basis == "Filtered chart cohort":
        data["ci_hover_pct"] = pct_rank(pd.to_numeric(data["Concentric Impulse"], errors="coerce"))
        data["p1_hover_pct"] = pct_rank(pd.to_numeric(data["P1 Concentric Impulse"], errors="coerce"))
        data["ci100_hover_pct"] = pct_rank(pd.to_numeric(data["Concentric Impulse-100ms"], errors="coerce"))
        data["rsi_hover_pct"] = pct_rank(pd.to_numeric(data["RSI-modified"], errors="coerce"))
    else:
        data["ci_hover_pct"] = pd.to_numeric(data.get("ci_pct_alltime"), errors="coerce")
        data["p1_hover_pct"] = pd.to_numeric(data.get("p1_ci_pct_alltime"), errors="coerce")
        data["ci100_hover_pct"] = pd.to_numeric(data.get("ci100_pct_alltime"), errors="coerce")
        data["rsi_hover_pct"] = pd.to_numeric(data.get("rsi_pct_alltime"), errors="coerce")

    data["Year_Display"] = pd.to_numeric(data["Year"], errors="coerce").astype("Int64").astype(str)
    data["Position_Display"] = data["Position"].astype(str).replace({"nan": "—", "": "—"})
    data["pos_group"] = data["pos_group"].astype(str).replace({"nan": "Unknown", "": "Unknown"})
    if "Quadrant" not in data.columns:
        data["Quadrant"] = "—"
    data["Quadrant_Display"] = data["Quadrant"].astype(str).replace({"nan": "—", "": "—"})

    custom_cols = [
        "athleteName", "Position_Display", "Year_Display",
        "Concentric Impulse", "P1 Concentric Impulse", "Concentric Impulse-100ms", "RSI-modified",
        "ci_hover_pct", "p1_hover_pct", "ci100_hover_pct", "rsi_hover_pct", "Quadrant_Display",
    ]

    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="pos_group",
        custom_data=custom_cols,
        labels={x_col: x_label, y_col: y_label, "pos_group": "Position group"},
    )
    fig.update_traces(
        marker=dict(size=10, opacity=0.78, line=dict(width=1, color="white")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Position: %{customdata[1]} | Year: %{customdata[2]}<br>"
            "Quadrant: %{customdata[11]}<br><br>"
            "CI (Total Output): %{customdata[3]:.1f} (%{customdata[7]:.0f}th)<br>"
            "P1 CI (Strength): %{customdata[4]:.1f} (%{customdata[8]:.0f}th)<br>"
            "CI-100ms: %{customdata[5]:.1f} (%{customdata[9]:.0f}th)<br>"
            "mRSI (Twitch): %{customdata[6]:.3f} (%{customdata[10]:.0f}th)"
            "<extra></extra>"
        ),
    )

    x_benchmark_value = None if x_benchmark_value is None or pd.isna(x_benchmark_value) else float(x_benchmark_value)
    y_benchmark_value = None if y_benchmark_value is None or pd.isna(y_benchmark_value) else float(y_benchmark_value)

    # Compute stable chart ranges that include the benchmark lines when present.
    x_values = pd.to_numeric(data[x_col], errors="coerce").dropna().tolist()
    y_values = pd.to_numeric(data[y_col], errors="coerce").dropna().tolist()
    if x_benchmark_value is not None:
        x_values.append(x_benchmark_value)
    if y_benchmark_value is not None:
        y_values.append(y_benchmark_value)

    if x_values and y_values:
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_pad = (x_max - x_min) * 0.08 if not np.isclose(x_max, x_min) else max(abs(x_max) * 0.08, 1.0)
        y_pad = (y_max - y_min) * 0.08 if not np.isclose(y_max, y_min) else max(abs(y_max) * 0.08, 0.05)
        x_range = [x_min - x_pad, x_max + x_pad]
        y_range = [y_min - y_pad, y_max + y_pad]
    else:
        x_range = None
        y_range = None

    # Optional quadrant background and labels. Assumes higher is better for all current options.
    if show_quadrants and x_benchmark_value is not None and y_benchmark_value is not None and x_range and y_range:
        x0, x1 = x_range
        y0, y1 = y_range
        quadrant_shapes = [
            (x0, x_benchmark_value, y0, y_benchmark_value, "rgba(186,12,47,0.055)"),
            (x_benchmark_value, x1, y0, y_benchmark_value, "rgba(226,193,136,0.070)"),
            (x0, x_benchmark_value, y_benchmark_value, y1, "rgba(107,127,163,0.065)"),
            (x_benchmark_value, x1, y_benchmark_value, y1, "rgba(76,175,130,0.070)"),
        ]
        for xa, xb, ya, yb, fill in quadrant_shapes:
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=xa, x1=xb, y0=ya, y1=yb,
                fillcolor=fill, line=dict(width=0), layer="below",
            )

        def mid(a, b):
            return (float(a) + float(b)) / 2.0

        quadrant_annotations = [
            (mid(x0, x_benchmark_value), mid(y0, y_benchmark_value), f"Low {x_label}<br>Low {y_label}"),
            (mid(x_benchmark_value, x1), mid(y0, y_benchmark_value), f"High {x_label}<br>Low {y_label}"),
            (mid(x0, x_benchmark_value), mid(y_benchmark_value, y1), f"Low {x_label}<br>High {y_label}"),
            (mid(x_benchmark_value, x1), mid(y_benchmark_value, y1), f"High {x_label}<br>High {y_label}"),
        ]
        for xa, ya, txt in quadrant_annotations:
            fig.add_annotation(
                x=xa, y=ya, text=txt, showarrow=False,
                font=dict(size=11, color="rgba(17,34,90,0.48)"),
                align="center", bgcolor="rgba(255,255,255,0.45)", borderpad=3,
            )

    # Add benchmark lines on whichever axes have active benchmark settings.
    if x_benchmark_value is not None:
        fig.add_vline(
            x=x_benchmark_value,
            line_width=1.7,
            line_dash="dash",
            line_color=RED,
            annotation_text=x_benchmark_label or f"{x_benchmark_value:g} {x_label}",
            annotation_position="top right",
        )
    if y_benchmark_value is not None:
        fig.add_hline(
            y=y_benchmark_value,
            line_width=1.7,
            line_dash="dash",
            line_color=RED,
            annotation_text=y_benchmark_label or f"{y_benchmark_value:g} {y_label}",
            annotation_position="top right",
        )

    fig.update_layout(**_layout(
        title=dict(text=f"{y_label} vs {x_label}", font=dict(size=18, color=NAV), x=0),
        height=650,
        margin=dict(l=35, r=25, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    fig.update_xaxes(title=x_label, zeroline=False, range=x_range)
    fig.update_yaxes(title=y_label, zeroline=False, range=y_range)
    return fig


def make_scorecard_pdf(row, df_all, strat_feats, sel_yr_display, is_pitcher=False):
    """Render the PDF export using the exact Scorecard Option 1 skin."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    try:
        from datetime import datetime
        import importlib
        import scorecard_skins
        scorecard_skins = importlib.reload(scorecard_skins)
        render_scorecard_option_1 = scorecard_skins.render_scorecard_option_1
    except Exception as e:
        raise ImportError("PDF export needs scorecard_skins.py and reportlab>=4.0 available in the app environment.") from e

    def get_val(key):
        try:
            v = row.get(key, np.nan)
            f = float(v)
            return f if not np.isnan(f) else np.nan
        except Exception:
            return np.nan

    def safe_file_name(name):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "athlete"

    def raw_num(key, digits=1, suffix="", missing_label="-"):
        v = get_val(key)
        return missing_label if pd.isna(v) else f"{v:.{digits}f}{suffix}"

    def safe_pdf_percentile(v, fallback=50):
        """Return a finite 0-100 integer, or None when a row should stay missing."""
        try:
            f = float(v)
            if not np.isfinite(f):
                return None if fallback is None else int(fallback)
            return max(0, min(100, int(round(f))))
        except Exception:
            return None if fallback is None else int(fallback)

    def pct_from_col(pct_key, fallback=50):
        v = get_val(pct_key)
        return safe_pdf_percentile(v, fallback=fallback)

    def pool_pct(value_key, inverse=False, fallback=50):
        v = get_val(value_key)
        if pd.isna(v) or value_key not in df_all.columns:
            return int(fallback)
        pool = pd.to_numeric(df_all[value_key], errors="coerce").dropna()
        if len(pool) == 0:
            return int(fallback)
        pct = (pool < v).mean() * 100.0
        pct = 100.0 - pct if inverse else pct
        return safe_pdf_percentile(pct, fallback=fallback)

    def pct_sfx_local(p):
        if p is None or pd.isna(p):
            return "-"
        p = int(round(float(p)))
        if 11 <= p % 100 <= 13:
            return f"{p}th"
        return f"{p}{({1:'st', 2:'nd', 3:'rd'}.get(p % 10, 'th'))}"

    def clean_text(x):
        if x is None or str(x) in ("nan", "None", "<NA>"):
            return "-"
        return str(x)

    def resolve_logo_path():
        candidates = [
            Path(os.environ.get('NATS_LOGO_PATH', '')),
            Path('nationals_logo.png'),
            Path('nats_logo.png'),
            Path('assets/nationals_logo.png'),
            Path('assets/nats_logo.png'),
            Path('/mnt/data/nationals_logo.png'),
            Path('/mnt/data/nats_logo.png'),
        ]
        for p in candidates:
            if str(p) and p.exists() and p.is_file():
                return str(p)
        return None

    athlete = clean_text(row.get('athleteName', 'Athlete'))
    pos = clean_text(row.get('Position', '-'))
    school = clean_text(row.get('School Type', '-'))
    athlete_group = clean_text(row.get('athlete_group', athlete_group_label(row.get('programming_category', '-'))))
    program_focus = clean_text(row.get('program_focus', program_focus_label(row.get('programming_category', '-'))))
    prog_short = clean_text(row.get('programming_category', '-'))
    if prog_short == 'Unclassified':
        prog_short = '-'
    ci_tier_short = clean_text(row.get('ci_tier', ci_tier_label(get_val('Concentric Impulse'))))
    aq = get_val('athlete_quality_score')
    pos_aq = get_val('aq_pos_score')
    pot = get_val('potential_score')
    report_date = datetime.now().strftime('%-m/%-d/%y') if os.name != 'nt' else datetime.now().strftime('%#m/%#d/%y')

    def score_text(v):
        return '-' if pd.isna(v) else f"{int(round(float(v)))}"

    def sixty_bwht_capacity_card():
        """Return the upside-only Capacity Score range at the 60th BW/Ht target.

        The PDF card intentionally shows just a score range (for example, 86–89),
        not the individual relative-CI scenarios. It returns ``No Change`` when the
        athlete already meets the 60th-percentile BW/Ht target or when none of the
        modeled outcomes improve both CI and Capacity Score.
        """
        no_change = {"value": "No Change", "percentile": None}

        ci = get_val("Concentric Impulse")
        mass_kg = get_val("Mass")
        height_cm = get_val("Height")
        current_capacity = get_val("athlete_quality_score")
        if not all(pd.notna(v) and v > 0 for v in [ci, mass_kg, height_cm, current_capacity]):
            return no_change

        bwht_pool = pd.to_numeric(df_all.get("bmi_raw"), errors="coerce").dropna()
        ci_pool = pd.to_numeric(df_all.get("Concentric Impulse"), errors="coerce").dropna()
        if len(bwht_pool) == 0 or len(ci_pool) == 0:
            return no_change

        target_bwht = float(np.nanpercentile(bwht_pool, 60))
        current_bwht = (mass_kg * 2.20462) / (height_cm / 2.54)
        # Do not imply that an athlete should lose mass to return to the target.
        if pd.notna(current_bwht) and current_bwht >= target_bwht:
            return no_change

        target_lbs = target_bwht * (height_cm / 2.54)
        target_kg = target_lbs / 2.20462
        if pd.isna(target_kg) or target_kg <= 0:
            return no_change

        sprint_pct = get_val("sprint_pct_alltime")
        rsi_pct = get_val("rsi_pct_alltime")
        pp_pct = get_val("pp_pct_alltime")

        def projected_capacity(projected_ci):
            projected_ci_pct = float((ci_pool < projected_ci).mean() * 100.0)
            if pd.notna(sprint_pct):
                components = [
                    (projected_ci_pct, w_ci),
                    (sprint_pct, w_sprint),
                    (rsi_pct, w_rsi),
                    (pp_pct, w_pp),
                ]
            else:
                components = [
                    (projected_ci_pct, w_ci_ns),
                    (rsi_pct, w_rsi_ns),
                    (pp_pct, w_pp_ns),
                ]

            valid_components = [(value, weight) for value, weight in components if pd.notna(value)]
            if not valid_components:
                return np.nan
            total_weight = sum(weight for _, weight in valid_components)
            return sum(value * weight for value, weight in valid_components) / total_weight

        ci_per_kg = ci / mass_kg
        # +1% CI/kg improvement through a 4% CI/kg penalty.
        ci_rel_changes = [0.01, 0.00, -0.01, -0.02, -0.03, -0.04]
        upside_capacity_scores = []
        for rel_change in ci_rel_changes:
            projected_ci = ci_per_kg * (1 + rel_change) * target_kg
            projected_capacity_score = projected_capacity(projected_ci)
            if (
                pd.notna(projected_ci)
                and pd.notna(projected_capacity_score)
                and projected_ci > ci
                and projected_capacity_score >= current_capacity
            ):
                upside_capacity_scores.append(projected_capacity_score)

        if not upside_capacity_scores:
            return no_change

        low = int(round(min(upside_capacity_scores)))
        high = int(round(max(upside_capacity_scores)))
        card_value = str(low) if low == high else f"{low}-{high}"
        # Use the midpoint for the card fill so the color reflects the range rather
        # than only its best-case endpoint.
        card_percentile = int(round((low + high) / 2))
        return {"value": card_value, "percentile": card_percentile}

    capacity_60th_card = sixty_bwht_capacity_card()

    force_rows = [
        {'label': 'CI', 'percentile': pct_from_col('ci_pct_alltime'), 'value': raw_num('Concentric Impulse', 1)},
        {'label': 'P1 CI', 'percentile': pct_from_col('p1_ci_pct_alltime'), 'value': raw_num('P1 Concentric Impulse', 1)},
        {'label': 'CI-100ms', 'percentile': pct_from_col('ci100_pct_alltime', fallback=None), 'value': raw_num('Concentric Impulse-100ms', 1, missing_label='Missing')},
        {'label': 'RSI-mod', 'percentile': pct_from_col('rsi_pct_alltime'), 'value': raw_num('RSI-modified', 3)},
        {'label': 'Pk Pwr/BM', 'percentile': pct_from_col('pp_pct_alltime'), 'value': raw_num('Peak Power / BM', 1)},
        {'label': 'Jump Ht', 'percentile': pct_from_col('jump_height_pct_alltime'), 'value': raw_num('Jump Height (Flight Time) in Inches', 2, ' in')},
    ]
    anthro_rows = [
        {'label': 'Height', 'percentile': pct_from_col('height_pct'), 'value': fmt_height(get_val('Height'))},
        {'label': 'Bodyweight', 'percentile': pool_pct('Mass'), 'value': fmt_mass(get_val('Mass'))},
        {'label': 'Wingspan', 'percentile': pool_pct('Wingspan'), 'value': fmt_wingspan(get_val('Wingspan'))},
        {'label': 'Wing Adv.', 'percentile': pct_from_col('wingspan_pct'), 'value': fmt_wingspan_adv(get_val('wingspan_advantage'))},
        {'label': 'BW/Ht Pct', 'percentile': pct_from_col('bmi_pct'), 'value': pct_sfx_local(pct_from_col('bmi_pct'))},
    ]
    sprint_rows = []
    if pd.notna(get_val('30yd Split')):
        sprint_rows.append({'label': '30yd', 'percentile': pct_from_col('sprint_pct_alltime'), 'value': raw_num('30yd Split', 3, 's')})
    if pd.notna(get_val('10yd Split')):
        sprint_rows.append({'label': '10yd', 'percentile': pool_pct('10yd Split', inverse=True), 'value': raw_num('10yd Split', 3, 's')})
    if pd.notna(get_val('20yd Split')):
        sprint_rows.append({'label': '20yd', 'percentile': pool_pct('20yd Split', inverse=True), 'value': raw_num('20yd Split', 3, 's')})

    scorecard_data = {
        'player_name': athlete,
        'position': pos,
        'context': f"{sel_yr_display} | {school} | Draft Scouting",
        'subtitle': f"{sel_yr_display} - {pos} - {school}",
        'banner_label': 'WASHINGTON NATIONALS - DRAFT SCOUTING',
        'report_date': report_date,
        'capacity': score_text(aq),
        'headshot_path': None,
        'logo_path': resolve_logo_path(),
        # Scorecard top cards use the stakeholder wording from the Option 1 mockup.
        # All percentile values are sanitized because the PDF renderer cannot draw NaN.
        'summary_cards': [
            {'label': 'Capacity', 'value': score_text(aq), 'percentile': safe_pdf_percentile(aq), 'filled': True},
            {'label': 'Pos. Capacity', 'value': score_text(pos_aq), 'percentile': safe_pdf_percentile(pos_aq)},
            {'label': 'Potential to Gain', 'value': score_text(pot), 'percentile': safe_pdf_percentile(pot)},
            {'label': 'Athlete Group', 'value': athlete_group},
            {'label': 'Program Focus', 'value': program_focus},
            {
                'label': 'Future Capacity',
                'value': capacity_60th_card['value'],
                'percentile': capacity_60th_card['percentile'],
            },
        ],
        'sections': [
            {'title': 'Force Plate', 'rows': force_rows},
            {'title': 'Physical Attributes', 'rows': anthro_rows},
        ],
    }
    if sprint_rows:
        scorecard_data['sections'].append({'title': 'Sprint', 'rows': sprint_rows})

    pdf_bytes = render_scorecard_option_1(scorecard_data)
    return pdf_bytes, safe_file_name(athlete)


# ─── Score Info PDF Explainer ────────────────────────────────────────────────
def make_score_info_pdf():
    """Create a one-page PDF explainer for the Score Info tab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception as e:
        raise ImportError("Score Info PDF export requires reportlab.") from e

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.42 * inch,
        leftMargin=0.42 * inch,
        topMargin=0.34 * inch,
        bottomMargin=0.34 * inch,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "NatsTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=22, textColor=colors.HexColor(NAV), alignment=TA_LEFT,
        spaceAfter=4,
    )
    kicker = ParagraphStyle(
        "Kicker", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7.5, leading=9, textColor=colors.HexColor(RED), alignment=TA_LEFT,
        uppercase=True, spaceAfter=4,
    )
    section = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=12, textColor=colors.HexColor(RED), spaceBefore=5, spaceAfter=2,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.9, leading=9.7, textColor=colors.HexColor(NAV), spaceAfter=3,
    )
    small = ParagraphStyle(
        "Small", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.2, leading=8.8, textColor=colors.HexColor(NAV), spaceAfter=2,
    )
    bullet = ParagraphStyle(
        "Bullet", parent=body, leftIndent=9, firstLineIndent=-6, spaceAfter=1.8,
    )

    story = []
    story.append(Paragraph("WASHINGTON NATIONALS · DRAFT SCOUTING", kicker))
    story.append(Paragraph("Score Info Explainer", title))

    story.append(Paragraph("Capacity Score", section))
    story.append(Paragraph('A composite score (out of 100) that represents current "physical capacity" relative to all athletes in the MLB combine dataset. Sample of 641 athletes.', body))
    story.append(Paragraph("The weighting is based on 45% CI percentile, 20% mRSI percentile, and 35% Rel/Peak Power percentile.", body))
    story.append(Paragraph("• Concentric Impulse (CI) is the #1 correlated metric to bat speed and velo.", bullet))
    story.append(Paragraph('• mRSI is representative of elasticity and "quickness." Ability to produce force quickly could have implications defensively.', bullet))
    story.append(Paragraph("• Rel/Peak Power looks at how much force an athlete can produce relative to their size. This helps round out our understanding of their athleticism, checking to see that CI isn't just driven by their size.", bullet))
    story.append(Paragraph("Not all position players have sprint values. If they do, the weighting is 35% CI, 30% 30 yard sprint, 15% mRSI, and 20% Rel/Peak Power.", body))
    story.append(Paragraph("These are the weights that S&C currently feels most convicted in. Since a small percentage of players do not participate in all parts of the combine, this score should be used as a rough guideline for athleticism rather than an exact number.", body))

    story.append(Paragraph("Capacity vs Position Group", section))
    story.append(Paragraph("Where the athlete's score falls relative to their position group: Pitchers, Outfielders, Infielders, or Catchers. Players without a designated position may not have a position-group score.", body))

    story.append(Paragraph("Athlete Group", section))
    story.append(Paragraph("Shows the initial CI/P1 bucket we would place them in and how we would approach their training in Player Development.", body))

    story.append(Paragraph("Potential to Gain", section))
    story.append(Paragraph("A composite score (out of 100) that represents ability to gain CI moving forward. Athletes with the most potential to gain are typically tall, skinny and/or springy, meaning there may be more room to increase CI by increasing body weight or filling out their frame.", body))
    story.append(Paragraph("The weighting is based on 20% Rel/Peak Power percentile, 25% height percentile, 30% BW/Ht percentile inverted, 10% wingspan advantage, and 15% school type. School type is used as a proxy for training history.", body))

    story.append(Paragraph("Force Plate Metric Notes", section))
    notes = [
        [Paragraph("<b>Concentric Impulse</b>", small), Paragraph("Total concentric force output; the main capacity metric and the #1 correlated metric to bat speed and velo.", small)],
        [Paragraph("<b>P1 Concentric Impulse</b>", small), Paragraph("Breaking inertia: early concentric force determines how easily the system starts moving and overcoming inertia.", small)],
        [Paragraph("<b>Relative Peak Power</b>", small), Paragraph("How much force an athlete can produce relative to their size.", small)],
        [Paragraph("<b>mRSI</b>", small), Paragraph('Representative of elasticity and "quickness." Ability to produce force quickly.', small)],
    ]
    tbl = Table(notes, colWidths=[1.55 * inch, 5.0 * inch], hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F7F8FA")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor(BORD)),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(tbl)
    doc.build(story)
    return buffer.getvalue()


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
st.markdown("""
<style>
.st-key-legacy_scorecard_toggle {
    position: fixed !important;
    bottom: 8px !important;
    left: 10px !important;
    z-index: 9999 !important;
    width: 32px !important;
    opacity: 0.30;
}
.st-key-legacy_scorecard_toggle:hover { opacity: 0.9; }
.st-key-legacy_scorecard_toggle button {
    min-height: 20px !important;
    height: 22px !important;
    width: 22px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    font-size: 9px !important;
    line-height: 1 !important;
    color: transparent !important;
    background: #9AAAC0 !important;
    border: 0 !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)
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

wp_pp = 0.20; wp_ht = 0.25; wp_bmi = 0.30; wp_school = 0.15; wp_wings = 0.10

# ─── Load & score ─────────────────────────────────────────────────────────────
try:
    raw = load_data(_v=4)
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

# Keep the full athlete scorecard available, but hide it by default so the main workflow stays clean.
if "show_athlete_scorecard_tab" not in st.session_state:
    st.session_state.show_athlete_scorecard_tab = False

# Tiny legacy scorecard toggle, visually tucked into the bottom-left corner.
# It stays before the tab declaration in code because Streamlit needs to know
# which tabs exist before rendering their contents.
with st.container(key="legacy_scorecard_toggle"):
    if st.button("•", key="toggle_athlete_scorecard_tab", help="Show/hide the full legacy athlete scorecard tab"):
        st.session_state.show_athlete_scorecard_tab = not st.session_state.show_athlete_scorecard_tab

if st.session_state.show_athlete_scorecard_tab:
    tab_board, tab_scatter, tab_card, tab_2026, tab_info = st.tabs(["Leaderboard", "Scatter Plot", "Athlete Scorecard", "2026 Scorecards", "Score Info"])
else:
    tab_board, tab_scatter, tab_2026, tab_info = st.tabs(["Leaderboard", "Scatter Plot", "2026 Scorecards", "Score Info"])
    tab_card = None


# =============================================================================
# TAB 0 — SCORE INFO
# =============================================================================
with tab_info:
    try:
        score_info_pdf = make_score_info_pdf()
        st.download_button(
            "Download One-Page PDF Explainer",
            data=score_info_pdf,
            file_name="score_info_explainer.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Score Info PDF export is unavailable: {e}")

    st.markdown(f"""
    <div class="card card-navy">
        <p class="label">Scoring Methodology</p>
        <h2 style="margin-top:0;color:{NAV};font-family:'Playfair Display',serif;">Score Info</h2>
        <p style="font-size:14px;line-height:1.55;color:{NAV};margin-bottom:0;">
            This tab describes how the dashboard's major summary scores should be interpreted.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card card-red">
        <h3 style="margin-top:0;color:{RED};font-family:'Playfair Display',serif;">Capacity Score:</h3>
        <p style="font-size:14px;line-height:1.6;color:{NAV};">
            A composite score (out of 100) that represents their current "physical capacity" relative to all athletes in the MLB combine dataset. (Sample of 641 athletes.)
        </p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};">
            The weighting is based on 45% CI percentile, 20% mRSI percentile, and 35% Rel/Peak Power percentile.
        </p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};">
            Concentric Impulse (CI) is the #1 correlated metric to bat speed and velo.
        </p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};">
            mRSI is representative of elasticity and "quickness." Ability to produce force quickly could have implications defensively.
        </p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};">
            Rel/Peak Power looks at how much force an athlete can produce relative to their size. This helps round out our understanding of their athleticism, checking to see that CI isn't just driven by their size.
        </p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};">
            Not all position players have sprint values. If they do, the weighting is 35% CI, 30% 30 yard sprint, 15% mRSI, and 20% Rel/Peak Power.
        </p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};margin-bottom:0;">
            These are the weights that S&C currently feels most convicted in. Since a small percentage of players do not participate in all parts of the combine (some skip the sprints or jumps altogether), this score is worth using as a rough guideline for their athleticism rather than an exact number. (Think of using this score as a litmus test for how closely our subjective idea of a player's athleticism lines up with the objective data they would be assessed on in the org.)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card card-navy">
        <h3 style="margin-top:0;color:{NAV};font-family:'Playfair Display',serif;">Capacity vs Position Group:</h3>
        <p style="font-size:14px;line-height:1.6;color:{NAV};margin-bottom:0;">
            Where their score falls relative to their position. Potential position groups are Pitchers, Outfielders, Infielders, Catchers. (A small percentage of players who didn't have a position designated for them in the combine data will not have a score for this.)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card card-gold">
        <h3 style="margin-top:0;color:{NAV};font-family:'Playfair Display',serif;">Athlete Group:</h3>
        <p style="font-size:14px;line-height:1.6;color:{NAV};margin-bottom:0;">
            Shows the initial CI/P1 bucket we would place them in and how we would approach their training in Player Development.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card card-green">
        <h3 style="margin-top:0;color:{GREEN};font-family:'Playfair Display',serif;">Potential to Gain:</h3>
        <p style="font-size:14px;line-height:1.6;color:{NAV};">
            A composite score (out of 100) that represents their ability to gain CI moving forward. Athletes with the most potential to gain are typically tall, skinny and/or springy. (In other words, there is more potential room to increase CI simply by increasing body weight / filling out their frame.)
        </p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};margin-bottom:0;">
            The weighting is based on 20% Rel/Peak Power percentile, 25% height percentile, 30% Bw/Ht percentile (inverted so a skinnier guy is shown as having more potential), 10% wingspan advantage (how much greater their wingspan is than their height), and 15% school type (High school has more potential than Juco, which has more than a 4 year, etc. This is used as a proxy for the quality of their training history.)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card card-red">
        <h3 style="margin-top:0;color:{RED};font-family:'Playfair Display',serif;">Force Plate Metric Notes:</h3>
        <p style="font-size:14px;line-height:1.6;color:{NAV};"><b>Concentric Impulse</b> - Total concentric force output; the main capacity metric and the #1 correlated metric to bat speed and velo.</p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};"><b>P1 Concentric Impulse</b> - Breaking Inertia, early concentric force determines how easily the system starts moving and overcoming inertia.</p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};"><b>Relative Peak Power</b> - How much force an athlete can produce relative to their size.</p>
        <p style="font-size:14px;line-height:1.6;color:{NAV};margin-bottom:0;"><b>mRSI</b> - Representative of elasticity and "quickness." Ability to produce force quickly.</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 1 — FORCE PLATE SCATTER
# =============================================================================
with tab_scatter:
    st.markdown(f"""
    <div class="card card-red">
        <p class="label">Force Plate Relationship</p>
        <h2 style="margin-top:0;color:{NAV};font-family:'Playfair Display',serif;">Scatter Plot</h2>
        <p style="font-size:14px;line-height:1.55;color:{NAV};margin-bottom:0;">
            Pick any force plate metric for either axis. Hover over any dot to see the athlete, raw metrics, and percentile ranks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    scatter_df = df.copy()
    scatter_df["Year"] = pd.to_numeric(scatter_df["Year"], errors="coerce")
    scatter_df["Position"] = scatter_df["Position"].astype(str).replace({"nan": "", "None": ""}).str.strip().str.upper()
    scatter_df["pos_group"] = scatter_df["pos_group"].astype(str).replace({"nan": "Unknown", "": "Unknown"})

    metric_options = {
        "P1 CI (Strength)": "P1 Concentric Impulse",
        "CI (Total Output)": "Concentric Impulse",
        "CI-100ms": "Concentric Impulse-100ms",
        "mRSI (Twitch)": "RSI-modified",
    }
    metric_percentile_cols = {
        "P1 CI (Strength)": "p1_ci_pct_alltime",
        "CI (Total Output)": "ci_pct_alltime",
        "CI-100ms": "ci100_pct_alltime",
        "mRSI (Twitch)": "rsi_pct_alltime",
    }
    metric_digits = {
        "P1 CI (Strength)": 1,
        "CI (Total Output)": 1,
        "CI-100ms": 1,
        "mRSI (Twitch)": 3,
    }

    years_available = sorted(scatter_df["Year"].dropna().astype(int).unique().tolist(), reverse=True)
    default_years = [2026] if 2026 in years_available else years_available
    pos_groups_available = [g for g in ["Pitcher", "Catcher", "Infielder", "Outfielder", "Unknown"] if g in scatter_df["pos_group"].unique()]
    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        selected_years = st.multiselect("Years", years_available, default=default_years, key="scatter_years")
    with c2:
        selected_pos_groups = st.multiselect(
            "Position groups",
            pos_groups_available,
            default=pos_groups_available,
            key="scatter_pos_groups",
        )

    c4, c5, c6 = st.columns([1.0, 1.0, 2.0])
    with c4:
        x_label = st.selectbox("X-axis metric", list(metric_options.keys()), index=0, key="scatter_x_metric")
    with c5:
        y_label = st.selectbox("Y-axis metric", list(metric_options.keys()), index=3, key="scatter_y_metric")
    with c6:
        percentile_basis = st.radio(
            "Hover percentiles based on",
            ["All loaded data", "Filtered chart cohort"],
            horizontal=True,
            key="scatter_percentile_basis",
        )

    if selected_years:
        scatter_df = scatter_df[scatter_df["Year"].isin(selected_years)]
    else:
        scatter_df = scatter_df.iloc[0:0]

    if selected_pos_groups:
        scatter_df = scatter_df[scatter_df["pos_group"].isin(selected_pos_groups)]
    else:
        scatter_df = scatter_df.iloc[0:0]

    x_col = metric_options[x_label]
    y_col = metric_options[y_label]
    required_cols = [
        "athleteName", "Year", "Position", "pos_group",
        "Concentric Impulse", "P1 Concentric Impulse", "Concentric Impulse-100ms", "RSI-modified",
        "ci_pct_alltime", "p1_ci_pct_alltime", "ci100_pct_alltime", "rsi_pct_alltime",
    ]
    for col in required_cols:
        if col not in scatter_df.columns:
            scatter_df[col] = np.nan

    chart_df = scatter_df.dropna(subset=list(dict.fromkeys([x_col, y_col]))).copy()

    # ── Adjustable benchmark and quadrant controls ───────────────────────────
    default_benchmarks = {
        "CI (Total Output)": 285.0,
        "P1 CI (Strength)": 195.0,
        "CI-100ms": 100.0,
        "mRSI (Twitch)": 0.8,
    }

    def _default_benchmark_value(metric_label, metric_col, base_df):
        if metric_label in default_benchmarks:
            return float(default_benchmarks[metric_label])
        s = pd.to_numeric(base_df.get(metric_col), errors="coerce").dropna()
        if len(s):
            return float(s.median())
        return 0.0

    def _percentile_threshold(base_df, metric_col, pct):
        s = pd.to_numeric(base_df.get(metric_col), errors="coerce").dropna()
        if len(s) == 0:
            return np.nan
        return float(np.nanpercentile(s, float(pct)))

    def _resolve_benchmark(axis_name, metric_label, metric_col, source_df):
        mode = st.session_state.get(f"scatter_{axis_name}_bench_mode", "Hard value")
        digits = metric_digits.get(metric_label, 1)
        if mode == "Off":
            return None, None
        if mode == "Hard value":
            val = st.session_state.get(
                f"scatter_{axis_name}_bench_value_{metric_label}",
                _default_benchmark_value(metric_label, metric_col, source_df),
            )
            try:
                val = float(val)
            except Exception:
                return None, None
            return val, f"{fmt(val, digits)} {metric_label}"
        pct = st.session_state.get(f"scatter_{axis_name}_bench_pct", 75)
        val = _percentile_threshold(source_df, metric_col, pct)
        if pd.isna(val):
            return None, None
        return val, f"{int(round(float(pct)))}th pct {metric_label}: {fmt(val, digits)}"

    with st.expander("Benchmark lines and quadrants", expanded=True):
        b1, b2, b3 = st.columns([1.15, 1.15, 1.25])
        with b1:
            st.markdown(f"**X-axis benchmark: {x_label}**")
            st.selectbox(
                "X benchmark type",
                ["Hard value", "Percentile", "Off"],
                index=0,
                key="scatter_x_bench_mode",
            )
            if st.session_state.get("scatter_x_bench_mode", "Hard value") == "Hard value":
                st.number_input(
                    "X hard value",
                    value=_default_benchmark_value(x_label, x_col, chart_df),
                    step=0.01 if x_col == "RSI-modified" else 1.0,
                    format="%.3f" if x_col == "RSI-modified" else "%.1f",
                    key=f"scatter_x_bench_value_{x_label}",
                )
            elif st.session_state.get("scatter_x_bench_mode") == "Percentile":
                st.slider("X percentile", 0, 100, 75, 1, key="scatter_x_bench_pct")

        with b2:
            st.markdown(f"**Y-axis benchmark: {y_label}**")
            st.selectbox(
                "Y benchmark type",
                ["Hard value", "Percentile", "Off"],
                index=0,
                key="scatter_y_bench_mode",
            )
            if st.session_state.get("scatter_y_bench_mode", "Hard value") == "Hard value":
                st.number_input(
                    "Y hard value",
                    value=_default_benchmark_value(y_label, y_col, chart_df),
                    step=0.01 if y_col == "RSI-modified" else 1.0,
                    format="%.3f" if y_col == "RSI-modified" else "%.1f",
                    key=f"scatter_y_bench_value_{y_label}",
                )
            elif st.session_state.get("scatter_y_bench_mode") == "Percentile":
                st.slider("Y percentile", 0, 100, 75, 1, key="scatter_y_bench_pct")

        with b3:
            benchmark_percentile_basis = st.radio(
                "Benchmark percentiles based on",
                ["Filtered chart cohort", "All loaded data"],
                horizontal=False,
                key="scatter_benchmark_percentile_basis",
            )
            show_quadrants = st.toggle(
                "Show quadrant shading and labels",
                value=True,
                key="scatter_show_quadrants",
            )

    benchmark_source_df = chart_df if benchmark_percentile_basis == "Filtered chart cohort" else df.copy()
    for col in required_cols:
        if col not in benchmark_source_df.columns:
            benchmark_source_df[col] = np.nan

    x_benchmark_value, x_benchmark_label = _resolve_benchmark("x", x_label, x_col, benchmark_source_df)
    y_benchmark_value, y_benchmark_label = _resolve_benchmark("y", y_label, y_col, benchmark_source_df)

    has_quadrant_benchmarks = x_benchmark_value is not None and y_benchmark_value is not None
    if has_quadrant_benchmarks and not chart_df.empty:
        x_high = pd.to_numeric(chart_df[x_col], errors="coerce") >= float(x_benchmark_value)
        y_high = pd.to_numeric(chart_df[y_col], errors="coerce") >= float(y_benchmark_value)
        chart_df["Quadrant"] = np.select(
            [x_high & y_high, x_high & ~y_high, ~x_high & y_high, ~x_high & ~y_high],
            [
                f"High {x_label} / High {y_label}",
                f"High {x_label} / Low {y_label}",
                f"Low {x_label} / High {y_label}",
                f"Low {x_label} / Low {y_label}",
            ],
            default="Unclassified",
        )
    else:
        chart_df["Quadrant"] = "—"

    if x_col == y_col:
        st.info("You selected the same metric for both axes. The chart will still render, but the dots will fall along a 1:1 relationship.")

    if chart_df.empty:
        st.warning("No athletes have both selected metrics for the active filters.")
    else:
        fig = make_force_plate_metric_scatter(
            chart_df,
            x_col,
            x_label,
            y_col,
            y_label,
            percentile_basis,
            x_benchmark_value=x_benchmark_value,
            x_benchmark_label=x_benchmark_label,
            y_benchmark_value=y_benchmark_value,
            y_benchmark_label=y_benchmark_label,
            show_quadrants=show_quadrants,
        )
        st.plotly_chart(fig, use_container_width=True)

        x_pct_col = metric_percentile_cols[x_label]
        y_pct_col = metric_percentile_cols[y_label]
        x_digits = metric_digits[x_label]
        y_digits = metric_digits[y_label]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Athletes", f"{len(chart_df):,}")
        m2.metric(f"Median {x_label}", fmt(pd.to_numeric(chart_df[x_col], errors="coerce").median(), x_digits))
        m3.metric(f"Median {y_label}", fmt(pd.to_numeric(chart_df[y_col], errors="coerce").median(), y_digits))
        m4.metric(f"Median {y_label} Pct", pct_sfx(pd.to_numeric(chart_df[y_pct_col], errors="coerce").median()))

        if show_quadrants and has_quadrant_benchmarks:
            quad_order = [
                f"High {x_label} / High {y_label}",
                f"High {x_label} / Low {y_label}",
                f"Low {x_label} / High {y_label}",
                f"Low {x_label} / Low {y_label}",
            ]
            quad_summary = (
                chart_df["Quadrant"]
                .value_counts()
                .reindex(quad_order, fill_value=0)
                .rename_axis("Quadrant")
                .reset_index(name="Athletes")
            )
            quad_summary["Share"] = (quad_summary["Athletes"] / max(len(chart_df), 1) * 100).round(1).astype(str) + "%"
            st.markdown("**Quadrant summary**")
            st.dataframe(quad_summary, use_container_width=True, hide_index=True)
        elif show_quadrants and not has_quadrant_benchmarks:
            st.info("Turn on both X and Y benchmark lines to split the chart into quadrants.")

        export_cols = [
            "athleteName", "Position", "pos_group", "Year", "Quadrant",
            "Concentric Impulse", "ci_pct_alltime",
            "P1 Concentric Impulse", "p1_ci_pct_alltime",
            "Concentric Impulse-100ms", "ci100_pct_alltime",
            "RSI-modified", "rsi_pct_alltime",
        ]
        export_df = chart_df[export_cols].rename(columns={
            "athleteName": "Athlete",
            "pos_group": "Position Group",
            "Concentric Impulse": "CI",
            "ci_pct_alltime": "CI Percentile",
            "P1 Concentric Impulse": "P1 CI",
            "p1_ci_pct_alltime": "P1 CI Percentile",
            "Concentric Impulse-100ms": "CI-100ms",
            "ci100_pct_alltime": "CI-100ms Percentile",
            "RSI-modified": "mRSI",
            "rsi_pct_alltime": "mRSI Percentile",
        }).copy()
        st.download_button(
            "Download filtered scatter data",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="force_plate_scatter_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =============================================================================
# TAB 2 — LEADERBOARD
# =============================================================================
with tab_board:

    # -------------------------------------------------------------------------
    # Leaderboard setup
    # -------------------------------------------------------------------------
    df_lb = df.copy()

    # Create display/filter columns in the units scouts actually want to see.
    df_lb["Height_in"] = pd.to_numeric(df_lb.get("Height"), errors="coerce") / 2.54
    df_lb["Bodyweight_lbs"] = pd.to_numeric(df_lb.get("Mass"), errors="coerce") * 2.20462
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
        # Physical attributes shown in the leaderboard
        ("Height_in", "Height", 1, " in", "Physical Attributes"),
        ("Bodyweight_lbs", "Bodyweight", 1, " lbs", "Physical Attributes"),
        ("Wingspan_in", "Wingspan", 1, " in", "Physical Attributes"),
        ("Wing_Adv_in", "Wingspan Adv.", 1, " in", "Physical Attributes"),
        ("BW_Ht_Pct", "BW/Ht Pct", 0, "th", "Physical Attributes"),
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
            "Capacity Score", "Pos. Capacity",
            "CI", "P1 Conc. Impulse", "CI-100ms", "RSI-modified", "Peak Power / BM",
            "Jump Height",
            "Height", "Bodyweight", "Wingspan", "Wingspan Adv.", "BW/Ht Pct",
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

        filter_groups = ["Force Plate", "Physical Attributes"]
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
        "Capacity Score": "athlete_quality_score",
        "Pos. Capacity": "aq_pos_score",
        "CI": "Concentric Impulse",
        "P1 Conc. Impulse": "P1 Concentric Impulse",
        "CI-100ms": "Concentric Impulse-100ms",
        "RSI-modified": "RSI-modified",
        "Peak Power / BM": "Peak Power / BM",
        "Jump Height": "Jump Height (Flight Time) in Inches",
        "Height": "Height_in",
        "Bodyweight": "Bodyweight_lbs",
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
        st.markdown(f'<p class="label">Force Plate 75th Percentiles</p>', unsafe_allow_html=True)
        fp_metrics = [m for m in LEADERBOARD_METRICS if m[4] == "Force Plate"]
        fp_cards = [
            ("Athletes", str(len(dff)))
        ]
        for col, label, digits, suffix, _ in fp_metrics:
            fp_cards.append((
                f"75th {label}",
                fmt(pd.to_numeric(dff[col], errors="coerce").quantile(0.75), digits, suffix),
            ))
        st.markdown(
            '<div class="median-row">' + ''.join(
                f'<div class="median-card"><div class="median-card-label">{label}</div>'
                f'<div class="median-card-value">{value}</div></div>'
                for label, value in fp_cards
            ) + '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f'<p class="label">Physical Attributes 75th Percentiles</p>', unsafe_allow_html=True)
        anthro_metrics = [
            m for m in LEADERBOARD_METRICS
            if m[4] == "Physical Attributes" and m[0] != "BW_Ht_Pct"
        ]
        anthro_cards = []
        for col, label, digits, suffix, _ in anthro_metrics:
            med_val = pd.to_numeric(dff[col], errors="coerce").quantile(0.75)
            anthro_cards.append((f"75th {label}", fmt(med_val, digits, suffix)))
        st.markdown(
            '<div class="median-row">' + ''.join(
                f'<div class="median-card"><div class="median-card-label">{label}</div>'
                f'<div class="median-card-value">{value}</div></div>'
                for label, value in anthro_cards
            ) + '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="card card-green" style="padding:14px 18px;margin-top:2px;margin-bottom:16px">'
            f'<p class="label" style="margin-bottom:6px">Benchmarks</p>'
            f'<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;color:{NAV}">'
            f'<span style="font-family:&quot;Playfair Display&quot;,serif;font-size:24px;font-weight:900;color:{GREEN}">285</span>'
            f'<span style="font-size:13px;color:{SLATE};margin-left:-10px">CI</span>'
            f'<span style="font-family:&quot;Playfair Display&quot;,serif;font-size:24px;font-weight:900;color:{GREEN}">195</span>'
            f'<span style="font-size:13px;color:{SLATE};margin-left:-10px">P1 CI</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        with st.expander("Cohort 75th percentiles: Pitchers vs Position Players", expanded=False):
            cohort_base = dff.copy()
            cohort_base["Cohort"] = np.where(cohort_base["pos_group"] == "Pitcher", "Pitcher", "Position Player")
            cohort_rows = []
            for cohort, sub in cohort_base.groupby("Cohort"):
                cohort_rows.append({
                    "Cohort": cohort,
                    "N": int(len(sub)),
                    "75th Capacity": round(pd.to_numeric(sub["athlete_quality_score"], errors="coerce").quantile(0.75), 1),
                    "75th CI": round(pd.to_numeric(sub["Concentric Impulse"], errors="coerce").quantile(0.75), 1),
                    "75th P1 CI": round(pd.to_numeric(sub["P1 Concentric Impulse"], errors="coerce").quantile(0.75), 1),
                    "75th RSI-mod": round(pd.to_numeric(sub["RSI-modified"], errors="coerce").quantile(0.75), 3),
                    "75th Bodyweight": round(pd.to_numeric(sub["Bodyweight_lbs"], errors="coerce").quantile(0.75), 1),
                })
            if cohort_rows:
                st.dataframe(pd.DataFrame(cohort_rows), use_container_width=True, hide_index=True, key="lb_cohort_75th")

        # -----------------------------------------------------------------
        # Leaderboard table: only text columns are Athlete and Position.
        # BW/Ht is shown as a percentile, not the raw pounds-per-inch value.
        # -----------------------------------------------------------------
        tbl_cols = [
            "athleteName", "Position", "Year",
            "athlete_quality_score", "aq_pos_score",
            "Concentric Impulse", "P1 Concentric Impulse", "Concentric Impulse-100ms",
            "RSI-modified", "Peak Power / BM", "Jump Height (Flight Time) in Inches",
            "Height_in", "Bodyweight_lbs", "Wingspan_in", "Wing_Adv_in", "BW_Ht_Pct",
        ]
        tbl = dff[tbl_cols].copy()
        tbl = tbl.rename(columns={
            "athleteName": "Athlete",
            "athlete_quality_score": "Capacity",
            "aq_pos_score": "Pos. Capacity",
            "Concentric Impulse": "CI",
            "P1 Concentric Impulse": "P1 Conc. Impulse",
            "Concentric Impulse-100ms": "CI-100ms",
            "Jump Height (Flight Time) in Inches": "Jump Height",
            "Height_in": "Height",
            "Bodyweight_lbs": "Bodyweight",
            "Wingspan_in": "Wingspan",
            "Wing_Adv_in": "Wingspan Adv.",
            "BW_Ht_Pct": "BW/Ht Pct",
        })

        round_map = {
            "Capacity": 1,
            "Pos. Capacity": 1,
            "CI": 1,
            "P1 Conc. Impulse": 1,
            "CI-100ms": 1,
            "RSI-modified": 3,
            "Peak Power / BM": 1,
            "Jump Height": 2,
            "Height": 1,
            "Bodyweight": 1,
            "Wingspan": 1,
            "Wingspan Adv.": 1,
            "BW/Ht Pct": 0,
        }
        for col, digits in round_map.items():
            if col in tbl.columns:
                tbl[col] = pd.to_numeric(tbl[col], errors="coerce").round(digits)

        st.caption(
            "Units: Height, Wingspan, and Wingspan Adv. are inches; "
            "Bodyweight is pounds; BW/Ht Pct in the table is the percentile rank of pounds per inch."
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
# TAB 3 — 2026 SCORECARDS
# =============================================================================
with tab_2026:
    df_2026 = df[(pd.to_numeric(df["Year"], errors="coerce") == 2026)].copy()
    if "Data Source" in df_2026.columns:
        fp2026_only = df_2026[df_2026["Data Source"].astype(str).eq("Force Plate 2026")].copy()
        if not fp2026_only.empty:
            df_2026 = fp2026_only

    st.markdown(f"""
    <div class="card card-red">
        <p class="label">Force Plate 2026</p>
        <h2 style="margin-top:0;color:{NAV};font-family:'Playfair Display',serif;">2026 Player Scorecards</h2>
        <p style="font-size:14px;line-height:1.55;color:{NAV};margin-bottom:0;">
            These cards are built from the <strong>Force Plate 2026</strong> tab. Height, School Type, and Position can come from that tab; missing wingspan, including cells marked X, is treated as a neutral 50th percentile input for Potential to Gain scoring only.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df_2026.empty:
        st.warning("No 2026 Force Plate rows are currently loaded.")
    else:
        st.caption(f"Loaded {len(df_2026)} 2026 scorecard rows.")

        sort_2026_options = {
            "Capacity": "athlete_quality_score",
            "Pos. Capacity": "aq_pos_score",
            "Potential": "potential_score",
            "CI": "Concentric Impulse",
            "P1 CI": "P1 Concentric Impulse",
            "mRSI": "RSI-modified",
            "Rel Peak Power": "Peak Power / BM",
            "Bodyweight": "Mass",
            "Height": "Height",
        }
        if "sort_2026_metric" not in st.session_state:
            st.session_state.sort_2026_metric = "Capacity"

        st.markdown(f'<p class="label">Tap a metric to sort the table and PDF cards</p>', unsafe_allow_html=True)
        sort_cols = st.columns(len(sort_2026_options))
        for j, metric_label in enumerate(sort_2026_options.keys()):
            active = metric_label == st.session_state.sort_2026_metric
            btn_label = f"✓ {metric_label}" if active else metric_label
            if sort_cols[j].button(btn_label, key=f"sort_2026_{metric_label}", use_container_width=True):
                st.session_state.sort_2026_metric = metric_label

        sort_metric_label = st.session_state.sort_2026_metric
        sort_col_2026 = sort_2026_options.get(sort_metric_label, "athlete_quality_score")
        if sort_col_2026 not in df_2026.columns:
            df_2026[sort_col_2026] = np.nan
        df_2026 = df_2026.sort_values([sort_col_2026, "athleteName"], ascending=[False, True], na_position="last")

        def _historical_sort_values(metric_col):
            """Reference pool for 2026 card color/marker.

            The 2026 cards stay filtered to 2026, but the gradient position is
            calculated against the full loaded historical dataset so the card
            colors match the all-time percentile logic used in the scorecards/PDFs.
            """
            if metric_col not in df.columns:
                return pd.Series(dtype="float64")
            vals = pd.to_numeric(df[metric_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            return vals

        def _sort_percentile(value, values, inverse=False):
            vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            v = sf(value)
            if pd.isna(v) or vals.empty:
                return np.nan
            pct = float((vals <= v).mean())
            return 1.0 - pct if inverse else pct

        def _green_to_red(value, values, inverse=False):
            pct = _sort_percentile(value, values, inverse=inverse)
            if pd.isna(pct):
                return "#9AAAC0"
            # Low historical percentile = red, high historical percentile = green.
            red_rgb = (186, 12, 47)
            green_rgb = (76, 175, 130)
            rgb = tuple(int(red_rgb[k] + (green_rgb[k] - red_rgb[k]) * pct) for k in range(3))
            return "#%02x%02x%02x" % rgb

        # Use all historical rows as the reference for the card gradient/marker.
        # The displayed cards are still only 2026.
        sort_values_historical = _historical_sort_values(sort_col_2026)

        # If you add sprint splits later, lower time should be better/greener.
        inverse_sort_metric = sort_col_2026 in {"10yd Split", "20yd Split", "30yd Split"}

        summary_cols = [
            "athleteName", "Position", "School Type", "athlete_quality_score", "aq_pos_score",
            "potential_score", "Concentric Impulse", "P1 Concentric Impulse", "RSI-modified",
            "Peak Power / BM", "Mass", "Height"
        ]
        for c in summary_cols:
            if c not in df_2026.columns:
                df_2026[c] = np.nan
        summary = df_2026[summary_cols].copy()
        summary["Height"] = pd.to_numeric(summary["Height"], errors="coerce") / 2.54
        summary["Mass"] = pd.to_numeric(summary["Mass"], errors="coerce") * 2.20462
        summary = summary.rename(columns={
            "athleteName": "Athlete",
            "athlete_quality_score": "Capacity",
            "aq_pos_score": "Pos. Capacity",
            "potential_score": "Potential",
            "Concentric Impulse": "CI",
            "P1 Concentric Impulse": "P1 CI",
            "RSI-modified": "mRSI",
            "Peak Power / BM": "Rel Peak Power",
            "Mass": "Bodyweight",
            "Height": "Height"
        })
        for c in ["Capacity", "Pos. Capacity", "Potential", "CI", "P1 CI", "Rel Peak Power", "Bodyweight", "Height"]:
            if c in summary.columns:
                summary[c] = pd.to_numeric(summary[c], errors="coerce").round(1)
        if "mRSI" in summary.columns:
            summary["mRSI"] = pd.to_numeric(summary["mRSI"], errors="coerce").round(3)

        numeric_summary_cols = [c for c in ["Capacity", "Pos. Capacity", "Potential", "CI", "P1 CI", "mRSI", "Rel Peak Power", "Bodyweight", "Height"] if c in summary.columns]
        try:
            styled_summary = summary.style.background_gradient(subset=numeric_summary_cols, cmap="RdYlGn")
            st.dataframe(styled_summary, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("### Scorecard PDFs")
        st.caption(f"PDF cards are sorted by {sort_metric_label}; the larger color bar runs red-to-green based on that metric's historical percentile across all loaded years.")

        card_cols = st.columns(3)
        for i, (_, r) in enumerate(df_2026.iterrows()):
            with card_cols[i % 3]:
                name = str(r.get("athleteName", "Athlete"))
                cap = sf(r.get("athlete_quality_score"))
                pot = sf(r.get("potential_score"))
                ci = sf(r.get("Concentric Impulse"))
                p1 = sf(r.get("P1 Concentric Impulse"))
                school = str(r.get("School Type", "—"))
                if school in ("nan", "None", ""):
                    school = "—"
                card_color = _green_to_red(r.get(sort_col_2026), sort_values_historical, inverse=inverse_sort_metric)
                sort_pct = _sort_percentile(r.get(sort_col_2026), sort_values_historical, inverse=inverse_sort_metric)
                marker_left = 50 if pd.isna(sort_pct) else max(0, min(100, sort_pct * 100))
                sort_display = fmt(sf(r.get(sort_col_2026)), 1)
                st.markdown(f"""
                <div style="background:white;border:1px solid {BORD};border-top:7px solid {card_color};
                    border-radius:10px;padding:14px 16px;margin-bottom:10px;box-shadow:0 2px 8px rgba(17,34,90,0.06)">
                    <div style="position:relative;height:18px;background:linear-gradient(90deg,{RED} 0%,{GOLD} 50%,{GREEN} 100%);border-radius:999px;margin:0 0 12px 0;border:1px solid rgba(17,34,90,0.18);box-shadow:inset 0 1px 2px rgba(0,0,0,0.18)">
                        <div style="position:absolute;left:calc({marker_left:.1f}% - 6px);top:-5px;width:12px;height:28px;background:{card_color};border:2px solid white;border-radius:999px;box-shadow:0 1px 5px rgba(17,34,90,0.35)"></div>
                    </div>
                    <div style="font-family:'Playfair Display',serif;font-size:20px;font-weight:900;color:{NAV};line-height:1.1;margin-bottom:6px">{name}</div>
                    <div style="font-size:11px;color:{SLATE};margin-bottom:4px">2026 · {school}</div>
                    <div style="font-size:10px;color:{SLATE};margin-bottom:10px">Sorted by <strong style="color:{card_color}">{sort_metric_label}: {sort_display}</strong></div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px;color:{NAV}">
                        <div><span style="color:{SLATE}">Capacity</span><br><strong>{fmt(cap,0)}</strong></div>
                        <div><span style="color:{SLATE}">Potential</span><br><strong>{fmt(pot,0)}</strong></div>
                        <div><span style="color:{SLATE}">CI</span><br><strong>{fmt(ci,1)}</strong></div>
                        <div><span style="color:{SLATE}">P1 CI</span><br><strong>{fmt(p1,1)}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                try:
                    is_pitcher_2026 = str(r.get("pos_group", "")).strip() == "Pitcher"
                    pdf_bytes, pdf_name = make_scorecard_pdf(r, df, strat_feats, "2026", is_pitcher=is_pitcher_2026)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"{pdf_name}_2026_scorecard.pdf",
                        mime="application/pdf",
                        key=f"download_2026_scorecard_{i}_{pdf_name}",
                        use_container_width=True,
                    )
                except Exception as pdf_err:
                    st.warning(f"PDF unavailable: {pdf_err}")

# =============================================================================
# TAB 3 — ATHLETE SCORECARD
# =============================================================================
if tab_card is not None:
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
            # Default to the most recent year instead of an all-years average.
            # This keeps the dashboard/PDF from showing ranges like "2021–2024" unless intentionally selected.
            yr_opts2   = [str(y) for y in ath_years] + (["All years"] if len(ath_years) > 1 else [])
            sel_yr_str = st.selectbox("Year", yr_opts2, key="sc_yr")
            sel_yr     = None if sel_yr_str == "All years" else int(sel_yr_str)

        ath_all = df[df["athleteName"] == sel_ath].sort_values("Year")
        if sel_yr is None:
            row_data = ath_all.select_dtypes(include=[np.number]).mean()
            latest   = ath_all.iloc[-1]
            for c in ath_all.columns:
                if c not in row_data.index: row_data[c] = latest[c]
            row            = row_data
            sel_yr_display = f"All years ({ath_years[-1]}–{ath_years[0]})" if len(ath_years)>1 else str(ath_years[0])
        else:
            sub = ath_all[ath_all["Year"]==sel_yr]
            if sub.empty: st.warning("No data for this year."); st.stop()
            row            = sub.iloc[0]
            sel_yr_display = str(sel_yr)

        # ── Core values ───────────────────────────────────────────────────────────
        aq_val  = sf(row.get("athlete_quality_score"))
        pos_val = sf(row.get("aq_pos_score"))
        pot_val = sf(row.get("potential_score"))
        ci_val  = sf(row.get("Concentric Impulse"))
        mass_kg = sf(row.get("Mass"))
        ht_cm   = sf(row.get("Height"))

        # Wingspan values
        wing_cm     = sf(row.get("Wingspan"))
        wing_adv_cm = sf(row.get("wingspan_advantage"))
        wing_pct    = sf(row.get("wingspan_pct"))
        wing_pct_str= pct_sfx(int(round(wing_pct))) if pd.notna(wing_pct) else "—"
        wing_adv_in = fmt_wingspan_adv(wing_adv_cm)

        # Wingspan/reach tier is based on the athlete's wingspan-advantage percentile.
        # This avoids broad labels like calling a 70th percentile reach "Average".
        if pd.notna(wing_pct):
            if wing_pct >= 85:
                wing_tier_label = "Notable Reach Advantage"
                wing_tier_color = GREEN
                wing_adv_css    = "wing-adv-pos"
            elif wing_pct >= 60:
                wing_tier_label = "Above-Average Reach"
                wing_tier_color = GREEN
                wing_adv_css    = "wing-adv-pos"
            elif wing_pct >= 40:
                wing_tier_label = "Average Reach"
                wing_tier_color = GOLD
                wing_adv_css    = "wing-adv-neu"
            elif wing_pct >= 20:
                wing_tier_label = "Below-Average Reach"
                wing_tier_color = GOLD
                wing_adv_css    = "wing-adv-neu"
            else:
                wing_tier_label = "Limited Reach"
                wing_tier_color = RED
                wing_adv_css    = "wing-adv-neg"
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
        athlete_group_val = str(row.get("athlete_group", athlete_group_label(prog_cat)))
        program_focus_val = str(row.get("program_focus", program_focus_label(prog_cat)))
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
                        border-radius:20px;letter-spacing:0.06em">{athlete_group_val}</span>
                    <span style="font-size:11px;color:{SLATE};margin-left:8px">Program Focus: {program_focus_val}</span>
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

        # ── Hero row: Capacity | Pos. gauge | Athlete Group | Potential to Gain ──
        h1, h2, h3, h4 = st.columns([1,1,1.25,1.05])

        with h1:
            bar_w  = min(100, max(0, float(aq_val or 0)))
            bar_cl = GREEN if pd.notna(aq_val) and aq_val>=75 else GOLD if pd.notna(aq_val) and aq_val>=50 else RED
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORD};border-top:6px solid {RED};
                border-radius:12px;padding:24px 20px;text-align:center;
                box-shadow:0 4px 16px rgba(186,12,47,0.12)">
                <div style="font-size:10px;font-weight:700;letter-spacing:0.18em;
                    text-transform:uppercase;color:{RED};margin-bottom:8px">★ CAPACITY SCORE</div>
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
                pos_lbl = f"Capacity vs {_pg}s"
            elif _pos and _pos not in ("nan","None",""):
                pos_lbl = f"Capacity vs {_pos}s"
            else:
                pos_lbl = "Position Data Unavailable"
            st.plotly_chart(
                make_gauge(pos_val if (pd.notna(pos_val) and pos_val>0) else None, pos_lbl, SLATE),
                use_container_width=True, key="g_pos")

        with h3:
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORD};border-top:6px solid {prog_color};
                border-radius:12px;padding:22px 18px;text-align:left;height:100%;
                box-shadow:0 4px 16px rgba(17,34,90,0.08)">
                <div style="font-size:10px;font-weight:700;letter-spacing:0.18em;
                    text-transform:uppercase;color:{SLATE};margin-bottom:8px">ATHLETE GROUP</div>
                <div style="font-family:'Playfair Display',serif;font-size:clamp(18px,1.6vw,24px);
                    font-weight:900;color:{NAV};line-height:1.15;margin-bottom:10px;white-space:normal;overflow-wrap:normal">
                    {athlete_group_val}</div>
                <div style="font-size:11px;color:{SLATE};line-height:1.45">
                    CI: <strong style="color:{NAV}">{fmt(ci_val,1)}</strong><br>
                    P1 CI: <strong style="color:{NAV}">{fmt(sf(row.get('P1 Concentric Impulse')),1)}</strong><br>
                    Focus:<br><strong style="color:{prog_color};font-size:12px;line-height:1.25">{program_focus_val}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with h4:
            pot_w = min(100, max(0, float(pot_val or 0)))
            pot_cl = GREEN if pd.notna(pot_val) and pot_val>=75 else GOLD if pd.notna(pot_val) and pot_val>=50 else RED
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORD};border-top:6px solid {GOLD};
                border-radius:12px;padding:22px 18px;text-align:center;height:100%;
                box-shadow:0 4px 16px rgba(226,193,136,0.16)">
                <div style="font-size:10px;font-weight:700;letter-spacing:0.18em;
                    text-transform:uppercase;color:{SLATE};margin-bottom:8px">POTENTIAL TO GAIN</div>
                <div style="font-family:'Playfair Display',serif;font-size:54px;
                    font-weight:900;color:{GOLD};line-height:1">
                    {str(int(round(pot_val))) if pd.notna(pot_val) else "—"}</div>
                <div style="font-size:11px;color:{SLATE};margin-top:6px;line-height:1.35">
                    Frame/projectability score<br>higher = more room to add capacity
                </div>
                <div style="margin-top:12px;background:#F0F3F8;border-radius:6px;height:8px">
                    <div style="width:{pot_w:.0f}%;background:{pot_cl};border-radius:6px;height:8px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Score legend ──────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 16px 0">
            <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
                border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
                <span style="width:10px;height:10px;border-radius:50%;background:{GREEN};display:inline-block"></span>
                <strong>75–100</strong>&nbsp;High-end present capacity
            </div>
            <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
                border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
                <span style="width:10px;height:10px;border-radius:50%;background:{GOLD};display:inline-block"></span>
                <strong>50–74</strong>&nbsp;Above-average present capacity
            </div>
            <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
                border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
                <span style="width:10px;height:10px;border-radius:50%;background:#D0D7E6;display:inline-block"></span>
                <strong>25–49</strong>&nbsp;Below-average present capacity
            </div>
            <div style="display:flex;align-items:center;gap:6px;background:white;border:1px solid {BORD};
                border-radius:20px;padding:5px 14px;font-size:11px;color:{NAV}">
                <span style="width:10px;height:10px;border-radius:50%;background:#9AAAC0;display:inline-block"></span>
                <strong>0–24</strong>&nbsp;Well below-average present capacity
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Capacity = present physical capacity relative to the loaded dataset. Potential to Gain = projectability/frame score; 50 is roughly middle-of-pool, not a draft grade.")

        # ── Pitcher flag ─────────────────────────────────────────────────────────
        is_pitcher = str(row.get("pos_group","")).strip() == "Pitcher"

        # ── PDF download ──────────────────────────────────────────────────────────
        try:
            pdf_bytes, pdf_name = make_scorecard_pdf(row, df, strat_feats, sel_yr_display, is_pitcher=is_pitcher)
            st.download_button(
                label="Download one-page PDF scorecard",
                data=pdf_bytes,
                file_name=f"{pdf_name}_scorecard.pdf",
                mime="application/pdf",
                use_container_width=False,
                key="download_scorecard_pdf",
            )
        except Exception as pdf_err:
            st.warning(f"PDF export is unavailable for this athlete: {pdf_err}")

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

        # ── Full-width percentile profile ─────────────────────────────────────────
        st.plotly_chart(make_radar(row, is_pitcher=is_pitcher),
                        use_container_width=True, key="g_profile_bars_full")

        # ── Development Projection table data ─────────────────────────────────────
        # Project the CI range at the 60th-percentile BW/Ht target from the full
        # loaded cohort. Height is held constant. The range covers a practical
        # response spectrum: a 1% CI/kg improvement through a 4% CI/kg penalty.
        # Capacity Score is recalculated with the exact same score weights used by
        # the app, changing only the athlete's CI percentile while holding sprint,
        # RSI-modified, and relative peak power constant. Athletes already at or
        # above the 60th percentile are not projected to lose weight.
        proj_df = pd.DataFrame()
        if (pd.notna(ci_val) and pd.notna(mass_kg) and pd.notna(ht_cm)
                and ci_val > 0 and mass_kg > 0 and ht_cm > 0):
            ci_per_kg = ci_val / mass_kg
            BWHT_TARGET_PCT = 60

            def sc_bwht(kg, cm):
                if pd.isna(kg) or pd.isna(cm) or cm == 0:
                    return np.nan
                return (kg * 2.20462) / (cm / 2.54)

            # Uses the same all-loaded-data reference pool as the BW/Ht percentile
            # displayed elsewhere on the scorecard.
            pool_r = pd.to_numeric(df["bmi_raw"], errors="coerce").dropna()
            target_bwht = (
                float(np.nanpercentile(pool_r, BWHT_TARGET_PCT))
                if len(pool_r) else np.nan
            )

            def sc_ci_pct(val):
                pool_c = pd.to_numeric(df["Concentric Impulse"], errors="coerce").dropna()
                if pd.isna(val) or len(pool_c) == 0:
                    return np.nan
                return float((pool_c < val).mean() * 100)

            def sc_capacity_score(projected_ci_pct):
                """Recalculate Capacity Score with only the CI percentile changing."""
                sprint_pct = sf(row.get("sprint_pct_alltime"))
                rsi_pct = sf(row.get("rsi_pct_alltime"))
                pp_pct = sf(row.get("pp_pct_alltime"))

                if pd.notna(sprint_pct):
                    components = [
                        (projected_ci_pct, w_ci),
                        (sprint_pct, w_sprint),
                        (rsi_pct, w_rsi),
                        (pp_pct, w_pp),
                    ]
                else:
                    components = [
                        (projected_ci_pct, w_ci_ns),
                        (rsi_pct, w_rsi_ns),
                        (pp_pct, w_pp_ns),
                    ]

                # This mirrors the app's Capacity Score structure. In the unlikely
                # event a component is missing, retain the relative weight of the
                # available components rather than turning the projection blank.
                valid = [(v, wt) for v, wt in components if pd.notna(v)]
                if not valid:
                    return np.nan
                total_weight = sum(wt for _, wt in valid)
                return sum(v * wt for v, wt in valid) / total_weight

            current_bwht = sc_bwht(mass_kg, ht_cm)
            # BW/Ht is pounds per inch, so target bodyweight in pounds is the
            # target ratio multiplied by the athlete's fixed height in inches.
            target_lbs_raw = target_bwht * (ht_cm / 2.54) if pd.notna(target_bwht) else np.nan
            target_kg_raw = target_lbs_raw / 2.20462 if pd.notna(target_lbs_raw) else np.nan

            at_or_above_target = (
                pd.notna(current_bwht)
                and pd.notna(target_bwht)
                and current_bwht >= target_bwht
            )
            target_kg = mass_kg if at_or_above_target else target_kg_raw
            target_label = (
                f"At / Above {BWHT_TARGET_PCT}th BW/Ht"
                if at_or_above_target
                else f"{BWHT_TARGET_PCT}th BW/Ht Target"
            )

            # The low end of the projection assumes a 4% CI/kg penalty; the high
            # end assumes a 1% CI/kg improvement. Individual scenario rows are
            # intentionally not displayed so the scorecard communicates a usable
            # range instead of false precision.
            ci_rel_changes = [0.01, 0.00, -0.01, -0.02, -0.03, -0.04]
            current_ci_pct = sc_ci_pct(ci_val)
            current_capacity = sf(row.get("athlete_quality_score"))

            rows_proj = [{
                "Scenario": "Current",
                "CI": f"{ci_val:.1f}",
                "Capacity Score": f"{current_capacity:.1f}" if pd.notna(current_capacity) else "—",
            }]

            if at_or_above_target:
                rows_proj.append({
                    "Scenario": target_label,
                    "CI": f"{ci_val:.1f}",
                    "Capacity Score": (
                        f"{current_capacity:.1f} (no change)"
                        if pd.notna(current_capacity) else "—"
                    ),
                })
                projection_note = (
                    f"Already at or above the all-pool {BWHT_TARGET_PCT}th-percentile BW/Ht target; "
                    "no bodyweight change is projected."
                )
            else:
                projected_ci_values = [
                    ci_per_kg * (1 + rel_change) * target_kg
                    for rel_change in ci_rel_changes
                ]
                projected_capacity_values = [
                    sc_capacity_score(sc_ci_pct(projected_ci))
                    for projected_ci in projected_ci_values
                ]

                # Only communicate upside. Some relative-CI penalty assumptions can
                # produce an absolute CI or Capacity Score below the athlete's current
                # level even after reaching the target BW/Ht. Those outcomes are not
                # useful as a development projection, so exclude them from the range.
                upside_pairs = []
                for projected_ci, projected_capacity in zip(
                    projected_ci_values, projected_capacity_values
                ):
                    ci_improves = pd.notna(projected_ci) and projected_ci > ci_val
                    capacity_not_worse = (
                        pd.isna(current_capacity)
                        or (
                            pd.notna(projected_capacity)
                            and projected_capacity >= current_capacity
                        )
                    )
                    if ci_improves and capacity_not_worse:
                        upside_pairs.append((projected_ci, projected_capacity))

                if upside_pairs:
                    upside_ci_values = [ci for ci, _ in upside_pairs]
                    upside_capacity_values = [cap for _, cap in upside_pairs if pd.notna(cap)]
                    ci_low, ci_high = min(upside_ci_values), max(upside_ci_values)

                    if upside_capacity_values:
                        capacity_low = min(upside_capacity_values)
                        capacity_high = max(upside_capacity_values)
                        if pd.notna(current_capacity):
                            capacity_range = (
                                f"{capacity_low:.1f}–{capacity_high:.1f} "
                                f"({capacity_low - current_capacity:+.1f} to "
                                f"{capacity_high - current_capacity:+.1f})"
                            )
                        else:
                            capacity_range = f"{capacity_low:.1f}–{capacity_high:.1f}"
                    else:
                        capacity_range = "—"

                    rows_proj.append({
                        "Scenario": f"{target_label} upside ({target_kg * 2.20462:.1f} lb)",
                        "CI": f"{ci_low:.1f}–{ci_high:.1f}",
                        "Capacity Score": capacity_range,
                    })
                    projection_note = (
                        f"Upside-only range at the all-pool {BWHT_TARGET_PCT}th-percentile BW/Ht target. "
                        "It includes only modeled outcomes that raise CI without reducing Capacity Score."
                    )
                else:
                    rows_proj.append({
                        "Scenario": f"{target_label} upside ({target_kg * 2.20462:.1f} lb)",
                        "CI": "No modeled improvement",
                        "Capacity Score": "No modeled improvement",
                    })
                    projection_note = (
                        f"None of the modeled CI/kg assumptions produced both a higher CI and a non-decreasing "
                        f"Capacity Score at the all-pool {BWHT_TARGET_PCT}th-percentile BW/Ht target."
                    )

            proj_df = pd.DataFrame(rows_proj)

        # ── Main body: metrics | wingspan + development projection ───────────────
        m1, m2 = st.columns([1.05, 1.95])

        with m1:
            # Wingspan row in physical attributes — always show raw numbers, but
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
                f'<span style="color:{SLATE};min-width:160px;font-size:12px">Bodyweight</span>'
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

            if not proj_df.empty:
                st.markdown(
                    f'<div style="background:white;border:1px solid {BORD};border-top:4px solid {GREEN};'
                    f'border-radius:10px;padding:14px 16px;margin-top:12px;'
                    f'box-shadow:0 2px 8px rgba(17,34,90,0.06)">'
                    f'<div style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
                    f'color:{GREEN};margin-bottom:4px">Development Projection</div>'
                    f'<div style="font-size:11px;color:{SLATE};margin-bottom:8px">'
                    f'{projection_note}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
                st.dataframe(proj_df, use_container_width=True, hide_index=True, key="sc_proj_tbl_side")

            # Year-over-year trends and jump strategy profile removed to reduce clutter.

        # Development Projection lives in the right-hand scorecard column to reduce white space.
