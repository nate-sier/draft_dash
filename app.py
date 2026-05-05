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
from sklearn.cluster import KMeans

# ─── Config ───────────────────────────────────────────────────────────────────
SHEET_ID   = "1RKyeb4CfU4wACUKqpbiR_--PJJ9KKO-LH-Ov9e6zNQs"
RANDOM_STATE = 42
MIN_GROUP_SIZE = 6

PITCHER_POSITIONS = {"P", "SP", "RP", "Starting Pitcher", "Relief Pitcher",
                     "Right Hand Pitcher", "Left Hand Pitcher", "SC", "PC",
                     "Starting pitcher", "starting pitcher"}

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
def load_data():
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
@st.cache_data(show_spinner="Building scores…")
def build_scores(_df,
                 w_ci=0.40, w_sprint=0.40, w_rsi=0.20,
                 wp_peakpow=0.25, wp_height=0.25, wp_bmi=0.20,
                 wp_school=0.15, wp_wingspan=0.15):
    df = _df.copy()

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

    # Strategy distance (Mahalanobis, all-time pool)
    X_s = df[strategy_features].fillna(0).values
    scaler_s = StandardScaler().fit(X_s)
    X_sc = scaler_s.transform(X_s)
    ncomp = min(len(strategy_features), max(3, min(6, len(df) - 1)))
    pca   = PCA(n_components=ncomp, whiten=True, random_state=RANDOM_STATE).fit(X_sc)
    X_pca = pca.transform(X_sc)
    try:
        mcd = MinCovDet(random_state=RANDOM_STATE, support_fraction=0.8).fit(X_pca)
        df["strategy_distance_raw"] = np.sqrt(mcd.mahalanobis(X_pca))
    except Exception:
        df["strategy_distance_raw"] = np.linalg.norm(X_pca, axis=1)
    df["strategy_distance_score"] = pct_rank(df["strategy_distance_raw"])

    # Archetypes
    df["archetype"]  = df.apply(lambda r: label_archetype(r.to_dict(), all_rz_cols), axis=1)
    df["why_flagged"] = df.apply(lambda r: explain_strategy(
        {f"rz_{f}": r.get(f"rz_{f}", 0) for f in strategy_features}, strategy_features), axis=1)

    # ── Athlete Quality Score ──────────────────────────────────────────────────
    # CI percentile (higher = better)
    df["ci_pct_alltime"]     = pct_rank(df["Concentric Impulse"])
    # RSI percentile (higher = better)
    df["rsi_pct_alltime"]    = pct_rank(df["RSI-modified"])
    # Sprint percentile (lower time = better → invert)
    df["sprint_pct_alltime"] = 100 - pct_rank(df["30yd Split"])

    df["athlete_quality_raw"] = (
        w_ci     * df["ci_pct_alltime"].fillna(50) +
        w_sprint * df["sprint_pct_alltime"].fillna(50) +
        w_rsi    * df["rsi_pct_alltime"].fillna(50)
    )
    df["athlete_quality_score"] = scaled_0_100(df["athlete_quality_raw"].values)

    # Per-year versions
    for col, pct_col, invert in [
        ("Concentric Impulse",  "ci_pct_yr",     False),
        ("RSI-modified",        "rsi_pct_yr",     False),
        ("30yd Split",          "sprint_pct_yr",  True),
    ]:
        df[pct_col] = np.nan
        for yr, idx in df.groupby("Year").groups.items():
            s = df.loc[idx, col]
            r = pct_rank(s)
            df.loc[idx, pct_col] = 100 - r if invert else r

    df["aq_yr_raw"] = (
        w_ci     * df["ci_pct_yr"].fillna(50) +
        w_sprint * df["sprint_pct_yr"].fillna(50) +
        w_rsi    * df["rsi_pct_yr"].fillna(50)
    )
    # Scale per year
    df["aq_score_yr"] = np.nan
    for yr, idx in df.groupby("Year").groups.items():
        df.loc[idx, "aq_score_yr"] = scaled_0_100(df.loc[idx, "aq_yr_raw"].values)

    # ── Potential Score ────────────────────────────────────────────────────────
    # Peak Power / BM percentile (higher = better)
    df["pp_pct"]      = pct_rank(df["Peak Power / BM"])
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
    df["overall_rank"] = df["athlete_quality_score"].rank(ascending=False, method="min").astype(int)

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
            "threshold": {"line": {"color": GOLD, "width": 3},
                          "thickness": 0.8, "value": 75},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="white", font=dict(family="Arial"))
    return fig

# ─── Radar chart ─────────────────────────────────────────────────────────────
def make_radar(row, label="Athlete"):
    cats = ["CI", "Sprint", "RSI-mod", "Peak Pwr", "Height"]
    vals = [
        row.get("ci_pct_alltime", 50) or 50,
        row.get("sprint_pct_alltime", 50) or 50,
        row.get("rsi_pct_alltime", 50) or 50,
        row.get("pp_pct", 50) or 50,
        row.get("height_pct", 50) or 50,
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
                f'text-transform:uppercase;color:#6b7fa3">Athlete Quality Weights</p>',
                unsafe_allow_html=True)
    w_ci     = st.slider("Concentric Impulse",  0, 100, 40, 5, key="w_ci") / 100
    w_sprint = st.slider("30yd Sprint",          0, 100, 40, 5, key="w_sprint") / 100
    w_rsi    = st.slider("RSI-modified",         0, 100, 20, 5, key="w_rsi") / 100
    total_aq = w_ci + w_sprint + w_rsi
    if not (0.99 < total_aq < 1.01):
        st.warning(f"Weights sum to {total_aq*100:.0f}% — should be 100%")

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
    raw = load_data()
    load_err = None
except Exception as e:
    raw = None; load_err = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"

if load_err:
    st.error("Could not load data.")
    st.code(load_err); st.stop()

df, strat_feats, all_rz_cols = build_scores(
    raw, w_ci, w_sprint, w_rsi, wp_pp, wp_ht, wp_bmi, wp_school, wp_wings)

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
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1.2, 1.5, 1.5, 1.2])
    with fc1: search  = st.text_input("Search athlete", placeholder="Name…", key="lb_search")
    with fc2:
        yr_opts = ["All"] + sorted(df["Year"].dropna().unique().astype(int).tolist(), reverse=True)
        yr_sel  = st.selectbox("Year", yr_opts, key="lb_year")
    with fc3:
        arch_opts = ["All"] + sorted(df["archetype"].dropna().unique())
        arch_sel  = st.selectbox("Archetype", arch_opts, key="lb_arch")
    with fc4:
        st_opts = ["All"] + sorted(df["School Type"].dropna().unique())
        st_sel  = st.selectbox("School Type", st_opts, key="lb_st")
    with fc5:
        sort_by = st.selectbox("Sort by", ["Athlete Quality", "Potential", "CI", "30yd Sprint"],
                               key="lb_sort")

    dff = df.copy()
    if search:       dff = dff[dff["athleteName"].str.contains(search, case=False, na=False)]
    if yr_sel != "All":  dff = dff[dff["Year"] == int(yr_sel)]
    if arch_sel != "All": dff = dff[dff["archetype"] == arch_sel]
    if st_sel != "All":   dff = dff[dff["School Type"] == st_sel]

    sort_col = {"Athlete Quality": "athlete_quality_score", "Potential": "potential_score",
                "CI": "Concentric Impulse", "30yd Sprint": "30yd Split"}[sort_by]
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

    tbl = dff[["athleteName","Year","School Type","archetype",
               "athlete_quality_score","potential_score",
               "Concentric Impulse","30yd Split","RSI-modified","Peak Power / BM"]].copy()
    tbl = tbl.rename(columns={
        "athleteName": "Athlete", "School Type": "School",
        "archetype": "Archetype", "athlete_quality_score": "Quality",
        "potential_score": "Potential", "Concentric Impulse": "CI",
        "30yd Split": "30yd (s)", "RSI-modified": "RSI-mod",
        "Peak Power / BM": "PkPwr/BM",
    })
    for c in ["Quality","Potential","CI","RSI-mod","PkPwr/BM"]:
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

    sc1, sc2 = st.columns([2, 1])
    with sc1:
        sel_ath = st.selectbox("Select athlete", athletes,
                               index=athletes.index(default_ath) if default_ath in athletes else 0,
                               key="sc_ath")
    with sc2:
        ath_years = sorted(df[df["athleteName"] == sel_ath]["Year"].dropna().unique().astype(int).tolist(),
                           reverse=True)
        sel_yr = st.selectbox("Year", ath_years, key="sc_yr")

    ath_all = df[df["athleteName"] == sel_ath].sort_values("Year")
    row = ath_all[ath_all["Year"] == sel_yr]
    if row.empty:
        st.warning("No data for this athlete/year combination.")
        st.stop()
    row = row.iloc[0]

    arch_color = ARCHETYPE_COLORS.get(row.get("archetype","Unclassified"), "#9AAAC0")

    # ── Header banner ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{NAV};border-radius:10px;padding:20px 28px;margin-bottom:20px;
        border-left:6px solid {RED};position:relative;overflow:hidden">
        <div style="position:absolute;top:0;left:0;right:0;height:4px;background:{GOLD}"></div>
        <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px">
            <div>
                <p style="font-size:9px;font-weight:700;letter-spacing:0.2em;color:#9AAAC0;margin:0 0 4px 0">
                    WASHINGTON NATIONALS · ATHLETE SCORECARD</p>
                <h2 style="font-family:'Playfair Display',serif;font-size:28px;color:white;margin:0 0 8px 0">
                    {sel_ath}</h2>
                <span class="arch-badge" style="background:{arch_color}">{row.get('archetype','—')}</span>
                <span style="font-size:12px;color:#9AAAC0;margin-left:10px">
                    {sel_yr} · {row.get('School Type','—')}</span>
            </div>
            <div style="text-align:right">
                <p style="font-size:9px;font-weight:700;letter-spacing:0.12em;color:{GOLD};margin:0">
                    OVERALL RANK</p>
                <p style="font-family:'Playfair Display',serif;font-size:36px;font-weight:900;
                    color:white;margin:0">#{int(row.get('overall_rank', 0))}</p>
                <p style="font-size:11px;color:#9AAAC0;margin:0">of {len(df[df['Year']==sel_yr])} in {sel_yr}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score gauges ──────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_gauge(row.get("athlete_quality_score"), "Athlete Quality", RED),
                        use_container_width=True, key="g_aq")
    with g2:
        st.plotly_chart(make_gauge(row.get("potential_score"), "Development Potential", NAV),
                        use_container_width=True, key="g_pot")
    with g3:
        st.plotly_chart(make_radar(row), use_container_width=True, key="g_radar")

    # ── Percentile cards ──────────────────────────────────────────────────────
    st.markdown('<p class="label" style="margin-top:4px">Percentiles</p>', unsafe_allow_html=True)
    pc = st.columns(5)
    pct_items = [
        ("CI",        "ci_pct_alltime",     "ci_pct_yr",      False),
        ("30yd Sprint","sprint_pct_alltime", "sprint_pct_yr",  True),
        ("RSI-mod",   "rsi_pct_alltime",    "rsi_pct_yr",     False),
        ("Pk Pwr/BM", "pp_pct",             None,             False),
        ("Height",    "height_pct",         None,             False),
    ]
    for col, pct_all_col, pct_yr_col, inv in pct_items:
        p_all = row.get(pct_all_col, np.nan)
        p_yr  = row.get(pct_yr_col, np.nan) if pct_yr_col else np.nan
        with pc[pct_items.index((col, pct_all_col, pct_yr_col, inv))]:
            st.markdown(f"""
            <div class="card card-red" style="text-align:center;padding:14px 10px">
                <div class="label">{col}</div>
                <div class="score-big" style="color:{RED};font-size:30px">{fmt(p_all, 0)}</div>
                <div style="font-size:10px;color:#9AAAC0;margin-top:2px">All-time pct</div>
                <div style="font-size:13px;font-weight:600;color:{NAV};margin-top:4px">
                    {fmt(p_yr, 0) if pd.notna(p_yr) else '—'}</div>
                <div style="font-size:10px;color:#9AAAC0">{sel_yr} pct</div>
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
