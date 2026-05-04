import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import gspread
import json
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Draft Scout · Nationals", layout="wide", page_icon="⚾")

NAV  = "#11225A"
RED  = "#BA0C2F"
GOLD = "#E2C188"
SURF = "#F7F8FA"
BORD = "#E8ECF0"
GRN  = "#4CAF82"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Source Sans 3', sans-serif; background: white; color: {NAV}; }}
.block-container {{ padding: 1.8rem 2.5rem; max-width: 1500px; }}
h1,h2,h3 {{ font-family: 'Playfair Display', serif; color: {NAV}; }}

/* Top bar */
.grad-bar {{ height: 4px; background: linear-gradient(90deg, {RED} 0%, {NAV} 60%, {GOLD} 100%); border-radius: 2px; margin-bottom: 1.2rem; }}

/* Metric cards */
div[data-testid="metric-container"] {{
    background: white; border: 1px solid {BORD}; border-top: 3px solid {RED};
    border-radius: 10px; padding: 14px 18px; box-shadow: 0 2px 8px rgba(17,34,90,0.06);
}}
div[data-testid="metric-container"] label {{
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b7fa3;
}}
div[data-testid="metric-container"] div[data-testid="metric-value"] {{
    font-family: 'Playfair Display', serif; font-size: 28px; color: {RED};
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: white; border-bottom: 2px solid {BORD}; gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: #6b7fa3; padding: 12px 28px; border: none; border-bottom: 3px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {RED} !important; border-bottom: 3px solid {RED} !important; background: white !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: white !important; border-right: 1px solid {BORD};
}}
.stSlider label, .stSelectbox label, .stMultiSelect label {{
    font-size: 10px !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important; color: #6b7fa3 !important;
}}

/* Cards */
.nat-card {{
    background: white; border: 1px solid {BORD}; border-radius: 10px;
    padding: 18px 22px; box-shadow: 0 2px 8px rgba(17,34,90,0.05); margin-bottom: 14px;
}}
.nat-card-red  {{ border-top: 4px solid {RED}; }}
.nat-card-navy {{ border-top: 4px solid {NAV}; }}
.nat-card-gold {{ border-top: 4px solid {GOLD}; }}
.nat-card-green {{ border-top: 4px solid {GRN}; }}

/* Stat rows */
.nat-label {{
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b7fa3; margin-bottom: 6px;
}}
.stat-row {{ display: flex; align-items: baseline; margin-bottom: 7px; font-size: 13px; }}
.stat-label {{ color: #6b7fa3; min-width: 150px; font-size: 12px; }}
.stat-val   {{ font-weight: 600; color: {NAV}; }}

/* Percentile bars */
.pct-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
.pct-label {{ color: #6b7fa3; font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; width: 130px; flex-shrink: 0; }}
.pct-track {{ flex: 1; background: {BORD}; border-radius: 3px; height: 6px; }}
.pct-fill  {{ height: 6px; border-radius: 3px; }}
.pct-num   {{ color: {NAV}; font-size: 11px; font-weight: 700; width: 28px; text-align: right; }}

/* Score badge */
.score-badge {{
    display: inline-block; font-family: 'Playfair Display', serif;
    font-size: 13px; font-weight: 700; padding: 3px 12px;
    border-radius: 20px; color: white; margin-left: 8px; vertical-align: middle;
}}

/* Derived pills */
.derived-pill {{
    background: {SURF}; border: 1px solid {BORD}; border-radius: 8px;
    padding: 10px 14px; display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 8px;
}}
.derived-name {{ color: {NAV}; font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }}
.derived-interp {{ color: #6b7fa3; font-size: 11px; margin-top: 2px; }}
.derived-val {{ font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; }}

/* Weight section */
.weight-header {{ font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: {RED}; margin: 1rem 0 0.4rem 0; }}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────
SPREADSHEET_ID   = st.secrets.get("GOOGLE_SHEET_ID", "1J27zw_UngoTNdq6VKPF6RhB8aqfmvlpX60GtrXjOsbs")
PITCHER_POSITIONS = {"Starting Pitcher", "Relief Pitcher", "Right Hand Pitcher"}

# ── AUTH & DATA ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_client():
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly",
              "https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(ttl=300, show_spinner=False)
def load_ws(sheet_id, name):
    ws = get_client().open_by_key(sheet_id).worksheet(name)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    df.columns = [str(c).strip() for c in df.columns]
    return df.replace("", np.nan).dropna(how="all")

@st.cache_data(ttl=300)
def load_data():
    try:
        sprint = load_ws(SPREADSHEET_ID, "Sprint")
        anthro = load_ws(SPREADSHEET_ID, "Anthropometrics")
        fp     = load_ws(SPREADSHEET_ID, "Force Plate")
    except Exception as e:
        st.error(f"Could not load data: {e}"); st.stop()

    def to_num(df, cols):
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def parse_unit(s):
        try: return float(str(s).replace("kg","").replace("cm","").strip())
        except: return np.nan

    sprint = to_num(sprint, ["10yd Best","20yd Best","30yd Best","30yd Best.","Split1","Split2","Split3","Year"])
    sprint["DPL ID"] = sprint["DPL ID"].astype(str).str.strip()

    # Normalize sprint split columns: 2023 uses named yd cols, 2024/2025 use Split1/2/3
    def resolve_sprint_splits(df):
        df = df.copy()
        # 10yd
        if "10yd Best" in df.columns:
            df["10yd"] = pd.to_numeric(df["10yd Best"], errors="coerce")
        elif "Split1" in df.columns:
            df["10yd"] = pd.to_numeric(df["Split1"], errors="coerce")
        else:
            df["10yd"] = np.nan
        # 20yd
        if "20yd Best" in df.columns:
            df["20yd"] = pd.to_numeric(df["20yd Best"], errors="coerce")
        elif "Split2" in df.columns:
            df["20yd"] = pd.to_numeric(df["Split2"], errors="coerce")
        else:
            df["20yd"] = np.nan
        # 30yd — prefer named col, fall back to Split3
        if "30yd Best" in df.columns:
            df["30yd"] = pd.to_numeric(df["30yd Best"], errors="coerce")
        elif "30yd Best." in df.columns:
            df["30yd"] = pd.to_numeric(df["30yd Best."], errors="coerce")
        elif "Split3" in df.columns:
            df["30yd"] = pd.to_numeric(df["Split3"], errors="coerce")
        else:
            df["30yd"] = np.nan
        # Fill gaps: for 2023 rows missing named cols, use Split cols if present
        if "Split1" in df.columns:
            df["10yd"] = df["10yd"].fillna(pd.to_numeric(df["Split1"], errors="coerce"))
        if "Split2" in df.columns:
            df["20yd"] = df["20yd"].fillna(pd.to_numeric(df["Split2"], errors="coerce"))
        if "Split3" in df.columns:
            df["30yd"] = df["30yd"].fillna(pd.to_numeric(df["Split3"], errors="coerce"))
        return df

    # Resolve sprint splits correctly across years
    # 2023: 10yd Best = cumulative 10yd, 30yd Best = cumulative 30yd total
    # 2024/2025: Split1 = cumulative 10yd, Total Time [s] = cumulative 30yd total
    sprint = to_num(sprint, ["Total Time [s]"])

    def resolve_sprint_splits(df):
        df = df.copy()
        # Cumulative 10yd
        df["10yd"] = pd.to_numeric(df.get("10yd Best", pd.Series(dtype=float)), errors="coerce")
        if "Split1" in df.columns:
            df["10yd"] = df["10yd"].fillna(pd.to_numeric(df["Split1"], errors="coerce"))

        # Cumulative 30yd total
        df["30yd"] = pd.to_numeric(df.get("30yd Best", pd.Series(dtype=float)), errors="coerce")
        if "30yd Best." in df.columns:
            df["30yd"] = df["30yd"].fillna(pd.to_numeric(df["30yd Best."], errors="coerce"))
        if "Total Time [s]" in df.columns:
            df["30yd"] = df["30yd"].fillna(pd.to_numeric(df["Total Time [s]"], errors="coerce"))

        # Drop implausible values: 10yd should be 0.5–1.5s, 30yd should be 2.5–6s
        df.loc[~df["10yd"].between(0.5, 1.5), "10yd"] = np.nan
        df.loc[~df["30yd"].between(2.5, 6.0), "30yd"] = np.nan

        return df

    sprint = resolve_sprint_splits(sprint)

    # Normalize name column
    if "Full Name Reverse" not in sprint.columns:
        if "Name" in sprint.columns:
            sprint["Full Name Reverse"] = sprint["Name"]
        elif "Player" in sprint.columns:
            sprint["Full Name Reverse"] = sprint["Player"]

    fp = to_num(fp, ["Concentric Impulse [Ns]","Concentric Impulse [N s]",
                     "RSI-Modified [m/s]","RSI-modified [m/s]",
                     "Peak Power [W]","Peak Power / BM [W/kg]","Year"])
    fp["DPL ID"] = fp["DPL ID"].astype(str).str.strip()

    # Normalize force plate column names across years
    if "Concentric Impulse [N s]" in fp.columns and "Concentric Impulse [Ns]" not in fp.columns:
        fp["Concentric Impulse [Ns]"] = fp["Concentric Impulse [N s]"]
    elif "Concentric Impulse [N s]" in fp.columns:
        fp["Concentric Impulse [Ns]"] = fp["Concentric Impulse [Ns]"].fillna(fp["Concentric Impulse [N s]"])

    if "RSI-modified [m/s]" in fp.columns and "RSI-Modified [m/s]" not in fp.columns:
        fp["RSI-Modified [m/s]"] = fp["RSI-modified [m/s]"]
    elif "RSI-modified [m/s]" in fp.columns:
        fp["RSI-Modified [m/s]"] = fp["RSI-Modified [m/s]"].fillna(fp["RSI-modified [m/s]"])

    if "Athlete" in fp.columns and "Full Name Reverse" not in fp.columns:
        fp["Full Name Reverse"] = fp["Athlete"]
    elif "Athlete" in fp.columns:
        fp["Full Name Reverse"] = fp["Full Name Reverse"].fillna(fp["Athlete"])

    # CMJ filter — different labels across years
    if "Test Type" in fp.columns:
        fp = fp[fp["Test Type"].isin({"CMJ", "Countermovement Jump"})]

    anthro["DPL ID"] = anthro["DPL ID"].astype(str).str.strip()
    anthro = to_num(anthro, ["Height","Body Weight","Body Weight (kg)","Arm Span","Year"])
    anthro["height_cm"]   = anthro["Height"].fillna(anthro["Stature Height 1"].apply(parse_unit) if "Stature Height 1" in anthro.columns else np.nan)
    anthro["weight_kg"]   = anthro["Body Weight (kg)"].fillna(anthro["Body Weight"]).fillna(anthro["Stature Body Weight 1"].apply(parse_unit) if "Stature Body Weight 1" in anthro.columns else np.nan)
    anthro["wingspan_cm"] = anthro["Arm Span"].fillna(anthro["Stature Arm Span 1"].apply(parse_unit) if "Stature Arm Span 1" in anthro.columns else np.nan)

    sprint_best = sprint.groupby("DPL ID").agg(
        sprint_10=("10yd","min"), sprint_30=("30yd","min"),
        sprint_name=("Full Name Reverse","first"), sprint_year=("Year","first")
    ).reset_index()
    sprint_best["sprint_20"] = np.nan  # 20yd segment splits not comparable across years

    fp_best = fp.groupby("DPL ID").agg(
        concentric_impulse=("Concentric Impulse [Ns]","max"),
        mrsi=("RSI-Modified [m/s]","max"),
        peak_power=("Peak Power [W]","max"),
        rel_peak_power=("Peak Power / BM [W/kg]","max"),
        fp_name=("Full Name Reverse","first"),
        fp_position=("Position","first"),
        fp_year=("Year","first")
    ).reset_index()

    anthro_best = anthro.groupby("DPL ID").agg(
        height_cm=("height_cm","first"), weight_kg=("weight_kg","first"), wingspan_cm=("wingspan_cm","first"),
        anthro_name=("Full Name Reverse","first"), anthro_position=("Position","first"), anthro_year=("Year","first")
    ).reset_index()

    m = fp_best.merge(anthro_best, on="DPL ID", how="outer").merge(sprint_best, on="DPL ID", how="outer")
    m["name"]     = m["fp_name"].fillna(m["anthro_name"]).fillna(m["sprint_name"])
    m["position"] = m["fp_position"].fillna(m["anthro_position"])
    m["year"]     = pd.to_numeric(m["fp_year"].fillna(m["anthro_year"]).fillna(m["sprint_year"]), errors="coerce").astype("Int64")
    m["height_in"]   = m["height_cm"] / 2.54
    m["weight_lb"]   = m["weight_kg"] * 2.205
    m["wingspan_in"] = m["wingspan_cm"] / 2.54

    # Derived
    m["ws_ht_ratio"]    = m["wingspan_cm"] / m["height_cm"]
    m["bmi"]            = m["weight_kg"] / ((m["height_cm"] / 100) ** 2)
    m["projection_raw"] = m["height_cm"] - (m["bmi"] * 1.5)

    # Only keep players who have both jump AND sprint data
    m = m[m["concentric_impulse"].notna() & m["sprint_30"].notna()].copy()

    return m[m["name"].notna()].reset_index(drop=True)

# ── SCORING ────────────────────────────────────────────────────────────────
def pct_rank(series, value, lower_is_better=False):
    valid = series.dropna()
    if len(valid) < 2 or pd.isna(value): return np.nan
    p = stats.percentileofscore(valid, value, kind="rank")
    return (100 - p) if lower_is_better else p

def compute_scores(df, aw, pw):
    rows = []
    for _, r in df.iterrows():
        is_p = r["position"] in PITCHER_POSITIONS
        ci_p   = pct_rank(df["concentric_impulse"], r["concentric_impulse"])
        mr_p   = pct_rank(df["mrsi"], r["mrsi"])
        s30_p  = pct_rank(df["sprint_30"], r["sprint_30"], True)
        h_p    = pct_rank(df["height_cm"], r["height_cm"])
        w_p    = pct_rank(df["weight_kg"], r["weight_kg"])
        rp_p   = pct_rank(df["rel_peak_power"], r["rel_peak_power"])
        ws_p   = pct_rank(df["wingspan_cm"], r["wingspan_cm"])
        wsr_p  = pct_rank(df["ws_ht_ratio"], r["ws_ht_ratio"])
        proj_p = pct_rank(df["projection_raw"], r["projection_raw"])

        ap, aww = [], []
        for k, v in [("ci", ci_p), ("mrsi", mr_p), ("sprint", s30_p)]:
            if not pd.isna(v): ap.append(v * aw[k]); aww.append(aw[k])
        ath = (sum(ap) / sum(aww)) if aww else np.nan

        if is_p:
            pot_items = [("height", h_p), ("weight", w_p), ("mrsi", mr_p),
                         ("rel_power", rp_p), ("sprint", s30_p), ("ws_ht_ratio", wsr_p)]
            wts = pw["pitcher"]
        else:
            pot_items = [("height", h_p), ("weight", w_p), ("mrsi", mr_p),
                         ("rel_power", rp_p), ("sprint", s30_p)]
            wts = pw["position"]

        pp, pww = [], []
        for k, v in pot_items:
            if not pd.isna(v): pp.append(v * wts[k]); pww.append(wts[k])
        pot = (sum(pp) / sum(pww)) if pww else np.nan

        rows.append({"athletic_score": ath, "potential_score": pot,
                     "ci_pct": ci_p, "mrsi_pct": mr_p, "s30_pct": s30_p,
                     "h_pct": h_p, "w_pct": w_p, "rpp_pct": rp_p,
                     "ws_pct": ws_p, "wsr_pct": wsr_p, "proj_pct": proj_p})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

# ── HELPERS ────────────────────────────────────────────────────────────────
def score_color(val):
    if pd.isna(val): return "#9AAAC0"
    if val >= 80: return GRN
    if val >= 60: return NAV
    if val >= 40: return GOLD
    return RED

def grade(val):
    if pd.isna(val): return ("—", "#9AAAC0")
    if val >= 80: return ("Elite", GRN)
    if val >= 65: return ("Plus", NAV)
    if val >= 50: return ("Avg+", GOLD)
    if val >= 35: return ("Avg", "#E8A020")
    return ("Below", RED)

def fh(inches):
    if pd.isna(inches): return "—"
    return f"{int(inches // 12)}'{inches % 12:.0f}\""

def fv(v, fmt=".1f", suffix=""):
    return f"{v:{fmt}}{suffix}" if not pd.isna(v) else "—"

def pct_bar(label, pct):
    if pd.isna(pct): return ""
    c = score_color(pct)
    return f"""<div class='pct-row'>
      <span class='pct-label'>{label}</span>
      <div class='pct-track'><div class='pct-fill' style='width:{pct:.0f}%;background:{c}'></div></div>
      <span class='pct-num'>{pct:.0f}</span>
    </div>"""

def gauge_fig(score, label):
    sd = 0 if pd.isna(score) else score
    color = score_color(sd)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=sd,
        number={"font": {"size": 44, "color": NAV, "family": "Playfair Display, serif"}, "suffix": ""},
        title={"text": label, "font": {"size": 11, "color": "#6b7fa3", "family": "Source Sans 3, sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#9AAAC0", "size": 9}, "nticks": 5},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": SURF, "borderwidth": 1, "bordercolor": BORD,
            "steps": [{"range": [0, 35], "color": "#fdf2f2"},
                      {"range": [35, 50], "color": "#fdf8ef"},
                      {"range": [50, 65], "color": "#f0f4fb"},
                      {"range": [65, 100], "color": "#f0faf5"}],
            "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.8, "value": sd}
        }
    ))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=45, b=5),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font={"family": "Source Sans 3, sans-serif"})
    return fig

def radar_fig(labels, values):
    vals = [max(v, 0) if not pd.isna(v) else 0 for v in values]
    colors_m = [score_color(v) for v in values]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself",
        fillcolor=f"rgba(17,34,90,0.07)",
        line=dict(color=NAV, width=2),
        marker=dict(color=colors_m + [colors_m[0]], size=8,
                    line=dict(color="white", width=1.5))
    ))
    fig.update_layout(
        polar=dict(bgcolor=SURF,
                   radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(color="#9AAAC0", size=8),
                                   gridcolor=BORD, linecolor=BORD),
                   angularaxis=dict(tickfont=dict(color=NAV, size=11,
                                                  family="Source Sans 3, sans-serif"),
                                    gridcolor=BORD, linecolor=BORD)),
        showlegend=False, paper_bgcolor="white",
        margin=dict(l=40, r=40, t=20, b=20), height=290
    )
    return fig

def scatter_fig(df):
    df2 = df.dropna(subset=["athletic_score", "potential_score"]).copy()
    fig = px.scatter(
        df2, x="athletic_score", y="potential_score",
        color="position", hover_name="name",
        hover_data={"concentric_impulse": ":.1f", "mrsi": ":.2f",
                    "sprint_30": ":.3f", "height_in": ":.1f",
                    "weight_lb": ":.0f", "ws_ht_ratio": ":.3f", "bmi": ":.1f"},
        labels={"athletic_score": "Athletic Score", "potential_score": "Potential Score"},
        template="plotly_white", height=560,
        color_discrete_sequence=[RED, NAV, GOLD, GRN, "#6b7fa3", "#c94060"],
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor=SURF,
        font=dict(family="Source Sans 3, sans-serif", color=NAV),
        xaxis=dict(gridcolor=BORD, zerolinecolor=BORD, title_font=dict(color="#6b7fa3")),
        yaxis=dict(gridcolor=BORD, zerolinecolor=BORD, title_font=dict(color="#6b7fa3")),
        legend=dict(font=dict(color=NAV, size=11)),
    )
    fig.add_hline(y=50, line_dash="dot", line_color=BORD, line_width=1)
    fig.add_vline(x=50, line_dash="dot", line_color=BORD, line_width=1)
    return fig

# ── LOAD ───────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    raw_df = load_data()

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<p style="font-size:10px;font-weight:600;letter-spacing:0.2em;color:{RED};margin-bottom:2px">WASHINGTON NATIONALS</p>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="margin:0 0 1rem 0;font-size:1.1rem">Draft Scout</h2>', unsafe_allow_html=True)

    years     = sorted(raw_df["year"].dropna().unique().tolist())
    sel_years = st.multiselect("Draft Class", years, default=years)
    positions = sorted(raw_df["position"].dropna().unique().tolist())
    sel_pos   = st.multiselect("Position", positions, default=positions)

    st.markdown(f'<div class="weight-header">⚡ Athletic Score Weights</div>', unsafe_allow_html=True)
    w_ci     = st.slider("Concentric Impulse", 0, 100, 40, key="w_ci")
    w_mrsi   = st.slider("mRSI", 0, 100, 35, key="w_mrsi")
    w_sprint = st.slider("Sprint (30yd)", 0, 100, 25, key="w_sprint")
    tot_a = w_ci + w_mrsi + w_sprint
    st.markdown(f'<p style="font-size:10px;text-align:right;color:{"#4CAF82" if tot_a==100 else RED}">Total: {tot_a} {"✓" if tot_a==100 else "(needs to = 100)"}</p>', unsafe_allow_html=True)

    st.markdown(f'<div class="weight-header">🎯 Potential — Position Players</div>', unsafe_allow_html=True)
    pp_h  = st.slider("Height", 0, 100, 20, key="pp_h")
    pp_w  = st.slider("Weight", 0, 100, 20, key="pp_w")
    pp_mr = st.slider("mRSI", 0, 100, 25, key="pp_mr")
    pp_rp = st.slider("Rel. Peak Power", 0, 100, 20, key="pp_rp")
    pp_sp = st.slider("Sprint", 0, 100, 15, key="pp_sp")
    tot_pp = pp_h+pp_w+pp_mr+pp_rp+pp_sp
    st.markdown(f'<p style="font-size:10px;text-align:right;color:{"#4CAF82" if tot_pp==100 else RED}">Total: {tot_pp} {"✓" if tot_pp==100 else "(needs to = 100)"}</p>', unsafe_allow_html=True)

    st.markdown(f'<div class="weight-header">⚾ Potential — Pitchers</div>', unsafe_allow_html=True)
    pi_h   = st.slider("Height", 0, 100, 18, key="pi_h")
    pi_w   = st.slider("Weight", 0, 100, 12, key="pi_w")
    pi_mr  = st.slider("mRSI", 0, 100, 20, key="pi_mr")
    pi_rp  = st.slider("Rel. Peak Power", 0, 100, 20, key="pi_rp")
    pi_sp  = st.slider("Sprint", 0, 100, 10, key="pi_sp")
    pi_wsr = st.slider("Wingspan : Height", 0, 100, 20, key="pi_wsr")
    tot_pi = pi_h+pi_w+pi_mr+pi_rp+pi_sp+pi_wsr
    st.markdown(f'<p style="font-size:10px;text-align:right;color:{"#4CAF82" if tot_pi==100 else RED}">Total: {tot_pi} {"✓" if tot_pi==100 else "(needs to = 100)"}</p>', unsafe_allow_html=True)

    st.markdown("---")
    sort_by = st.selectbox("Sort Rankings By",
        ["Athletic Score","Potential Score","30yd Sprint","mRSI","Concentric Impulse","Projection"])
    st.markdown("---")
    if st.button("↻ Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.rerun()

# ── SCORE ──────────────────────────────────────────────────────────────────
aw = {"ci": w_ci/100, "mrsi": w_mrsi/100, "sprint": w_sprint/100}
pw = {
    "position": {"height": pp_h/100, "weight": pp_w/100, "mrsi": pp_mr/100,
                 "rel_power": pp_rp/100, "sprint": pp_sp/100},
    "pitcher":  {"height": pi_h/100, "weight": pi_w/100, "mrsi": pi_mr/100,
                 "rel_power": pi_rp/100, "sprint": pi_sp/100, "ws_ht_ratio": pi_wsr/100},
}
df = compute_scores(raw_df, aw, pw)

# ── FILTER ─────────────────────────────────────────────────────────────────
filtered = df[df["year"].isin(sel_years) & df["position"].isin(sel_pos)].copy()
sc_map = {"Athletic Score":("athletic_score",False),"Potential Score":("potential_score",False),
          "30yd Sprint":("sprint_30",True),"mRSI":("mrsi",False),
          "Concentric Impulse":("concentric_impulse",False),"Projection":("proj_pct",False)}
sc, sa = sc_map[sort_by]
filtered = filtered.sort_values(sc, ascending=sa, na_position="last")

# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown('<div class="grad-bar"></div>', unsafe_allow_html=True)
hc1, hc2, hc3, hc4, hc5 = st.columns([3,1,1,1,1])
with hc1:
    st.markdown(f'<p style="font-size:10px;font-weight:600;letter-spacing:0.2em;color:{RED};margin-bottom:4px">WASHINGTON NATIONALS · PLAYER DEVELOPMENT</p>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="margin:0;font-size:2rem">Draft Athletic Scouting</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#6b7fa3;font-size:12px;margin:0">Athletic Qualities + Physical Potential · 2023–2025 Draft Classes</p>', unsafe_allow_html=True)
hc2.metric("Athletes", len(filtered))
hc3.metric("Pitchers", int(filtered["position"].isin(PITCHER_POSITIONS).sum()))
hc4.metric("w/ Sprint", int(filtered["sprint_30"].notna().sum()))
hc5.metric("w/ Jump", int(filtered["concentric_impulse"].notna().sum()))

st.markdown("<br>", unsafe_allow_html=True)

tab_rank, tab_profile, tab_scatter = st.tabs(["Rankings", "Player Profile", "Scatter"])

# ── TAB 1: RANKINGS ────────────────────────────────────────────────────────
with tab_rank:
    disp = filtered[[
        "name","position","year",
        "athletic_score","potential_score",
        "concentric_impulse","mrsi","rel_peak_power",
        "sprint_10","sprint_30",
        "height_in","weight_lb","wingspan_in",
        "ws_ht_ratio","bmi","proj_pct"
    ]].copy()
    disp.rename(columns={
        "name":"Athlete","position":"Position","year":"Year",
        "athletic_score":"Athletic","potential_score":"Potential",
        "concentric_impulse":"CI (Ns)","mrsi":"mRSI","rel_peak_power":"Rel Pwr (W/kg)",
        "sprint_10":"10yd","sprint_30":"30yd",
        "height_in":"Ht (in)","weight_lb":"Wt (lb)","wingspan_in":"Wingspan (in)",
        "ws_ht_ratio":"WS:Ht","bmi":"BMI","proj_pct":"Proj %ile"
    }, inplace=True)
    for c in ["Athletic","Potential","CI (Ns)","mRSI","Rel Pwr (W/kg)",
              "10yd","30yd","Ht (in)","Wt (lb)","Wingspan (in)","WS:Ht","BMI","Proj %ile"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(2)

    st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=560,
        column_config={
            "Athletic": st.column_config.ProgressColumn("Athletic", min_value=0, max_value=100, format="%.1f"),
            "Potential": st.column_config.ProgressColumn("Potential", min_value=0, max_value=100, format="%.1f"),
            "Proj %ile": st.column_config.ProgressColumn("Proj %ile", min_value=0, max_value=100, format="%.0f"),
        })
    csv = disp.to_csv(index=False).encode()
    st.download_button("↓ Download CSV", csv, "draft_scout.csv", "text/csv")

# ── TAB 2: PLAYER PROFILE ──────────────────────────────────────────────────
with tab_profile:
    player_names = filtered["name"].dropna().sort_values().tolist()
    if not player_names:
        st.warning("No athletes match the current filters.")
    else:
        search = st.text_input("Search athlete", placeholder="Type a name…", label_visibility="collapsed")
        opts   = [n for n in player_names if search.lower() in n.lower()] if search else player_names
        sel    = st.selectbox("Athlete", opts, label_visibility="collapsed")
        r      = filtered[filtered["name"] == sel].iloc[0]
        is_p   = r["position"] in PITCHER_POSITIONS

        st.markdown("<br>", unsafe_allow_html=True)

        # Name row
        n1, n2 = st.columns([3, 1])
        with n1:
            ath_g, ath_c = grade(r["athletic_score"])
            pot_g, pot_c = grade(r["potential_score"])
            yr = int(r["year"]) if not pd.isna(r["year"]) else "—"
            pitcher_tag = f' <span class="score-badge" style="background:{RED};font-size:11px">⚾ Pitcher</span>' if is_p else ""
            st.markdown(f'<h2 style="margin:0">{r["name"]}{pitcher_tag}</h2>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#6b7fa3;font-size:12px;margin:4px 0 0 0">{r["position"]} · {yr} Draft Class</p>', unsafe_allow_html=True)
        with n2:
            st.markdown(f"""
            <div style="text-align:right">
              <span class="nat-label">Athletic</span><br>
              <span style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;color:{ath_c}">{fv(r["athletic_score"],".0f")}</span>
              <span class="score-badge" style="background:{ath_c}">{ath_g}</span><br>
              <span class="nat-label" style="margin-top:6px;display:block">Potential</span>
              <span style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;color:{pot_c}">{fv(r["potential_score"],".0f")}</span>
              <span class="score-badge" style="background:{pot_c}">{pot_g}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauges
        g1, g2 = st.columns(2)
        with g1: st.plotly_chart(gauge_fig(r["athletic_score"], "ATHLETIC SCORE"), use_container_width=True, key="ga")
        with g2: st.plotly_chart(gauge_fig(r["potential_score"], "POTENTIAL SCORE"), use_container_width=True, key="gp")

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns([1.1, 0.9])

        with left:
            # Physical
            st.markdown(f'<p class="nat-label">Physical Measurements</p>', unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            for col, label, val, pct_val in [
                (p1, "Height", fh(r["height_in"]), r["h_pct"]),
                (p2, "Weight", fv(r["weight_lb"],".0f"," lb"), r["w_pct"]),
                (p3, "Wingspan", fh(r["wingspan_in"]), r["ws_pct"]),
            ]:
                g_label, g_color = grade(pct_val)
                col.markdown(f"""
                <div class="nat-card nat-card-navy" style="text-align:center;padding:14px">
                  <div class="nat-label">{label}</div>
                  <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:{NAV}">{val}</div>
                  <div style="font-size:11px;color:#6b7fa3;margin-top:4px">{fv(pct_val,".0f")}th %ile
                    <span class="score-badge" style="background:{g_color};font-size:9px;padding:2px 7px">{g_label}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Derived
            st.markdown(f'<p class="nat-label" style="margin-top:1rem">Derived Metrics</p>', unsafe_allow_html=True)
            wsr = r["ws_ht_ratio"]
            wsr_c = GRN if not pd.isna(wsr) and wsr > 1.02 else (GOLD if not pd.isna(wsr) and wsr >= 0.98 else RED)
            wsr_note = "Wingspan exceeds height — positive for a pitcher" if not pd.isna(wsr) and wsr > 1.0 else ("Wingspan ≈ height" if not pd.isna(wsr) and wsr >= 0.98 else "Below average arm length")
            bmi = r["bmi"]
            bmi_c = GRN if not pd.isna(bmi) and bmi < 22 else (GOLD if not pd.isna(bmi) and bmi < 25 else RED)
            bmi_note = "Lean frame — significant room to add mass" if not pd.isna(bmi) and bmi < 22 else ("Athletic build — moderate projection" if not pd.isna(bmi) and bmi < 25 else "Filled out — limited mass projection")
            proj_g, proj_c = grade(r["proj_pct"])

            st.markdown(f"""
            <div class="derived-pill">
              <div><div class="derived-name">Wingspan : Height Ratio</div><div class="derived-interp">{wsr_note}</div></div>
              <div class="derived-val" style="color:{wsr_c}">{fv(wsr,".3f")}</div>
            </div>
            <div class="derived-pill">
              <div><div class="derived-name">BMI — Mass Projection Index</div><div class="derived-interp">{bmi_note}</div></div>
              <div class="derived-val" style="color:{bmi_c}">{fv(bmi,".1f")}</div>
            </div>
            <div class="derived-pill">
              <div><div class="derived-name">Physical Projection %ile</div><div class="derived-interp">Height × leanness vs. draft class</div></div>
              <div class="derived-val" style="color:{proj_c}">{fv(r["proj_pct"],".0f")}th <span class="score-badge" style="background:{proj_c};font-size:10px">{proj_g}</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Force plate
            st.markdown(f'<p class="nat-label" style="margin-top:1rem">Force Plate — CMJ</p>', unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            for col, label, val_str, pct_val, unit in [
                (f1, "Conc. Impulse", fv(r["concentric_impulse"],".0f"), r["ci_pct"], "Ns"),
                (f2, "mRSI", fv(r["mrsi"],".2f"), r["mrsi_pct"], "m/s"),
                (f3, "Rel. Peak Power", fv(r["rel_peak_power"],".1f"), r["rpp_pct"], "W/kg"),
            ]:
                g_label, g_color = grade(pct_val)
                col.markdown(f"""
                <div class="nat-card nat-card-red" style="text-align:center;padding:14px">
                  <div class="nat-label">{label}</div>
                  <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:{NAV}">{val_str}</div>
                  <div style="font-size:11px;color:#6b7fa3;margin-top:4px">{unit} · {fv(pct_val,".0f")}th %ile
                    <span class="score-badge" style="background:{g_color};font-size:9px;padding:2px 7px">{g_label}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Sprint
            st.markdown(f'<p class="nat-label" style="margin-top:1rem">Sprint</p>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            for col, label, val_str, pct_val in [
                (s1, "10 Yard", fv(r["sprint_10"],".3f"), None),
                (s2, "30 Yard", fv(r["sprint_30"],".3f"), r["s30_pct"]),
            ]:
                g_label, g_color = grade(pct_val) if pct_val is not None else ("—", "#9AAAC0")
                sub = f"seconds · {fv(pct_val,'.0f')}th %ile" if pct_val is not None else "seconds"
                col.markdown(f"""
                <div class="nat-card nat-card-gold" style="text-align:center;padding:14px">
                  <div class="nat-label">{label}</div>
                  <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:{NAV}">{val_str}</div>
                  <div style="font-size:11px;color:#6b7fa3;margin-top:4px">{sub}
                    {"<span class='score-badge' style='background:" + g_color + ";font-size:9px;padding:2px 7px'>" + g_label + "</span>" if pct_val is not None else ""}
                  </div>
                </div>""", unsafe_allow_html=True)

        with right:
            st.markdown(f'<p class="nat-label">Percentile Profile</p>', unsafe_allow_html=True)
            if is_p:
                rl = ["CI","mRSI","Sprint","Height","Weight","Rel Pwr","WS:Ht"]
                rv = [r["ci_pct"],r["mrsi_pct"],r["s30_pct"],r["h_pct"],r["w_pct"],r["rpp_pct"],r["wsr_pct"]]
            else:
                rl = ["CI","mRSI","Sprint","Height","Weight","Rel Pwr"]
                rv = [r["ci_pct"],r["mrsi_pct"],r["s30_pct"],r["h_pct"],r["w_pct"],r["rpp_pct"]]
            st.plotly_chart(radar_fig(rl, rv), use_container_width=True, key="rd")

            st.markdown(f'<p class="nat-label" style="margin-top:0.5rem">Percentile Bars</p>', unsafe_allow_html=True)
            all_bars = [
                ("Conc. Impulse", r["ci_pct"]),
                ("mRSI", r["mrsi_pct"]),
                ("Sprint 30yd", r["s30_pct"]),
                ("Height", r["h_pct"]),
                ("Weight", r["w_pct"]),
                ("Rel. Peak Power", r["rpp_pct"]),
                ("Wingspan", r["ws_pct"]),
                ("WS : Height", r["wsr_pct"]),
                ("Projection", r["proj_pct"]),
            ]
            st.markdown("".join(pct_bar(lbl, pct) for lbl, pct in all_bars if not pd.isna(pct)), unsafe_allow_html=True)

# ── TAB 3: SCATTER ─────────────────────────────────────────────────────────
with tab_scatter:
    st.markdown(f'<p class="nat-label">Athletic Score vs. Physical Potential — each dot is a player, colored by position</p>', unsafe_allow_html=True)
    st.plotly_chart(scatter_fig(filtered), use_container_width=True)
