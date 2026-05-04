import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import gspread
import json
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Draft Scout", layout="wide", page_icon="⚾")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.main { background: #080c10; }
.block-container { padding: 2rem 2.5rem; max-width: 1600px; }

/* Header */
.page-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
  background: linear-gradient(135deg, #e8f4f8 0%, #7eb8d4 50%, #3a7ca5 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -0.02em; margin: 0; line-height: 1; }
.page-sub { color: #4a6572; font-size: 0.72rem; letter-spacing: 0.15em;
  text-transform: uppercase; margin-top: 0.5rem; }

/* Stat cards */
.stat-card { background: #0d1520; border: 1px solid #1a2535;
  border-radius: 8px; padding: 1.2rem 1.4rem; position: relative; overflow: hidden; }
.stat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 2px; background: linear-gradient(90deg, #3a7ca5, #7eb8d4); }
.stat-label { color: #4a6572; font-size: 0.65rem; letter-spacing: 0.15em;
  text-transform: uppercase; margin-bottom: 0.4rem; }
.stat-value { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700;
  color: #e8f4f8; line-height: 1; }
.stat-sub { color: #4a6572; font-size: 0.65rem; margin-top: 0.3rem; }

/* Score cards */
.score-card { background: #0d1520; border: 1px solid #1a2535; border-radius: 8px;
  padding: 1.5rem; text-align: center; }
.score-label { color: #4a6572; font-size: 0.65rem; letter-spacing: 0.15em;
  text-transform: uppercase; margin-bottom: 0.6rem; }
.score-value { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
  line-height: 1; }
.score-grade { font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase;
  margin-top: 0.4rem; font-weight: 500; }

/* Player name */
.player-name { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800;
  color: #e8f4f8; letter-spacing: -0.02em; margin: 0; }
.player-meta { color: #4a6572; font-size: 0.7rem; letter-spacing: 0.12em;
  text-transform: uppercase; margin-top: 0.4rem; }
.pitcher-badge { display: inline-block; background: #0d2235; border: 1px solid #3a7ca5;
  color: #7eb8d4; border-radius: 4px; padding: 0.2rem 0.6rem;
  font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; margin-left: 0.6rem; }

/* Section headers */
.section-label { color: #4a6572; font-size: 0.65rem; letter-spacing: 0.2em;
  text-transform: uppercase; border-bottom: 1px solid #1a2535;
  padding-bottom: 0.5rem; margin-bottom: 1rem; margin-top: 1.5rem; }

/* Pct bar */
.pct-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.6rem; }
.pct-label { color: #4a6572; font-size: 0.65rem; letter-spacing: 0.08em;
  text-transform: uppercase; width: 110px; flex-shrink: 0; }
.pct-track { flex: 1; background: #1a2535; border-radius: 2px; height: 4px; }
.pct-fill { height: 4px; border-radius: 2px; }
.pct-num { color: #e8f4f8; font-size: 0.7rem; font-weight: 500; width: 32px;
  text-align: right; flex-shrink: 0; }

/* Derived metric pill */
.derived-pill { background: #0d2235; border: 1px solid #1a3550; border-radius: 6px;
  padding: 0.8rem 1rem; display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 0.5rem; }
.derived-name { color: #7eb8d4; font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; }
.derived-val { font-family: 'Syne', sans-serif; color: #e8f4f8; font-size: 1rem; font-weight: 700; }
.derived-interp { color: #4a6572; font-size: 0.65rem; margin-top: 0.2rem; }

/* Weight sliders */
.weight-section { background: #0a1018; border: 1px solid #1a2535; border-radius: 8px;
  padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
.weight-title { color: #7eb8d4; font-size: 0.65rem; letter-spacing: 0.15em;
  text-transform: uppercase; margin-bottom: 0.6rem; font-weight: 600; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #080c10 !important; border-right: 1px solid #1a2535; }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0d1520; border-radius: 6px;
  padding: 3px; border: 1px solid #1a2535; gap: 2px; }
.stTabs [data-baseweb="tab"] { color: #4a6572; border-radius: 4px;
  font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #1a2d42 !important; color: #7eb8d4 !important; }

/* Streamlit overrides */
.stSelectbox label, .stMultiSelect label, .stSlider label { color: #4a6572 !important;
  font-size: 0.65rem !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; }
div[data-testid="stMetric"] { background: #0d1520; border: 1px solid #1a2535;
  border-radius: 8px; padding: 0.8rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SPREADSHEET_ID = st.secrets.get("GOOGLE_SHEET_ID", "1J27zw_UngoTNdq6VKPF6RhB8aqfmvlpX60GtrXjOsbs")
PITCHER_POSITIONS = {"Starting Pitcher", "Relief Pitcher", "Right Hand Pitcher"}

# ── AUTH ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_gspread_client():
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly",
              "https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(ttl=300, show_spinner=False)
def load_worksheet(sheet_id, name):
    client = get_gspread_client()
    ws = client.open_by_key(sheet_id).worksheet(name)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    df.columns = [str(c).strip() for c in df.columns]
    return df.replace("", np.nan).dropna(how="all")

@st.cache_data(ttl=300)
def load_data():
    try:
        sprint = load_worksheet(SPREADSHEET_ID, "Sprint")
        anthro = load_worksheet(SPREADSHEET_ID, "Anthropometrics")
        fp     = load_worksheet(SPREADSHEET_ID, "Force Plate")
    except Exception as e:
        st.error(f"Failed to load sheets: {e}")
        st.stop()

    def to_num(df, cols):
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def parse_unit(s):
        try: return float(str(s).replace("kg","").replace("cm","").strip())
        except: return np.nan

    sprint = to_num(sprint, ["10yd","20yd","30yd","Year"])
    sprint["DPL ID"] = sprint["DPL ID"].astype(str).str.strip()

    fp = to_num(fp, ["Concentric Impulse [Ns]","RSI-Modified [m/s]","Peak Power [W]","Peak Power / BM [W/kg]","Year"])
    fp["DPL ID"] = fp["DPL ID"].astype(str).str.strip()
    if "Test Type" in fp.columns: fp = fp[fp["Test Type"] == "CMJ"]

    anthro["DPL ID"] = anthro["DPL ID"].astype(str).str.strip()
    anthro = to_num(anthro, ["Height","Body Weight","Body Weight (kg)","Arm Span","Year"])
    anthro["height_cm"]   = anthro["Height"].fillna(anthro["Stature Height 1"].apply(parse_unit) if "Stature Height 1" in anthro.columns else np.nan)
    anthro["weight_kg"]   = anthro["Body Weight (kg)"].fillna(anthro["Body Weight"]).fillna(anthro["Stature Body Weight 1"].apply(parse_unit) if "Stature Body Weight 1" in anthro.columns else np.nan)
    anthro["wingspan_cm"] = anthro["Arm Span"].fillna(anthro["Stature Arm Span 1"].apply(parse_unit) if "Stature Arm Span 1" in anthro.columns else np.nan)

    sprint_best = sprint.groupby("DPL ID").agg(
        sprint_10=("10yd","min"), sprint_20=("20yd","min"), sprint_30=("30yd","min"),
        sprint_name=("Full Name Reverse","first"), sprint_year=("Year","first")
    ).reset_index()

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

    # ── Derived metrics
    # Wingspan-to-height ratio (>1 is good, especially for pitchers)
    m["ws_ht_ratio"] = m["wingspan_cm"] / m["height_cm"]
    # BMI as proxy for mass relative to height (lower = more room to add mass)
    # Using kg/m² — lower BMI for height suggests projection room
    m["bmi"] = m["weight_kg"] / ((m["height_cm"] / 100) ** 2)
    # Projection score: tall + lean = high upside to add mass
    # z-score height high, z-score bmi low → projection
    m["projection_raw"] = m["height_cm"] - (m["bmi"] * 1.5)

    return m[m["name"].notna()].reset_index(drop=True)

# ── SCORING ───────────────────────────────────────────────────────────────────
def pct_rank(series, value, lower_is_better=False):
    valid = series.dropna()
    if len(valid) < 2 or pd.isna(value): return np.nan
    p = stats.percentileofscore(valid, value, kind="rank")
    return (100-p) if lower_is_better else p

def compute_scores(df, aw, pw):
    """aw = athletic weights dict, pw = potential weights dict (pitcher & position)"""
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
        wsr_p  = pct_rank(df["ws_ht_ratio"], r["ws_ht_ratio"])  # higher ratio = better
        proj_p = pct_rank(df["projection_raw"], r["projection_raw"])  # higher = more projection

        # Athletic score
        ath_map = [("ci", ci_p), ("mrsi", mr_p), ("sprint", s30_p)]
        ap, aww = [], []
        for k, v in ath_map:
            if not pd.isna(v): ap.append(v * aw[k]); aww.append(aw[k])
        ath = (sum(ap)/sum(aww)) if aww else np.nan

        # Potential score
        if is_p:
            pot_map = [("height", h_p), ("weight", w_p), ("mrsi", mr_p),
                       ("rel_power", rp_p), ("sprint", s30_p), ("ws_ht_ratio", wsr_p)]
            wts = pw["pitcher"]
        else:
            pot_map = [("height", h_p), ("weight", w_p), ("mrsi", mr_p),
                       ("rel_power", rp_p), ("sprint", s30_p)]
            wts = pw["position"]

        pp, pww = [], []
        for k, v in pot_map:
            if not pd.isna(v): pp.append(v * wts[k]); pww.append(wts[k])
        pot = (sum(pp)/sum(pww)) if pww else np.nan

        rows.append({
            "athletic_score": ath, "potential_score": pot,
            "ci_pct": ci_p, "mrsi_pct": mr_p, "s30_pct": s30_p,
            "h_pct": h_p, "w_pct": w_p, "rpp_pct": rp_p,
            "ws_pct": ws_p, "wsr_pct": wsr_p, "proj_pct": proj_p,
        })
    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return scored

# ── HELPERS ───────────────────────────────────────────────────────────────────
def score_color(val):
    if pd.isna(val): return "#2a3a4a"
    if val >= 80: return "#2dd4a0"
    if val >= 60: return "#f0b429"
    if val >= 40: return "#f07029"
    return "#e84545"

def grade(val):
    if pd.isna(val): return ("—", "#4a6572")
    if val >= 80: return ("ELITE", "#2dd4a0")
    if val >= 65: return ("PLUS", "#7eb8d4")
    if val >= 50: return ("AVG+", "#f0b429")
    if val >= 35: return ("AVG", "#f07029")
    return ("BELOW", "#e84545")

def fmt_h(inches):
    if pd.isna(inches): return "—"
    return f"{int(inches//12)}'{inches%12:.0f}\""

def fv(v, fmt=".1f", suffix=""):
    return f"{v:{fmt}}{suffix}" if not pd.isna(v) else "—"

def pct_bar_html(label, pct, extra=""):
    if pd.isna(pct): return ""
    c = score_color(pct)
    return f"""<div class='pct-row'>
      <span class='pct-label'>{label}</span>
      <div class='pct-track'><div class='pct-fill' style='width:{pct:.0f}%;background:{c}'></div></div>
      <span class='pct-num'>{pct:.0f}</span>
    </div>{"<div style='padding-left:118px;margin-top:-4px;margin-bottom:6px;color:#4a6572;font-size:0.6rem'>"+extra+"</div>" if extra else ""}"""

def gauge_fig(score, label, color):
    sd = 0 if pd.isna(score) else score
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=sd,
        number={"font":{"size":42,"color":"#e8f4f8","family":"Syne, sans-serif"},"suffix":""},
        title={"text":label,"font":{"size":11,"color":"#4a6572","family":"DM Mono, monospace"}},
        gauge={
            "axis":{"range":[0,100],"tickfont":{"color":"#4a6572","size":9},"tickwidth":1,"tickcolor":"#1a2535","nticks":5},
            "bar":{"color":color,"thickness":0.2},
            "bgcolor":"#0d1520","borderwidth":0,
            "steps":[{"range":[0,25],"color":"#090e15"},{"range":[25,50],"color":"#0b1219"},
                     {"range":[50,75],"color":"#0d1520"},{"range":[75,100],"color":"#0f1825"}],
            "threshold":{"line":{"color":color,"width":2},"thickness":0.75,"value":sd}
        }
    ))
    fig.update_layout(height=180, margin=dict(l=15,r=15,t=40,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family":"DM Mono, monospace"})
    return fig

def radar_fig(labels, values):
    vals = [max(v,0) if not pd.isna(v) else 0 for v in values]
    colors = [score_color(v) for v in values]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=labels+[labels[0]], fill="toself",
        fillcolor="rgba(62,140,180,0.08)",
        line=dict(color="#3a7ca5", width=1.5),
        marker=dict(color=colors+[colors[0]], size=7, line=dict(color="#0d1520", width=1))
    ))
    fig.update_layout(
        polar=dict(bgcolor="#0d1520",
                   radialaxis=dict(visible=True, range=[0,100],
                                   tickfont=dict(color="#2a3a4a",size=8),
                                   gridcolor="#1a2535", linecolor="#1a2535"),
                   angularaxis=dict(tickfont=dict(color="#7eb8d4",size=10,family="DM Mono"),
                                    gridcolor="#1a2535", linecolor="#1a2535")),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40,r=40,t=20,b=20), height=280
    )
    return fig

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner(""):
    raw_df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#7eb8d4;letter-spacing:-0.01em;margin-bottom:0.2rem'>⚾ DRAFT SCOUT</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#4a6572;font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:1.2rem'>Athletic Intelligence Platform</div>", unsafe_allow_html=True)

    st.markdown("<div style='color:#4a6572;font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.4rem'>Filters</div>", unsafe_allow_html=True)
    years = sorted(raw_df["year"].dropna().unique().tolist())
    sel_years = st.multiselect("Draft Class", years, default=years)
    positions = sorted(raw_df["position"].dropna().unique().tolist())
    sel_positions = st.multiselect("Position", positions, default=positions)

    st.markdown("---")
    st.markdown("<div style='color:#7eb8d4;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.8rem;font-weight:600'>⚡ Athletic Score Weights</div>", unsafe_allow_html=True)
    w_ci     = st.slider("Concentric Impulse", 0, 100, 40, key="w_ci")
    w_mrsi   = st.slider("mRSI", 0, 100, 35, key="w_mrsi")
    w_sprint = st.slider("Sprint (30yd)", 0, 100, 25, key="w_sprint")
    ath_total = w_ci + w_mrsi + w_sprint
    st.markdown(f"<div style='color:{'#2dd4a0' if ath_total==100 else '#e84545'};font-size:0.65rem;text-align:right'>Total: {ath_total} {'✓' if ath_total==100 else '(should = 100)'}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#7eb8d4;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.8rem;font-weight:600'>🎯 Potential — Position Players</div>", unsafe_allow_html=True)
    pp_h   = st.slider("Height", 0, 100, 20, key="pp_h")
    pp_w   = st.slider("Weight", 0, 100, 20, key="pp_w")
    pp_mr  = st.slider("mRSI", 0, 100, 25, key="pp_mr")
    pp_rp  = st.slider("Rel. Peak Power", 0, 100, 20, key="pp_rp")
    pp_sp  = st.slider("Sprint", 0, 100, 15, key="pp_sp")
    pp_total = pp_h+pp_w+pp_mr+pp_rp+pp_sp
    st.markdown(f"<div style='color:{'#2dd4a0' if pp_total==100 else '#e84545'};font-size:0.65rem;text-align:right'>Total: {pp_total} {'✓' if pp_total==100 else '(should = 100)'}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#7eb8d4;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.8rem;font-weight:600'>⚾ Potential — Pitchers</div>", unsafe_allow_html=True)
    pi_h   = st.slider("Height", 0, 100, 20, key="pi_h")
    pi_w   = st.slider("Weight", 0, 100, 15, key="pi_w")
    pi_mr  = st.slider("mRSI", 0, 100, 20, key="pi_mr")
    pi_rp  = st.slider("Rel. Peak Power", 0, 100, 20, key="pi_rp")
    pi_sp  = st.slider("Sprint", 0, 100, 10, key="pi_sp")
    pi_wsr = st.slider("Wingspan:Height Ratio", 0, 100, 15, key="pi_wsr")
    pi_total = pi_h+pi_w+pi_mr+pi_rp+pi_sp+pi_wsr
    st.markdown(f"<div style='color:{'#2dd4a0' if pi_total==100 else '#e84545'};font-size:0.65rem;text-align:right'>Total: {pi_total} {'✓' if pi_total==100 else '(should = 100)'}</div>", unsafe_allow_html=True)

    st.markdown("---")
    sort_by = st.selectbox("Sort Leaderboard By",
        ["Athletic Score","Potential Score","30yd Sprint","mRSI","Concentric Impulse","Projection"])
    st.markdown("---")
    if st.button("↺  Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.rerun()

# ── COMPUTE SCORES ────────────────────────────────────────────────────────────
aw = {"ci": w_ci/100, "mrsi": w_mrsi/100, "sprint": w_sprint/100}
pw = {
    "position": {"height": pp_h/100,"weight": pp_w/100,"mrsi": pp_mr/100,"rel_power": pp_rp/100,"sprint": pp_sp/100},
    "pitcher":  {"height": pi_h/100,"weight": pi_w/100,"mrsi": pi_mr/100,"rel_power": pi_rp/100,"sprint": pi_sp/100,"ws_ht_ratio": pi_wsr/100},
}
df = compute_scores(raw_df, aw, pw)

# ── FILTER ────────────────────────────────────────────────────────────────────
filtered = df[df["year"].isin(sel_years) & df["position"].isin(sel_positions)].copy()
sc_map = {"Athletic Score":("athletic_score",False),"Potential Score":("potential_score",False),
          "30yd Sprint":("sprint_30",True),"mRSI":("mrsi",False),
          "Concentric Impulse":("concentric_impulse",False),"Projection":("proj_pct",False)}
sc, sa = sc_map[sort_by]
filtered = filtered.sort_values(sc, ascending=sa, na_position="last")

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
h1, h2 = st.columns([2,1])
with h1:
    st.markdown("<div class='page-title'>Draft Scout</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Athletic Qualities + Physical Potential · 2023–2025</div>", unsafe_allow_html=True)
with h2:
    k1,k2,k3 = st.columns(3)
    k1.metric("Athletes", len(filtered))
    k2.metric("Pitchers", int((filtered["position"].isin(PITCHER_POSITIONS)).sum()))
    k3.metric("w/ Sprint", int(filtered["sprint_30"].notna().sum()))

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_board, tab_profile, tab_scatter = st.tabs(["RANKINGS", "PLAYER PROFILE", "SCATTER"])

# ── TAB 1: RANKINGS ───────────────────────────────────────────────────────────
with tab_board:
    disp = filtered[[
        "name","position","year",
        "athletic_score","potential_score",
        "concentric_impulse","mrsi",
        "sprint_10","sprint_20","sprint_30",
        "height_in","weight_lb","wingspan_in",
        "ws_ht_ratio","bmi","rel_peak_power","proj_pct"
    ]].copy()

    disp.rename(columns={
        "name":"Athlete","position":"Position","year":"Year",
        "athletic_score":"Athletic","potential_score":"Potential",
        "concentric_impulse":"CI (Ns)","mrsi":"mRSI",
        "sprint_10":"10yd","sprint_20":"20yd","sprint_30":"30yd",
        "height_in":"Ht (in)","weight_lb":"Wt (lb)","wingspan_in":"Wingspan",
        "ws_ht_ratio":"WS:Ht","bmi":"BMI","rel_peak_power":"Rel Pwr",
        "proj_pct":"Projection %ile"
    }, inplace=True)

    for c in ["Athletic","Potential","CI (Ns)","mRSI","10yd","20yd","30yd",
              "Ht (in)","Wt (lb)","Wingspan","WS:Ht","BMI","Rel Pwr","Projection %ile"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(2)

    st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=560,
        column_config={
            "Athletic":    st.column_config.ProgressColumn("Athletic",    min_value=0, max_value=100, format="%.1f"),
            "Potential":   st.column_config.ProgressColumn("Potential",   min_value=0, max_value=100, format="%.1f"),
            "Projection %ile": st.column_config.ProgressColumn("Projection %ile", min_value=0, max_value=100, format="%.0f"),
        })

    csv = disp.to_csv(index=False).encode()
    st.download_button("↓ Export CSV", csv, "draft_scout.csv", "text/csv")

# ── TAB 2: PLAYER PROFILE ─────────────────────────────────────────────────────
with tab_profile:
    player_names = filtered["name"].dropna().sort_values().tolist()
    if not player_names:
        st.warning("No athletes match filters.")
    else:
        search = st.text_input("Search", placeholder="Type a name...", label_visibility="collapsed")
        name_opts = [n for n in player_names if search.lower() in n.lower()] if search else player_names
        sel = st.selectbox("Athlete", name_opts, label_visibility="collapsed")
        r = filtered[filtered["name"] == sel].iloc[0]
        is_p = r["position"] in PITCHER_POSITIONS

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Name + meta
        n1, n2 = st.columns([3,1])
        with n1:
            pitcher_badge = "<span class='pitcher-badge'>⚾ Pitcher</span>" if is_p else ""
            st.markdown(f"<div class='player-name'>{r['name']}{pitcher_badge}</div>", unsafe_allow_html=True)
            yr = int(r['year']) if not pd.isna(r['year']) else '—'
            st.markdown(f"<div class='player-meta'>{r['position']} &nbsp;·&nbsp; {yr} Draft Class</div>", unsafe_allow_html=True)
        with n2:
            ath_g, ath_c = grade(r["athletic_score"])
            pot_g, pot_c = grade(r["potential_score"])
            st.markdown(f"""
            <div style='text-align:right'>
              <div style='font-size:0.6rem;color:#4a6572;letter-spacing:0.12em;text-transform:uppercase'>Athletic</div>
              <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:{ath_c}'>{fv(r["athletic_score"],".0f")} <span style='font-size:0.7rem'>{ath_g}</span></div>
              <div style='font-size:0.6rem;color:#4a6572;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.3rem'>Potential</div>
              <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:{pot_c}'>{fv(r["potential_score"],".0f")} <span style='font-size:0.7rem'>{pot_g}</span></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        # Gauges
        g1,g2 = st.columns(2)
        with g1: st.plotly_chart(gauge_fig(r["athletic_score"],"ATHLETIC SCORE",score_color(r["athletic_score"])), use_container_width=True, key="ga")
        with g2: st.plotly_chart(gauge_fig(r["potential_score"],"POTENTIAL SCORE",score_color(r["potential_score"])), use_container_width=True, key="gp")

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        # Main content: left metrics, right radar + bars
        left, right = st.columns([1.1, 0.9])

        with left:
            # Physical
            st.markdown("<div class='section-label'>Physical</div>", unsafe_allow_html=True)
            p1,p2,p3 = st.columns(3)
            with p1:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>Height</div><div class='stat-value'>{fmt_h(r['height_in'])}</div><div class='stat-sub'>{fv(r['h_pct'],'.0f')}th %ile</div></div>", unsafe_allow_html=True)
            with p2:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>Weight</div><div class='stat-value'>{fv(r['weight_lb'],'.0f')} lb</div><div class='stat-sub'>{fv(r['w_pct'],'.0f')}th %ile</div></div>", unsafe_allow_html=True)
            with p3:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>Wingspan</div><div class='stat-value'>{fmt_h(r['wingspan_in'])}</div><div class='stat-sub'>{fv(r['ws_pct'],'.0f')}th %ile</div></div>", unsafe_allow_html=True)

            # Derived metrics
            st.markdown("<div class='section-label'>Derived Metrics</div>", unsafe_allow_html=True)

            # Wingspan:Height ratio
            wsr = r["ws_ht_ratio"]
            wsr_interp = "Positive — wingspan exceeds height" if not pd.isna(wsr) and wsr > 1.0 else ("Neutral — wingspan ≈ height" if not pd.isna(wsr) and wsr >= 0.98 else "Below avg — limited arm length")
            wsr_color = "#2dd4a0" if not pd.isna(wsr) and wsr > 1.02 else ("#f0b429" if not pd.isna(wsr) and wsr >= 0.98 else "#e84545")
            st.markdown(f"""<div class='derived-pill'>
              <div><div class='derived-name'>Wingspan : Height Ratio</div><div class='derived-interp'>{wsr_interp}</div></div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:{wsr_color}'>{fv(wsr,".3f")}</div>
            </div>""", unsafe_allow_html=True)

            # BMI + projection
            bmi = r["bmi"]
            bmi_interp = "Lean frame — significant room to add mass" if not pd.isna(bmi) and bmi < 22 else ("Athletic build — moderate projection" if not pd.isna(bmi) and bmi < 25 else "Filled out — limited mass projection")
            bmi_color = "#2dd4a0" if not pd.isna(bmi) and bmi < 22 else ("#f0b429" if not pd.isna(bmi) and bmi < 25 else "#e84545")
            st.markdown(f"""<div class='derived-pill'>
              <div><div class='derived-name'>BMI (Mass Projection Index)</div><div class='derived-interp'>{bmi_interp}</div></div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:{bmi_color}'>{fv(bmi,".1f")}</div>
            </div>""", unsafe_allow_html=True)

            proj_g, proj_c = grade(r["proj_pct"])
            st.markdown(f"""<div class='derived-pill'>
              <div><div class='derived-name'>Physical Projection %ile</div><div class='derived-interp'>Height × leanness relative to class</div></div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:{proj_c}'>{fv(r["proj_pct"],".0f")}th <span style='font-size:0.7rem'>{proj_g}</span></div>
            </div>""", unsafe_allow_html=True)

            # Force plate
            st.markdown("<div class='section-label'>Force Plate — CMJ</div>", unsafe_allow_html=True)
            f1,f2,f3 = st.columns(3)
            with f1:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>Conc. Impulse</div><div class='stat-value'>{fv(r['concentric_impulse'],'.0f')}</div><div class='stat-sub'>Ns · {fv(r['ci_pct'],'.0f')}th %ile</div></div>", unsafe_allow_html=True)
            with f2:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>mRSI</div><div class='stat-value'>{fv(r['mrsi'],'.2f')}</div><div class='stat-sub'>m/s · {fv(r['mrsi_pct'],'.0f')}th %ile</div></div>", unsafe_allow_html=True)
            with f3:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>Rel. Peak Power</div><div class='stat-value'>{fv(r['rel_peak_power'],'.1f')}</div><div class='stat-sub'>W/kg · {fv(r['rpp_pct'],'.0f')}th %ile</div></div>", unsafe_allow_html=True)

            # Sprint
            st.markdown("<div class='section-label'>Sprint</div>", unsafe_allow_html=True)
            s1,s2,s3 = st.columns(3)
            with s1:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>10 Yard</div><div class='stat-value'>{fv(r['sprint_10'],'.3f')}</div><div class='stat-sub'>seconds</div></div>", unsafe_allow_html=True)
            with s2:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>20 Yard</div><div class='stat-value'>{fv(r['sprint_20'],'.3f')}</div><div class='stat-sub'>seconds</div></div>", unsafe_allow_html=True)
            with s3:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>30 Yard</div><div class='stat-value'>{fv(r['sprint_30'],'.3f')}</div><div class='stat-sub'>{fv(r['s30_pct'],'.0f')}th %ile</div></div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='section-label'>Percentile Profile</div>", unsafe_allow_html=True)

            if is_p:
                rl = ["CI","mRSI","Sprint","Height","Weight","Rel Pwr","WS:Ht"]
                rv = [r["ci_pct"],r["mrsi_pct"],r["s30_pct"],r["h_pct"],r["w_pct"],r["rpp_pct"],r["wsr_pct"]]
            else:
                rl = ["CI","mRSI","Sprint","Height","Weight","Rel Pwr"]
                rv = [r["ci_pct"],r["mrsi_pct"],r["s30_pct"],r["h_pct"],r["w_pct"],r["rpp_pct"]]

            st.plotly_chart(radar_fig(rl, rv), use_container_width=True, key="rd")

            st.markdown("<div class='section-label'>Percentile Bars</div>", unsafe_allow_html=True)
            full_bars = [
                ("Conc. Impulse", r["ci_pct"]),
                ("mRSI", r["mrsi_pct"]),
                ("Sprint 30yd", r["s30_pct"]),
                ("Height", r["h_pct"]),
                ("Weight", r["w_pct"]),
                ("Rel. Peak Pwr", r["rpp_pct"]),
                ("Wingspan", r["ws_pct"]),
                ("WS:Ht Ratio", r["wsr_pct"]),
                ("Projection", r["proj_pct"]),
            ]
            bars_html = "".join(pct_bar_html(lbl, pct) for lbl, pct in full_bars if not pd.isna(pct))
            st.markdown(bars_html, unsafe_allow_html=True)

# ── TAB 3: SCATTER ────────────────────────────────────────────────────────────
with tab_scatter:
    chart_df = filtered.dropna(subset=["athletic_score","potential_score"]).copy()
    if len(chart_df) < 2:
        st.info("Not enough scored players to chart.")
    else:
        fig = px.scatter(
            chart_df, x="athletic_score", y="potential_score",
            color="position", hover_name="name",
            hover_data={"concentric_impulse":True,"mrsi":True,"sprint_30":True,
                        "height_in":True,"weight_lb":True,"ws_ht_ratio":True,"bmi":True},
            labels={"athletic_score":"Athletic Score","potential_score":"Potential Score"},
            title="Athletic Score vs Physical Potential",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            paper_bgcolor="#080c10", plot_bgcolor="#0d1520",
            font=dict(family="DM Mono, monospace", color="#7eb8d4"),
            xaxis=dict(gridcolor="#1a2535", zerolinecolor="#1a2535"),
            yaxis=dict(gridcolor="#1a2535", zerolinecolor="#1a2535"),
            height=600,
        )
        fig.add_hline(y=50, line_dash="dot", line_color="#2a3a4a")
        fig.add_vline(x=50, line_dash="dot", line_color="#2a3a4a")
        st.plotly_chart(fig, use_container_width=True)
