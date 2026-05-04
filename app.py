import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import gspread
import json

st.set_page_config(page_title="Draft Athletic Profile", layout="wide", page_icon="⚾")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .metric-card { background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;text-align:center;margin-bottom:12px; }
  .metric-label { color:#8b949e;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px; }
  .metric-value { color:#e6edf3;font-size:28px;font-weight:700; }
  .metric-pct { color:#8b949e;font-size:12px;margin-top:2px; }
  .player-header { color:#e6edf3;font-size:26px;font-weight:700;margin:0; }
  .player-sub { color:#8b949e;font-size:14px;margin-top:4px; }
  .section-header { color:#e6edf3;font-size:14px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid #30363d;padding-bottom:8px;margin-bottom:16px; }
  div[data-testid="stSidebar"] { background:#0d1117;border-right:1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

SPREADSHEET_ID = "1J27zw_UngoTNdq6VKPF6RhB8aqfmvlpX60GtrXjOsbs"
PITCHER_POSITIONS = {"Starting Pitcher", "Relief Pitcher", "Right Hand Pitcher"}

from google.oauth2.service_account import Credentials

@st.cache_resource(show_spinner=False)
def get_gspread_client():
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(ttl=300, show_spinner=False)
def load_worksheet(sheet_id, worksheet_name):
    client = get_gspread_client()
    spreadsheet = client.open_by_key(sheet_id)
    worksheet = spreadsheet.worksheet(worksheet_name)
    records = worksheet.get_all_records()
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
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def parse_unit(s):
        try: return float(str(s).replace("kg","").replace("cm","").strip())
        except: return np.nan

    sprint = to_num(sprint, ["10yd","20yd","30yd","Year"])
    sprint["DPL ID"] = sprint["DPL ID"].astype(str).str.strip()

    fp = to_num(fp, ["Concentric Impulse [Ns]","RSI-Modified [m/s]","Peak Power [W]","Peak Power / BM [W/kg]","Year"])
    fp["DPL ID"] = fp["DPL ID"].astype(str).str.strip()
    if "Test Type" in fp.columns:
        fp = fp[fp["Test Type"] == "CMJ"]

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

    merged = fp_best.merge(anthro_best, on="DPL ID", how="outer").merge(sprint_best, on="DPL ID", how="outer")
    merged["name"]     = merged["fp_name"].fillna(merged["anthro_name"]).fillna(merged["sprint_name"])
    merged["position"] = merged["fp_position"].fillna(merged["anthro_position"])
    merged["year"]     = pd.to_numeric(merged["fp_year"].fillna(merged["anthro_year"]).fillna(merged["sprint_year"]), errors="coerce").astype("Int64")
    merged["height_in"]   = merged["height_cm"] / 2.54
    merged["weight_lb"]   = merged["weight_kg"] * 2.205
    merged["wingspan_in"] = merged["wingspan_cm"] / 2.54

    return merged[merged["name"].notna()].reset_index(drop=True)

def pct_rank(series, value, lower_is_better=False):
    valid = series.dropna()
    if len(valid) < 2 or pd.isna(value): return np.nan
    p = stats.percentileofscore(valid, value, kind="rank")
    return (100-p) if lower_is_better else p

def compute_scores(df):
    rows = []
    for _, r in df.iterrows():
        is_p  = r["position"] in PITCHER_POSITIONS
        ci_p  = pct_rank(df["concentric_impulse"], r["concentric_impulse"])
        mr_p  = pct_rank(df["mrsi"], r["mrsi"])
        s30_p = pct_rank(df["sprint_30"], r["sprint_30"], True)
        h_p   = pct_rank(df["height_cm"], r["height_cm"])
        w_p   = pct_rank(df["weight_kg"], r["weight_kg"])
        rp_p  = pct_rank(df["rel_peak_power"], r["rel_peak_power"])
        ws_p  = pct_rank(df["wingspan_cm"], r["wingspan_cm"])

        ap, aw = [], []
        if not pd.isna(ci_p):  ap.append(ci_p*0.40);  aw.append(0.40)
        if not pd.isna(mr_p):  ap.append(mr_p*0.35);  aw.append(0.35)
        if not pd.isna(s30_p): ap.append(s30_p*0.25); aw.append(0.25)
        ath = (sum(ap)/sum(aw)) if aw else np.nan

        pm = [(h_p,0.20),(w_p,0.15),(mr_p,0.20),(rp_p,0.20),(s30_p,0.15),(ws_p,0.10)] if is_p \
             else [(h_p,0.20),(w_p,0.20),(mr_p,0.25),(rp_p,0.20),(s30_p,0.15)]
        pp, pw = [], []
        for v, wt in pm:
            if not pd.isna(v): pp.append(v*wt); pw.append(wt)
        pot = (sum(pp)/sum(pw)) if pw else np.nan

        rows.append({"athletic_score":ath,"potential_score":pot,
                     "ci_pct":ci_p,"mrsi_pct":mr_p,"s30_pct":s30_p,
                     "h_pct":h_p,"w_pct":w_p,"rpp_pct":rp_p,"ws_pct":ws_p})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

def score_color(val):
    if pd.isna(val): return "#4d5566"
    if val >= 80: return "#3fb950"
    if val >= 60: return "#d29922"
    if val >= 40: return "#db6d28"
    return "#f85149"

def fmt_height(inches):
    if pd.isna(inches): return "—"
    return f"{int(inches//12)}'{inches%12:.0f}\""

def fmt_val(v, fmt=".2f", suffix=""):
    return f"{v:{fmt}}{suffix}" if not pd.isna(v) else "—"

def gauge(score, title, color):
    sd = 0 if pd.isna(score) else score
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=sd,
        number={"font":{"size":36,"color":"#e6edf3","family":"Inter"}},
        title={"text":title,"font":{"size":13,"color":"#8b949e","family":"Inter"}},
        gauge={"axis":{"range":[0,100],"tickfont":{"color":"#8b949e","size":10}},
               "bar":{"color":color,"thickness":0.25},"bgcolor":"#161b22","borderwidth":0,
               "steps":[{"range":[0,40],"color":"#1a1f2e"},{"range":[40,60],"color":"#1a2030"},
                        {"range":[60,80],"color":"#1a2820"},{"range":[80,100],"color":"#1a3020"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":sd}}
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family":"Inter"})
    return fig

def radar(labels, values):
    vals = [v if not pd.isna(v) else 0 for v in values]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=labels+[labels[0]], fill="toself",
        fillcolor="rgba(56,139,253,0.15)",
        line=dict(color="#388bfd",width=2), marker=dict(color="#388bfd",size=6)
    ))
    fig.update_layout(
        polar=dict(bgcolor="#161b22",
                   radialaxis=dict(visible=True,range=[0,100],tickfont=dict(color="#8b949e",size=9),gridcolor="#30363d",linecolor="#30363d"),
                   angularaxis=dict(tickfont=dict(color="#e6edf3",size=11),gridcolor="#30363d",linecolor="#30363d")),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50,r=50,t=30,b=30), height=300, font={"family":"Inter"}
    )
    return fig

# ── LOAD ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data from Google Sheets..."):
    df = compute_scores(load_data())

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ Draft Scout")
    st.markdown("---")
    years = sorted(df["year"].dropna().unique().tolist())
    sel_years = st.multiselect("Draft Class", years, default=years)
    positions = sorted(df["position"].dropna().unique().tolist())
    sel_positions = st.multiselect("Position", positions, default=positions)
    st.markdown("---")
    min_ath = st.slider("Min Athletic Score", 0, 100, 0)
    min_pot = st.slider("Min Potential Score", 0, 100, 0)
    st.markdown("---")
    sort_by = st.selectbox("Sort By", ["Athletic Score","Potential Score","30yd Sprint","mRSI","Concentric Impulse"])
    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── FILTER ────────────────────────────────────────────────────────────────────
filtered = df[
    df["year"].isin(sel_years) &
    df["position"].isin(sel_positions) &
    (df["athletic_score"].fillna(0) >= min_ath) &
    (df["potential_score"].fillna(0) >= min_pot)
].copy()

sc, sa = {"Athletic Score":("athletic_score",False),"Potential Score":("potential_score",False),
          "30yd Sprint":("sprint_30",True),"mRSI":("mrsi",False),
          "Concentric Impulse":("concentric_impulse",False)}[sort_by]
filtered = filtered.sort_values(sc, ascending=sa, na_position="last")

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("# Draft Athletic Profiles")
st.markdown(f"<span style='color:#8b949e'>{len(filtered)} athletes · {len(sel_years)} draft class(es)</span>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["📋 Leaderboard", "🔍 Player Profile"])

with tab1:
    disp = filtered[["name","position","year","athletic_score","potential_score",
                      "concentric_impulse","mrsi","sprint_10","sprint_20","sprint_30",
                      "height_in","weight_lb","rel_peak_power","wingspan_in"]].copy()
    disp.rename(columns={"name":"Athlete","position":"Position","year":"Year",
                         "athletic_score":"Athletic Score","potential_score":"Potential Score",
                         "concentric_impulse":"Conc. Impulse (Ns)","mrsi":"mRSI (m/s)",
                         "sprint_10":"10yd (s)","sprint_20":"20yd (s)","sprint_30":"30yd (s)",
                         "height_in":"Height (in)","weight_lb":"Weight (lb)",
                         "rel_peak_power":"Rel. Peak Power (W/kg)","wingspan_in":"Wingspan (in)"}, inplace=True)
    for c in ["Athletic Score","Potential Score","Conc. Impulse (Ns)","mRSI (m/s)",
              "10yd (s)","20yd (s)","30yd (s)","Height (in)","Weight (lb)",
              "Rel. Peak Power (W/kg)","Wingspan (in)"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(2)
    st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=500,
        column_config={
            "Athletic Score":  st.column_config.ProgressColumn("Athletic Score",  min_value=0, max_value=100, format="%.1f"),
            "Potential Score": st.column_config.ProgressColumn("Potential Score", min_value=0, max_value=100, format="%.1f"),
        })

with tab2:
    player_names = filtered["name"].dropna().sort_values().tolist()
    if not player_names:
        st.warning("No athletes match the current filters.")
    else:
        sel_name = st.selectbox("Select Athlete", player_names)
        row = filtered[filtered["name"] == sel_name].iloc[0]
        is_p = row["position"] in PITCHER_POSITIONS

        c1, c2 = st.columns([3,1])
        with c1:
            st.markdown(f"<p class='player-header'>{row['name']}</p>", unsafe_allow_html=True)
            yr = int(row['year']) if not pd.isna(row['year']) else '—'
            st.markdown(f"<p class='player-sub'>{row['position']} · {yr} Draft Class</p>", unsafe_allow_html=True)
        with c2:
            if is_p:
                st.markdown("<div style='text-align:right;color:#388bfd;font-size:12px;font-weight:600;padding-top:8px'>⚾ PITCHER — WINGSPAN INCLUDED</div>", unsafe_allow_html=True)

        st.markdown("---")
        g1, g2 = st.columns(2)
        with g1: st.plotly_chart(gauge(row["athletic_score"], "ATHLETIC SCORE", score_color(row["athletic_score"])), use_container_width=True, key="ga")
        with g2: st.plotly_chart(gauge(row["potential_score"], "POTENTIAL SCORE", score_color(row["potential_score"])), use_container_width=True, key="gp")

        st.markdown("---")
        left, right = st.columns(2)

        with left:
            st.markdown("<p class='section-header'>Physical Profile</p>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Height</div><div class='metric-value'>{fmt_height(row['height_in'])}</div><div class='metric-pct'>{fmt_val(row['h_pct'],'.0f','th %ile')}</div></div>", unsafe_allow_html=True)
            with m2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Weight</div><div class='metric-value'>{fmt_val(row['weight_lb'],'.0f',' lb')}</div><div class='metric-pct'>{fmt_val(row['w_pct'],'.0f','th %ile')}</div></div>", unsafe_allow_html=True)
            with m3:
                ws_show = fmt_height(row['wingspan_in']) if not pd.isna(row['wingspan_in']) else '—'
                ws_pct  = fmt_val(row['ws_pct'],'.0f','th %ile') if is_p else "—"
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Wingspan</div><div class='metric-value'>{ws_show}</div><div class='metric-pct'>{ws_pct}</div></div>", unsafe_allow_html=True)

            st.markdown("<p class='section-header' style='margin-top:20px'>Force Plate (CMJ)</p>", unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            with f1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Conc. Impulse</div><div class='metric-value'>{fmt_val(row['concentric_impulse'],'.1f')}</div><div class='metric-pct'>Ns · {fmt_val(row['ci_pct'],'.0f','th %ile')}</div></div>", unsafe_allow_html=True)
            with f2: st.markdown(f"<div class='metric-card'><div class='metric-label'>mRSI</div><div class='metric-value'>{fmt_val(row['mrsi'],'.2f')}</div><div class='metric-pct'>m/s · {fmt_val(row['mrsi_pct'],'.0f','th %ile')}</div></div>", unsafe_allow_html=True)
            with f3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Rel. Peak Power</div><div class='metric-value'>{fmt_val(row['rel_peak_power'],'.1f')}</div><div class='metric-pct'>W/kg · {fmt_val(row['rpp_pct'],'.0f','th %ile')}</div></div>", unsafe_allow_html=True)

            st.markdown("<p class='section-header' style='margin-top:20px'>Sprint</p>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1: st.markdown(f"<div class='metric-card'><div class='metric-label'>10 Yard</div><div class='metric-value'>{fmt_val(row['sprint_10'],'.3f')}</div><div class='metric-pct'>seconds</div></div>", unsafe_allow_html=True)
            with s2: st.markdown(f"<div class='metric-card'><div class='metric-label'>20 Yard</div><div class='metric-value'>{fmt_val(row['sprint_20'],'.3f')}</div><div class='metric-pct'>seconds</div></div>", unsafe_allow_html=True)
            with s3: st.markdown(f"<div class='metric-card'><div class='metric-label'>30 Yard</div><div class='metric-value'>{fmt_val(row['sprint_30'],'.3f')}</div><div class='metric-pct'>seconds · {fmt_val(row['s30_pct'],'.0f','th %ile')}</div></div>", unsafe_allow_html=True)

        with right:
            st.markdown("<p class='section-header'>Percentile Breakdown</p>", unsafe_allow_html=True)
            if is_p:
                rl = ["Conc. Impulse","mRSI","Sprint (30yd)","Height","Weight","Rel. Power","Wingspan"]
                rv = [row["ci_pct"],row["mrsi_pct"],row["s30_pct"],row["h_pct"],row["w_pct"],row["rpp_pct"],row["ws_pct"]]
            else:
                rl = ["Conc. Impulse","mRSI","Sprint (30yd)","Height","Weight","Rel. Power"]
                rv = [row["ci_pct"],row["mrsi_pct"],row["s30_pct"],row["h_pct"],row["w_pct"],row["rpp_pct"]]
            st.plotly_chart(radar(rl, rv), use_container_width=True, key="rd")

            st.markdown("<p class='section-header' style='margin-top:8px'>Percentile Bars</p>", unsafe_allow_html=True)
            for label, pct in zip(rl, rv):
                if pd.isna(pct): continue
                color = score_color(pct)
                st.markdown(f"""<div style='margin-bottom:10px'>
                  <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                    <span style='color:#8b949e;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em'>{label}</span>
                    <span style='color:#e6edf3;font-size:11px;font-weight:700'>{pct:.0f}th</span>
                  </div>
                  <div style='background:#30363d;border-radius:4px;height:6px'>
                    <div style='background:{color};width:{pct}%;height:6px;border-radius:4px'></div>
                  </div></div>""", unsafe_allow_html=True)
