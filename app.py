import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Draft Athletic Profile", layout="wide", page_icon="⚾")

# ── STYLES ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #0d1117; }
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 12px;
  }
  .metric-label { color: #8b949e; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }
  .metric-value { color: #e6edf3; font-size: 28px; font-weight: 700; }
  .metric-pct   { color: #8b949e; font-size: 12px; margin-top: 2px; }
  .score-ring-label { color: #8b949e; font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
  .player-header { color: #e6edf3; font-size: 26px; font-weight: 700; margin: 0; }
  .player-sub    { color: #8b949e; font-size: 14px; margin-top: 4px; }
  .section-header { color: #e6edf3; font-size: 14px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; border-bottom: 1px solid #30363d; padding-bottom: 8px; margin-bottom: 16px; }
  .stSelectbox label, .stMultiSelect label { color: #8b949e !important; font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
  div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    fp    = pd.read_excel("All Years.xlsx", sheet_name="Force Plate")
    sprint = pd.read_excel("All Years.xlsx", sheet_name="Sprint")
    anthro = pd.read_excel("All Years.xlsx", sheet_name="Anthropometrics")

    def parse_num(s):
        try: return float(str(s).replace("kg","").replace("cm","").strip())
        except: return np.nan

    anthro = anthro.copy()
    anthro["height_cm"]   = anthro["Height"].fillna(anthro["Stature Height 1"].apply(parse_num))
    anthro["weight_kg"]   = anthro["Body Weight (kg)"].fillna(anthro["Body Weight"]).fillna(anthro["Stature Body Weight 1"].apply(parse_num))
    anthro["wingspan_cm"] = anthro["Arm Span"].fillna(anthro["Stature Arm Span 1"].apply(parse_num))

    for df in [fp, anthro, sprint]:
        df["DPL ID"] = df["DPL ID"].astype(str).str.strip()

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
    merged["year"]     = merged["fp_year"].fillna(merged["anthro_year"]).fillna(merged["sprint_year"]).astype(float).astype("Int64")

    # Height/weight in imperial for display
    merged["height_in"] = merged["height_cm"] / 2.54
    merged["weight_lb"] = merged["weight_kg"] * 2.205
    merged["wingspan_in"] = merged["wingspan_cm"] / 2.54

    merged = merged[merged["name"].notna()].reset_index(drop=True)
    return merged

df = load_data()

PITCHER_POSITIONS = {"Starting Pitcher", "Relief Pitcher", "Right Hand Pitcher"}

# ── SCORING ───────────────────────────────────────────────────────────────────
def percentile_rank(series, value, lower_is_better=False):
    valid = series.dropna()
    if len(valid) < 2 or pd.isna(value):
        return np.nan
    pct = stats.percentileofscore(valid, value, kind="rank")
    return (100 - pct) if lower_is_better else pct

def compute_scores(row, df):
    is_pitcher = row["position"] in PITCHER_POSITIONS

    # ── Athletic Score: concentric impulse (40%), mRSI (35%), sprint_30 (25%)
    ci_pct    = percentile_rank(df["concentric_impulse"], row["concentric_impulse"])
    mrsi_pct  = percentile_rank(df["mrsi"], row["mrsi"])
    s30_pct   = percentile_rank(df["sprint_30"], row["sprint_30"], lower_is_better=True)

    ath_parts, ath_weights = [], []
    if not pd.isna(ci_pct):   ath_parts.append(ci_pct * 0.40); ath_weights.append(0.40)
    if not pd.isna(mrsi_pct): ath_parts.append(mrsi_pct * 0.35); ath_weights.append(0.35)
    if not pd.isna(s30_pct):  ath_parts.append(s30_pct * 0.25); ath_weights.append(0.25)
    athletic = (sum(ath_parts) / sum(ath_weights)) if ath_weights else np.nan

    # ── Potential Score
    h_pct    = percentile_rank(df["height_cm"], row["height_cm"])
    w_pct    = percentile_rank(df["weight_kg"], row["weight_kg"])
    rpp_pct  = percentile_rank(df["rel_peak_power"], row["rel_peak_power"])
    ws_pct   = percentile_rank(df["wingspan_cm"], row["wingspan_cm"])

    if is_pitcher:
        pot_map = [(h_pct, 0.20), (w_pct, 0.15), (mrsi_pct, 0.20), (rpp_pct, 0.20), (s30_pct, 0.15), (ws_pct, 0.10)]
    else:
        pot_map = [(h_pct, 0.20), (w_pct, 0.20), (mrsi_pct, 0.25), (rpp_pct, 0.20), (s30_pct, 0.15)]

    pot_parts, pot_weights = [], []
    for val, wt in pot_map:
        if not pd.isna(val):
            pot_parts.append(val * wt); pot_weights.append(wt)
    potential = (sum(pot_parts) / sum(pot_weights)) if pot_weights else np.nan

    return pd.Series({
        "athletic_score": athletic, "potential_score": potential,
        "ci_pct": ci_pct, "mrsi_pct": mrsi_pct, "s30_pct": s30_pct,
        "h_pct": h_pct, "w_pct": w_pct, "rpp_pct": rpp_pct, "ws_pct": ws_pct
    })

scores = df.apply(compute_scores, axis=1, df=df)
df = pd.concat([df, scores], axis=1)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def score_color(val):
    if pd.isna(val): return "#4d5566"
    if val >= 80: return "#3fb950"
    if val >= 60: return "#d29922"
    if val >= 40: return "#db6d28"
    return "#f85149"

def fmt_height(inches):
    if pd.isna(inches): return "—"
    ft = int(inches // 12); inch = inches % 12
    return f"{ft}'{inch:.0f}\""

def fmt_val(v, fmt=".2f", suffix=""):
    return f"{v:{fmt}}{suffix}" if not pd.isna(v) else "—"

def gauge_chart(score, title, color):
    if pd.isna(score): score_disp = 0; text = "N/A"
    else: score_disp = score; text = f"{score:.0f}"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_disp,
        number={"font": {"size": 36, "color": "#e6edf3", "family": "Inter"}, "suffix": ""},
        title={"text": title, "font": {"size": 13, "color": "#8b949e", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#8b949e", "size": 10}, "tickwidth": 1, "tickcolor": "#30363d"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#1a1f2e"},
                {"range": [40, 60], "color": "#1a2030"},
                {"range": [60, 80], "color": "#1a2820"},
                {"range": [80, 100],"color": "#1a3020"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": score_disp}
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}
    )
    return fig

def radar_chart(labels, values, title):
    vals = [v if not pd.isna(v) else 0 for v in values]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(56, 139, 253, 0.15)",
        line=dict(color="#388bfd", width=2),
        marker=dict(color="#388bfd", size=6)
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#8b949e", size=9), gridcolor="#30363d", linecolor="#30363d"),
            angularaxis=dict(tickfont=dict(color="#e6edf3", size=11), gridcolor="#30363d", linecolor="#30363d")
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=50, t=30, b=30),
        height=300,
        font={"family": "Inter"}
    )
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ Draft Scout")
    st.markdown("---")

    years = sorted(df["year"].dropna().unique().astype(int).tolist())
    sel_years = st.multiselect("Draft Class", years, default=years)

    positions = sorted(df["position"].dropna().unique().tolist())
    sel_positions = st.multiselect("Position", positions, default=positions)

    st.markdown("---")
    min_ath = st.slider("Min Athletic Score", 0, 100, 0)
    min_pot = st.slider("Min Potential Score", 0, 100, 0)

    st.markdown("---")
    sort_by = st.selectbox("Sort Leaderboard By", ["Athletic Score", "Potential Score", "30yd Sprint", "mRSI", "Concentric Impulse"])

# ── FILTER ────────────────────────────────────────────────────────────────────
filtered = df[
    df["year"].isin(sel_years) &
    df["position"].isin(sel_positions) &
    (df["athletic_score"].fillna(0) >= min_ath) &
    (df["potential_score"].fillna(0) >= min_pot)
].copy()

sort_col_map = {
    "Athletic Score": ("athletic_score", False),
    "Potential Score": ("potential_score", False),
    "30yd Sprint": ("sprint_30", True),
    "mRSI": ("mrsi", False),
    "Concentric Impulse": ("concentric_impulse", False),
}
sc, sa = sort_col_map[sort_by]
filtered = filtered.sort_values(sc, ascending=sa, na_position="last")

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
st.markdown("# Draft Athletic Profiles")
st.markdown(f"<span style='color:#8b949e'>{len(filtered)} athletes · {len(sel_years)} draft class(es)</span>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["📋 Leaderboard", "🔍 Player Profile"])

# ── TAB 1: LEADERBOARD ────────────────────────────────────────────────────────
with tab1:
    cols_display = ["name", "position", "year", "athletic_score", "potential_score",
                    "concentric_impulse", "mrsi", "sprint_10", "sprint_20", "sprint_30",
                    "height_in", "weight_lb", "rel_peak_power", "wingspan_in"]
    cols_present = [c for c in cols_display if c in filtered.columns]
    disp = filtered[cols_present].copy()

    disp.rename(columns={
        "name": "Athlete", "position": "Position", "year": "Year",
        "athletic_score": "Athletic Score", "potential_score": "Potential Score",
        "concentric_impulse": "Conc. Impulse (Ns)", "mrsi": "mRSI (m/s)",
        "sprint_10": "10yd (s)", "sprint_20": "20yd (s)", "sprint_30": "30yd (s)",
        "height_in": "Height (in)", "weight_lb": "Weight (lb)",
        "rel_peak_power": "Rel. Peak Power (W/kg)", "wingspan_in": "Wingspan (in)"
    }, inplace=True)

    # Round numerics
    for c in ["Athletic Score", "Potential Score", "Conc. Impulse (Ns)", "mRSI (m/s)",
              "10yd (s)", "20yd (s)", "30yd (s)", "Height (in)", "Weight (lb)",
              "Rel. Peak Power (W/kg)", "Wingspan (in)"]:
        if c in disp.columns:
            disp[c] = disp[c].round(2)

    st.dataframe(
        disp.reset_index(drop=True),
        use_container_width=True,
        height=500,
        column_config={
            "Athletic Score": st.column_config.ProgressColumn("Athletic Score", min_value=0, max_value=100, format="%.1f"),
            "Potential Score": st.column_config.ProgressColumn("Potential Score", min_value=0, max_value=100, format="%.1f"),
        }
    )

# ── TAB 2: PLAYER PROFILE ─────────────────────────────────────────────────────
with tab2:
    player_names = filtered["name"].dropna().sort_values().tolist()
    if not player_names:
        st.warning("No athletes match the current filters.")
    else:
        sel_name = st.selectbox("Select Athlete", player_names)
        row = filtered[filtered["name"] == sel_name].iloc[0]
        is_pitcher = row["position"] in PITCHER_POSITIONS

        # Header
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"<p class='player-header'>{row['name']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='player-sub'>{row['position']} · {int(row['year']) if not pd.isna(row['year']) else '—'} Draft Class</p>", unsafe_allow_html=True)
        with c2:
            if is_pitcher:
                st.markdown("<div style='text-align:right;color:#388bfd;font-size:12px;font-weight:600;letter-spacing:0.06em;padding-top:8px'>⚾ PITCHER — WINGSPAN INCLUDED</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Score gauges
        g1, g2 = st.columns(2)
        with g1:
            color_a = score_color(row["athletic_score"])
            st.plotly_chart(gauge_chart(row["athletic_score"], "ATHLETIC SCORE", color_a), use_container_width=True, key="gauge_a")
        with g2:
            color_p = score_color(row["potential_score"])
            st.plotly_chart(gauge_chart(row["potential_score"], "POTENTIAL SCORE", color_p), use_container_width=True, key="gauge_p")

        st.markdown("---")

        # Two columns: raw metrics + radar
        left, right = st.columns([1, 1])

        with left:
            st.markdown("<p class='section-header'>Physical Profile</p>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class='metric-card'><div class='metric-label'>Height</div>
                <div class='metric-value'>{fmt_height(row['height_in'])}</div>
                <div class='metric-pct'>{fmt_val(row['h_pct'],'.0f','th %ile')}</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class='metric-card'><div class='metric-label'>Weight</div>
                <div class='metric-value'>{fmt_val(row['weight_lb'],'.0f',' lb')}</div>
                <div class='metric-pct'>{fmt_val(row['w_pct'],'.0f','th %ile')}</div></div>""", unsafe_allow_html=True)
            with m3:
                ws_show = fmt_height(row['wingspan_in']) if not pd.isna(row['wingspan_in']) else '—'
                ws_pct  = fmt_val(row['ws_pct'],'.0f','th %ile') if is_pitcher else "—"
                st.markdown(f"""<div class='metric-card'><div class='metric-label'>Wingspan</div>
                <div class='metric-value'>{ws_show}</div>
                <div class='metric-pct'>{ws_pct}</div></div>""", unsafe_allow_html=True)

            st.markdown("<p class='section-header' style='margin-top:20px'>Force Plate (CMJ)</p>", unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown(f"""<div class='metric-card'><div class='metric-label'>Conc. Impulse</div>
                <div class='metric-value'>{fmt_val(row['concentric_impulse'],'.1f')}</div>
                <div class='metric-pct'>Ns · {fmt_val(row['ci_pct'],'.0f','th %ile')}</div></div>""", unsafe_allow_html=True)
            with f2:
                st.markdown(f"""<div class='metric-card'><div class='metric-label'>mRSI</div>
                <div class='metric-value'>{fmt_val(row['mrsi'],'.2f')}</div>
                <div class='metric-pct'>m/s · {fmt_val(row['mrsi_pct'],'.0f','th %ile')}</div></div>""", unsafe_allow_html=True)
            with f3:
                st.markdown(f"""<div class='metric-card'><div class='metric-label'>Rel. Peak Power</div>
                <div class='metric-value'>{fmt_val(row['rel_peak_power'],'.1f')}</div>
                <div class='metric-pct'>W/kg · {fmt_val(row['rpp_pct'],'.0f','th %ile')}</div></div>""", unsafe_allow_html=True)

            st.markdown("<p class='section-header' style='margin-top:20px'>Sprint</p>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f"""<div class='metric-card'><div class='metric-label'>10 Yard</div>
                <div class='metric-value'>{fmt_val(row['sprint_10'],'.3f')}</div>
                <div class='metric-pct'>seconds</div></div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""<div class='metric-card'><div class='metric-label'>20 Yard</div>
                <div class='metric-value'>{fmt_val(row['sprint_20'],'.3f')}</div>
                <div class='metric-pct'>seconds</div></div>""", unsafe_allow_html=True)
            with s3:
                st.markdown(f"""<div class='metric-card'><div class='metric-label'>30 Yard</div>
                <div class='metric-value'>{fmt_val(row['sprint_30'],'.3f')}</div>
                <div class='metric-pct'>seconds · {fmt_val(row['s30_pct'],'.0f','th %ile')}</div></div>""", unsafe_allow_html=True)

        with right:
            st.markdown("<p class='section-header'>Percentile Breakdown</p>", unsafe_allow_html=True)
            if is_pitcher:
                radar_labels = ["Conc. Impulse", "mRSI", "Sprint (30yd)", "Height", "Weight", "Rel. Power", "Wingspan"]
                radar_values = [row["ci_pct"], row["mrsi_pct"], row["s30_pct"], row["h_pct"], row["w_pct"], row["rpp_pct"], row["ws_pct"]]
            else:
                radar_labels = ["Conc. Impulse", "mRSI", "Sprint (30yd)", "Height", "Weight", "Rel. Power"]
                radar_values = [row["ci_pct"], row["mrsi_pct"], row["s30_pct"], row["h_pct"], row["w_pct"], row["rpp_pct"]]

            st.plotly_chart(radar_chart(radar_labels, radar_values, ""), use_container_width=True, key="radar")

            # Percentile bars
            st.markdown("<p class='section-header' style='margin-top:8px'>Percentile Bars</p>", unsafe_allow_html=True)
            bar_items = list(zip(radar_labels, radar_values))
            for label, pct in bar_items:
                if pd.isna(pct): continue
                color = score_color(pct)
                st.markdown(f"""
                <div style='margin-bottom:10px'>
                  <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                    <span style='color:#8b949e;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em'>{label}</span>
                    <span style='color:#e6edf3;font-size:11px;font-weight:700'>{pct:.0f}th</span>
                  </div>
                  <div style='background:#30363d;border-radius:4px;height:6px'>
                    <div style='background:{color};width:{pct}%;height:6px;border-radius:4px'></div>
                  </div>
                </div>""", unsafe_allow_html=True)
