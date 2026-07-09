"""
app.py — APT Attribution Engine · Home Page
MITRE ATT&CK Orange Dark Theme
Run: streamlit run app.py
"""

import json, pathlib, re, warnings
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import joblib
from inference import predict_top3

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="APT Attribution Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT      = pathlib.Path(__file__).parent.resolve()
ARTIFACTS = ROOT / "artifacts"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    ev = json.loads((ARTIFACTS/"eval_summary.json").read_text()) if (ARTIFACTS/"eval_summary.json").exists() else {}
    fv = joblib.load(ARTIFACTS/"feature_vocab.joblib") if (ARTIFACTS/"feature_vocab.joblib").exists() else {}
    gc = json.loads((ARTIFACTS/"group_tech_counts.json").read_text()) if (ARTIFACTS/"group_tech_counts.json").exists() else {}
    return ev, fv, gc

eval_summary, feature_vocab, group_counts = load_artifacts()

# ─── APT group → country/region mapping ─────────────────────────────────────
APT_COUNTRY = {
    # Russia
    "APT28":"Russia","APT29":"Russia","Sandworm Team":"Russia","Turla":"Russia",
    "Gamaredon Group":"Russia","HAFNIUM":"Russia","Cozy Bear":"Russia",
    "Fancy Bear":"Russia","Ember Bear":"Russia","IRON TILDEN":"Russia",
    "Dragonfly":"Russia","Energetic Bear":"Russia","Berserk Bear":"Russia",
    # China
    "APT1":"China","APT10":"China","APT17":"China","APT19":"China",
    "APT40":"China","APT41":"China","APT3":"China","APT30":"China",
    "Winnti Group":"China","BRONZE BUTLER":"China","Ke3chang":"China",
    "Leviathan":"China","Mustang Panda":"China","TA413":"China",
    "Threat Group-3390":"China","menuPass":"China","Deep Panda":"China",
    "Elderwood":"China","GALLIUM":"China","TEMP.Veles":"Russia",
    # North Korea
    "Lazarus Group":"North Korea","Kimsuky":"North Korea","APT37":"North Korea",
    "APT38":"North Korea","TEMP.Hermit":"North Korea","Andariel":"North Korea",
    "BlueNoroff":"North Korea","DarkHotel":"North Korea",
    # Iran
    "APT33":"Iran","APT34":"Iran","APT35":"Iran","APT39":"Iran",
    "Charming Kitten":"Iran","MuddyWater":"Iran","OilRig":"Iran",
    "Magic Hound":"Iran","HEXANE":"Iran","Agrius":"Iran","Pioneer Kitten":"Iran",
    # USA / Five Eyes (cybercriminal or state-adjacent)
    "Equation Group":"USA","Longhorn":"USA","The Lamberts":"USA",
    # Vietnam
    "APT32":"Vietnam","OceanLotus":"Vietnam","APT-C-00":"Vietnam",
    # Pakistan
    "Sidewinder":"Pakistan","Transparent Tribe":"Pakistan",
    # India
    "Donot Team":"India",
    # Turkey
    "Sea Turtle":"Turkey","PROMETHIUM":"Turkey",
    # Lebanon
    "Dark Caracal":"Lebanon",
    # Gaza / Palestinian
    "Gaza Cybergang":"Palestinian Territory","Molerats":"Palestinian Territory",
    # Multiple / Unknown
    "FIN7":"Unknown","FIN6":"Unknown","FIN4":"Unknown","Carbanak":"Unknown",
    "Cobalt Group":"Unknown","TA505":"Unknown","Silence":"Unknown",
    "DarkHydrus":"Unknown","Gorgon Group":"Unknown","CopyKittens":"Unknown",
}

COUNTRY_COORDS = {
    "Russia":               (61.52, 105.32),
    "China":                (35.86, 104.19),
    "North Korea":          (40.34, 127.51),
    "Iran":                 (32.43, 53.69),
    "USA":                  (37.09, -95.71),
    "Vietnam":              (14.06, 108.28),
    "Pakistan":             (30.38, 69.35),
    "India":                (20.59, 78.96),
    "Turkey":               (38.96, 35.24),
    "Lebanon":              (33.85, 35.86),
    "Palestinian Territory":(31.95, 35.23),
    "Unknown":              (0.0,   0.0),
}

COUNTRY_COLOR = {
    "Russia":"#E8451A","China":"#FF6B1A","North Korea":"#D84315",
    "Iran":"#F4511E","USA":"#BF360C","Vietnam":"#FF7043",
    "Pakistan":"#FFAB40","India":"#FFD180","Turkey":"#FF9800",
    "Lebanon":"#FFB74D","Palestinian Territory":"#FFCC02","Unknown":"#555",
}

PRESETS = {
    "APT28 — Fancy Bear (Russia)":    ["T1059","T1078","T1566","T1053","T1027","T1105","T1036","T1016","T1057","T1083"],
    "Lazarus Group (North Korea)":    ["T1059","T1003","T1071","T1041","T1105","T1547","T1055","T1070","T1140","T1486"],
    "APT29 — Cozy Bear (Russia)":     ["T1078","T1566","T1027","T1036","T1071","T1090","T1102","T1560","T1048","T1070"],
    "APT41 — Winnti (China)":         ["T1059","T1078","T1021","T1047","T1053","T1027","T1036","T1055","T1070","T1112"],
    "MuddyWater (Iran)":              ["T1059","T1036","T1027","T1105","T1070","T1047","T1053","T1016","T1057","T1083"],
    "Minimal incident — 2 techniques":["T1059","T1003"],
}

RANK_COLORS = ["#FF6B1A","#FF9800","#FFAB40"]
RANK_BG     = ["rgba(255,107,26,0.10)","rgba(255,152,0,0.08)","rgba(255,171,64,0.06)"]
RANK_MEDALS = ["","",""]

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #0D0D0D !important;
    color: #E8E0D8 !important;
}
section[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #1E1E1E !important;
}
section[data-testid="stSidebar"] * { color: #E8E0D8 !important; }

/* Banner */
.banner {
    background: linear-gradient(135deg, #111 0%, #1A0E00 60%, #200A00 100%);
    border: 1px solid #FF6B1A33;
    border-radius: 12px;
    padding: 28px 32px 24px;
    margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.banner::before {
    content:"";
    position:absolute; inset:0;
    background: repeating-linear-gradient(
        90deg, transparent, transparent 40px,
        rgba(255,107,26,0.03) 40px, rgba(255,107,26,0.03) 41px
    );
    pointer-events:none;
}
.banner-eye {
    font-family:'IBM Plex Mono',monospace;
    font-size:10px; letter-spacing:.18em; color:#FF6B1A;
    text-transform:uppercase; margin-bottom:8px;
}
.banner-title {
    font-size:30px; font-weight:700; color:#FFF;
    letter-spacing:-0.5px; margin-bottom:4px;
}
.banner-sub {
    font-size:12px; color:#8A7A6A;
    font-family:'IBM Plex Mono',monospace;
}
.metric-row { display:flex; gap:10px; margin-top:20px; flex-wrap:wrap; }
.mpill {
    background:rgba(255,107,26,0.07);
    border:1px solid rgba(255,107,26,0.2);
    border-radius:8px; padding:10px 18px; min-width:100px;
}
.mpill .val {
    font-family:'IBM Plex Mono',monospace;
    font-size:22px; font-weight:600; color:#FF6B1A;
}
.mpill .lbl {
    font-size:10px; color:#6A5A4A;
    text-transform:uppercase; letter-spacing:.1em; margin-top:2px;
}

/* Section labels */
.sec-lbl {
    font-family:'IBM Plex Mono',monospace;
    font-size:10px; font-weight:600; letter-spacing:.15em;
    text-transform:uppercase; color:#FF6B1A; margin-bottom:10px;
}

/* Result cards */
.rcard {
    border-radius:10px; padding:16px 20px;
    margin-bottom:10px; border-left:4px solid;
}
.rcard .rbadge {
    font-family:'IBM Plex Mono',monospace;
    font-size:9px; letter-spacing:.15em; text-transform:uppercase;
    opacity:.5; margin-bottom:4px; color:#E8E0D8;
}
.rcard .rname { font-size:17px; font-weight:600; color:#FFF; margin-bottom:2px; }
.rcard .rconf {
    font-family:'IBM Plex Mono',monospace;
    font-size:11px; color:#8A7A6A; margin-top:4px;
}
.rcard .rbar-bg {
    height:3px; background:rgba(255,255,255,0.07);
    border-radius:2px; margin-top:10px; overflow:hidden;
}
.rcard .rbar-fill { height:100%; border-radius:2px; }

/* Empty state */
.empty-state {
    border:1.5px dashed rgba(255,107,26,0.2);
    border-radius:10px; padding:48px 20px;
    text-align:center; color:rgba(255,107,26,0.35);
}

/* Sidebar nav */
.snav-lbl {
    font-family:'IBM Plex Mono',monospace;
    font-size:9px; font-weight:600; letter-spacing:.15em;
    text-transform:uppercase; color:#FF6B1A !important;
    margin-bottom:6px; margin-top:16px; display:block;
}

/* Buttons */
.stButton>button {
    background:#FF6B1A !important;
    color:#000 !important;
    font-weight:600 !important;
    border:none !important;
    border-radius:6px !important;
}
.stButton>button:hover { background:#FF8C42 !important; }

/* Input */
.stTextArea textarea {
    background:#161616 !important;
    border:1px solid #2A2A2A !important;
    color:#E8E0D8 !important;
    font-family:'IBM Plex Mono',monospace !important;
    font-size:13px !important;
    border-radius:8px !important;
}
.stTextArea textarea:focus { border-color:#FF6B1A !important; }

/* Selectbox */
.stSelectbox>div>div {
    background:#161616 !important;
    border:1px solid #2A2A2A !important;
    color:#E8E0D8 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background:#161616 !important;
    border:1px solid #2A2A2A !important;
    border-radius:8px !important;
    color:#E8E0D8 !important;
}
.streamlit-expanderContent {
    background:#111 !important;
    border:1px solid #2A2A2A !important;
    border-top:none !important;
}

/* Divider */
hr { border-color:#1E1E1E !important; }

/* About accordion */
.about-item {
    border:1px solid #222;
    border-radius:8px;
    margin-bottom:8px;
    overflow:hidden;
}
.about-item summary {
    padding:12px 16px;
    cursor:pointer;
    font-size:13px; font-weight:600;
    color:#E8E0D8;
    background:#161616;
    list-style:none;
    display:flex; align-items:center; gap:8px;
}
.about-item summary::-webkit-details-marker { display:none; }
.about-item summary::before { content:"▸"; color:#FF6B1A; font-size:10px; }
details[open] .about-item summary::before { content:"▾"; }
.about-item .body {
    padding:14px 16px; font-size:13px;
    color:#8A7A6A; background:#111; line-height:1.7;
}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px; border-bottom:1px solid #1E1E1E; margin-bottom:8px;'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#FF6B1A;letter-spacing:.15em;'>
         APT ATTRIBUTION
        </div>
        <div style='font-size:13px;font-weight:600;color:#FFF;margin-top:4px;'>
        Navigation
        </div>
    </div>
    """, unsafe_allow_html=True)
    

    
    st.markdown("""
<div style='display:flex;flex-direction:column;gap:6px;'>
  <a href='/' target='_self' style='display:block;padding:8px 12px;border-radius:6px;
     background:rgba(255,107,26,0.12);border:1px solid rgba(255,107,26,0.25);
     color:#FF6B1A;text-decoration:none;font-size:13px;font-weight:500;'>
     &nbsp; Home — Attribution</a>
  <a href='/analytics' target='_self' style='display:block;padding:8px 12px;border-radius:6px;
     background:rgba(255,255,255,0.04);border:1px solid #2A2A2A;
     color:#8A7A6A;text-decoration:none;font-size:13px;font-weight:500;'>
     &nbsp; Analytics &amp; Reports</a>
</div>""", unsafe_allow_html=True)

    st.markdown('<span class="snav-lbl">Preset Incidents</span>', unsafe_allow_html=True)
    preset_choice = st.selectbox(
        "preset", ["— enter manually —"] + list(PRESETS.keys()),
        label_visibility="collapsed",
    )

    st.markdown('<span class="snav-lbl"></span>', unsafe_allow_html=True)
    f1v = eval_summary.get("macro_f1", None)
    t3v = eval_summary.get("top3_accuracy", None)

# ─── BANNER ──────────────────────────────────────────────────────────────────
n_test = eval_summary.get("n_test_samples",    "—")
n_cls  = eval_summary.get("n_classes_in_test", "—")
f1_s   = f"{f1v:.3f}" if isinstance(f1v, float) else "—"
t3_s   = f"{t3v:.3f}" if isinstance(t3v, float) else "—"
n_grp  = len(group_counts) if group_counts else "—"

st.markdown(f"""
<div class="banner">
  <div class="banner-eye">MITRE ATT&CK · ML Attribution Engine · Enterprise v1.0</div>
  <div class="banner-title">APT Attribution Engine</div>
  <div class="banner-sub">XGBoost + Random Forest + SVM Ensemble &nbsp;·&nbsp; Optuna-tuned &nbsp;·&nbsp; SHAP Explainability</div>
  <div class="metric-row">
    <div class="mpill"><div class="val">{f1_s}</div><div class="lbl">Macro F1</div></div>
    <div class="mpill"><div class="val">{t3_s}</div><div class="lbl">Top-3 Acc</div></div>
    <div class="mpill"><div class="val">{n_grp}</div><div class="lbl">Tracked Groups</div></div>
    <div class="mpill"><div class="val">{n_test}</div><div class="lbl">Test Samples</div></div>
    <div class="mpill"><div class="val">{n_cls}</div><div class="lbl">Attributable</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── WORLD MAP ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl"> Global Threat Actor Map</div>', unsafe_allow_html=True)

# Build per-country aggregated data
country_groups = {}
for grp, cnt in group_counts.items():
    country = next((APT_COUNTRY[k] for k in APT_COUNTRY if k.lower() in grp.lower()), None)
    if not country:
        country = APT_COUNTRY.get(grp, "Unknown")
    if country not in country_groups:
        country_groups[country] = []
    country_groups[country].append((grp, cnt))

# One scatter point per group
lats, lons, texts, colors_list, sizes = [], [], [], [], []
for grp, cnt in group_counts.items():
    country = next((APT_COUNTRY[k] for k in APT_COUNTRY if k.lower() in grp.lower()), None)
    if not country:
        country = APT_COUNTRY.get(grp, "Unknown")
    if country == "Unknown":
        continue
    coords = COUNTRY_COORDS.get(country, (0, 0))
    # Jitter so overlapping groups are visible
    jlat = coords[0] + np.random.uniform(-3, 3)
    jlon = coords[1] + np.random.uniform(-3, 3)
    lats.append(jlat)
    lons.append(jlon)
    texts.append(f"<b>{grp}</b><br>Country: {country}<br>Techniques: {cnt}")
    colors_list.append(COUNTRY_COLOR.get(country, "#FF6B1A"))
    sizes.append(max(8, min(22, cnt // 3 + 7)))

fig_map = go.Figure()

# Country cluster markers (large, semi-transparent)
for country, grps in country_groups.items():
    if country == "Unknown":
        continue
    coords = COUNTRY_COORDS.get(country, (0, 0))
    total_tech = sum(c for _, c in grps)
    fig_map.add_trace(go.Scattergeo(
        lat=[coords[0]], lon=[coords[1]],
        mode="markers+text",
        marker=dict(
            size=max(20, min(50, len(grps) * 8)),
            color=COUNTRY_COLOR.get(country, "#FF6B1A"),
            opacity=0.15,
            line=dict(width=0),
        ),
        text=[country],
        textposition="top center",
        textfont=dict(size=10, color=COUNTRY_COLOR.get(country, "#FF6B1A"),
                      family="IBM Plex Mono"),
        hoverinfo="skip",
        showlegend=False,
    ))

# Individual group dots
fig_map.add_trace(go.Scattergeo(
    lat=lats, lon=lons,
    mode="markers",
    marker=dict(
        size=sizes,
        color=colors_list,
        opacity=0.85,
        line=dict(width=1, color="#0D0D0D"),
    ),
    text=texts,
    hovertemplate="%{text}<extra></extra>",
    showlegend=False,
))

fig_map.update_geos(
    projection_type="natural earth",
    showland=True, landcolor="#1A1400",
    showocean=True, oceancolor="#0D0D0D",
    showlakes=False,
    showcountries=True, countrycolor="#2A2000",
    showcoastlines=True, coastlinecolor="#2A2000",
    bgcolor="#0D0D0D",
    framecolor="#2A2000",
)
fig_map.update_layout(
    paper_bgcolor="#0D0D0D",
    plot_bgcolor="#0D0D0D",
    margin=dict(l=0, r=0, t=0, b=0),
    height=420,
    geo=dict(bgcolor="#0D0D0D"),
)

st.plotly_chart(fig_map, use_container_width=True)

# Legend
legend_countries = [c for c in COUNTRY_COLOR if c != "Unknown"]
legend_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px;">'
for c in legend_countries:
    col = COUNTRY_COLOR[c]
    grp_count = len(country_groups.get(c, []))
    if grp_count == 0:
        continue
    legend_html += f'<div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#8A7A6A;font-family:IBM Plex Mono,monospace;"><span style="width:10px;height:10px;border-radius:50%;background:{col};display:inline-block;"></span>{c} ({grp_count})</div>'
legend_html += '</div>'
st.markdown(legend_html, unsafe_allow_html=True)

st.divider()

# ─── ATTRIBUTION ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="sec-lbl">Incident Technique IDs</div>', unsafe_allow_html=True)

    default_text = ""
    if preset_choice != "— enter manually —":
        default_text = "\n".join(PRESETS[preset_choice])

    raw_input = st.text_area(
        "tids", value=default_text, height=200,
        placeholder="T1059\nT1003\nT1566\nT1071\nT1027",
        help="One ATT&CK technique ID per line. Sub-techniques (T1059.001) auto-collapsed.",
        label_visibility="collapsed",
    )

    parsed = []
    if raw_input.strip():
        parsed = [t.strip().upper() for t in re.split(r"[\n,]+", raw_input) if t.strip()]

    col_b, col_i = st.columns([2, 3])
    with col_b:
        run = st.button("Attribute →", type="primary", use_container_width=True)
    with col_i:
        st.caption(f"{len(parsed)} technique ID(s) entered" if parsed else "Enter technique IDs")

    with st.expander("ℹ How to use"):
        st.markdown("""
**1.** Enter ATT&CK technique IDs from your incident  
**2.** Click **Attribute →**  
**3.** Review top-3 candidate threat groups with confidence scores

Sub-technique IDs (e.g. `T1059.001`) are automatically collapsed to root (`T1059`).  
Use the sidebar presets to explore known APT signatures.
        """)

with col_right:
    st.markdown('<div class="sec-lbl">Attribution Results</div>', unsafe_allow_html=True)

    if run:
        if not parsed:
            st.warning("Enter at least one technique ID.")
        else:
            with st.spinner("Running inference…"):
                try:
                    results = predict_top3(parsed)

                    for r in results:
                        pct   = r["confidence"] * 100
                        color = RANK_COLORS[r["rank"]-1]
                        bg    = RANK_BG[r["rank"]-1]
                        medal = RANK_MEDALS[r["rank"]-1]
                        bar_w = int(r["confidence"] * 100)
                        country = next((APT_COUNTRY[k] for k in APT_COUNTRY
                                        if k.lower() in r["group"].lower()), "Unknown")
                        flag_note = f"&nbsp;·&nbsp;{country}" if country != "Unknown" else ""

                        st.markdown(f"""
<div class="rcard" style="border-color:{color};background:{bg};">
  <div class="rbadge">{medal} Rank #{r["rank"]}</div>
  <div class="rname">{r["group"]}<span style="font-size:12px;font-weight:400;color:#6A5A4A;">{flag_note}</span></div>
  <div class="rconf">Confidence: {r["confidence"]:.4f} ({pct:.1f}%)</div>
  <div class="rbar-bg"><div class="rbar-fill" style="width:{bar_w}%;background:{color};"></div></div>
</div>""", unsafe_allow_html=True)

                    # Confidence chart
                    fig = go.Figure(go.Bar(
                        x=[r["confidence"] for r in results],
                        y=[r["group"] for r in results],
                        orientation="h",
                        marker_color=RANK_COLORS[:len(results)],
                        text=[f"{r['confidence']:.4f}" for r in results],
                        textposition="outside", cliponaxis=False,
                    ))
                    fig.update_layout(
                        xaxis=dict(title="Confidence", range=[0, min(1.0, results[0]["confidence"]*1.4)],
                                   gridcolor="#1E1E1E", color="#6A5A4A"),
                        yaxis=dict(autorange="reversed", color="#6A5A4A"),
                        height=200,
                        margin=dict(l=0,r=60,t=4,b=30),
                        plot_bgcolor="#111", paper_bgcolor="#111",
                        font=dict(size=12, color="#E8E0D8"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Matched / Unknown technique IDs"):
                        matched_ids = [t.split(".")[0] if "." in t else t for t in parsed]
                        known   = [t for t in matched_ids if t in feature_vocab]
                        unknown = [t for t in parsed if (t.split(".")[0] if "." in t else t) not in feature_vocab]
                        if known:
                            st.success(f"**{len(known)} known:** {', '.join(sorted(set(known)))}")
                        if unknown:
                            st.warning(f"**{len(unknown)} skipped (not in vocab):** {', '.join(unknown)}")

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
    else:
        st.markdown("""
<div class="empty-state">
  <div style="font-size:36px;margin-bottom:10px;"></div>
  <div style="font-size:14px;">Enter technique IDs and click <strong>Attribute →</strong></div>
  <div style="font-size:12px;margin-top:6px;opacity:.7;">Or load a preset from the sidebar</div>
</div>""", unsafe_allow_html=True)

st.divider()

# ─── ABOUT (accordion dropdowns) ─────────────────────────────────────────────
st.markdown('<div class="sec-lbl">About This System</div>', unsafe_allow_html=True)

about_items = [
    ("", "What is APT Attribution?",
     "Advanced Persistent Threats (APTs) are sophisticated, state-sponsored hacking groups. While attackers frequently rotate infrastructure, they rarely change their <em>Tactics, Techniques and Procedures (TTPs)</em>. This engine parses the MITRE ATT&CK STIX 2.1 database — containing signatures for 170+ tracked global threat groups — and trains an ensemble ML classifier to attribute unknown incidents to the most likely threat actor based on observed techniques."),

    ("", "Model Architecture",
     "<b>Ensemble:</b> Soft-voting classifier combining three base models:<br>"
     "• <code>XGBoost</code> (hist tree, mlogloss, weighted x2–x4)<br>"
     "• <code>Random Forest</code> (balanced_subsample class weighting)<br>"
     "• <code>Calibrated SVM</code> (linear kernel, isotonic calibration)<br><br>"
     "<b>Tuning:</b> Optuna with TPE sampler, 60 trials, macro F1 objective, StratifiedKFold CV.<br>"
     "<b>Imbalance handling:</b> SMOTE on training set only (k=2). 30 synthetic samples per group via random technique sub-sampling."),

    ("", "Feature Engineering",
     "Each threat group is represented as a <b>sparse binary vector</b> over all unique ATT&CK technique IDs.<br><br>"
     "• <b>Root-only mode:</b> Sub-technique IDs (T1059.001) are collapsed to root (T1059). This reduces feature space while retaining discriminative signal for sparse groups.<br>"
     "• <b>Shape:</b> (n_groups × n_techniques) CSR sparse matrix.<br>"
     "• <b>Sparsity:</b> >95% — most groups use a small fraction of all techniques."),

    ("", "Known Limitations",
     "• Groups with fewer than 3 recorded techniques are excluded from training — too few signal points for reliable attribution.<br>"
     "• Single-sample classes (groups with only 1 training example after augmentation) never appear in the test set — metrics are computed only on verifiable classes.<br>"
     "• False-flag contamination: attackers deliberately reuse other groups' tools. The model looks for <em>combinations</em> of techniques, but adversarial cross-contamination can lower confidence.<br>"
     "• The dataset reflects what the community has <em>publicly documented</em> — novel or low-profile groups are underrepresented."),

    ("", "Evaluation Strategy",
     "Because each ATT&CK group originally has exactly 1 row in the raw feature matrix, we use a <b>data augmentation + held-out test set</b> strategy:<br><br>"
     "1. Generate 30 synthetic sub-samples per group (random technique subsets)<br>"
     "2. Stratified 80/20 train/test split on augmented data<br>"
     "3. Single-sample classes forced into train only<br>"
     "4. Metrics: Macro Precision, Recall, F1 · Top-3 Accuracy · Per-class report · Confusion matrix<br>"
     "5. Explainability: SHAP TreeExplainer on XGBoost base (mean |SHAP| per technique)"),
]

for icon, title, body in about_items:
    st.markdown(f"""
<details class="about-item">
  <summary>{icon}&nbsp;&nbsp;{title}</summary>
  <div class="body">{body}</div>
</details>
""", unsafe_allow_html=True)