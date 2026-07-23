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

# ─── description-mapper pipeline (lazy-loaded only when that tab is used) ───
@st.cache_resource(show_spinner=False)
def load_description_pipeline():
    from sentence_transformers import SentenceTransformer
    from description_mapper import load_pipeline
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus, alias_table, tids, technique_embeddings, idf_table = load_pipeline(model)
    return model, corpus, alias_table, tids, technique_embeddings, idf_table

# ─── APT group → country/region mapping ─────────────────────────────────────
APT_COUNTRY = {
    # ── Russia ────────────────────────────────────────────────
    "APT28":"Russia","APT29":"Russia","Sandworm Team":"Russia","Turla":"Russia",
    "Gamaredon Group":"Russia","Cozy Bear":"Russia","Fancy Bear":"Russia",
    "Ember Bear":"Russia","IRON TILDEN":"Russia","Dragonfly":"Russia",
    "Energetic Bear":"Russia","Berserk Bear":"Russia","TEMP.Veles":"Russia",
    "Wizard Spider":"Russia","Indrik Spider":"Russia","Carbon Spider":"Russia",
    "Primitive Bear":"Russia","Venomous Bear":"Russia","Gossamer Bear":"Russia",
    "IRON VIKING":"Russia","Trickbot":"Russia","Ryuk":"Russia",
    "Silence":"Russia","MoneyTaker":"Russia",
    # ── China ─────────────────────────────────────────────────
    "APT1":"China","APT10":"China","APT17":"China","APT19":"China",
    "APT40":"China","APT41":"China","APT3":"China","APT30":"China",
    "Winnti Group":"China","BRONZE BUTLER":"China","Ke3chang":"China",
    "Leviathan":"China","Mustang Panda":"China","TA413":"China",
    "Threat Group-3390":"China","menuPass":"China","Deep Panda":"China",
    "Elderwood":"China","GALLIUM":"China","Wicked Panda":"China",
    "Aquatic Panda":"China","Bronze President":"China","Calypso":"China",
    "APT15":"China","APT31":"China","APT41":"China","Gothic Panda":"China",
    "Judgment Panda":"China","Maverick Panda":"China","Numbered Panda":"China",
    "Stone Panda":"China","Union Jack":"China","Vixen Panda":"China",
    "HAFNIUM":"China","RedAlpha":"China","TA416":"China","RedDelta":"China",
    "Lotus Panda":"China","Naikon":"China","Ke3chang":"China",
    "UNC215":"China","UNC2630":"China","BackdoorDiplomacy":"China",
    # ── North Korea ───────────────────────────────────────────
    "Lazarus Group":"North Korea","Kimsuky":"North Korea","APT37":"North Korea",
    "APT38":"North Korea","TEMP.Hermit":"North Korea","Andariel":"North Korea",
    "BlueNoroff":"North Korea","DarkHotel":"North Korea","Ricochet Chollima":"North Korea",
    "Labyrinth Chollima":"North Korea","Stardust Chollima":"North Korea",
    "Silent Chollima":"North Korea","Velvet Chollima":"North Korea",
    "Group 123":"North Korea","Reaper":"North Korea","ScarCruft":"North Korea",
    "Temp.Reaper":"North Korea","UNC2970":"North Korea","TraderTraitor":"North Korea",
    # ── Iran ──────────────────────────────────────────────────
    "APT33":"Iran","APT34":"Iran","APT35":"Iran","APT39":"Iran",
    "Charming Kitten":"Iran","MuddyWater":"Iran","OilRig":"Iran",
    "Magic Hound":"Iran","HEXANE":"Iran","Agrius":"Iran","Pioneer Kitten":"Iran",
    "Cobalt Gypsy":"Iran","Cobalt Illusion":"Iran","Refined Kitten":"Iran",
    "Helix Kitten":"Iran","Remix Kitten":"Iran","Static Kitten":"Iran",
    "Tortoiseshell":"Iran","Fox Kitten":"Iran","Phosphorus":"Iran",
    "CopyKittens":"Iran","Crambus":"Iran","Seedworm":"Iran",
    "Volatile Cedar":"Lebanon","MERCURY":"Iran","DEV-0343":"Iran",
    # ── USA / Five Eyes ───────────────────────────────────────
    "Equation Group":"USA","Longhorn":"USA","The Lamberts":"USA",
    "Tailored Access Operations":"USA","TAO":"USA","NOBUS":"USA",
    # ── EU / Western cybercriminal (attributed to Eastern Europe/Russia often) ──
    "FIN7":"Eastern Europe","FIN6":"Eastern Europe","FIN4":"Eastern Europe",
    "Carbanak":"Eastern Europe","Cobalt Group":"Eastern Europe",
    "TA505":"Eastern Europe","DarkHydrus":"Eastern Europe",
    "Gorgon Group":"Eastern Europe","GOLD SOUTHFIELD":"Eastern Europe",
    "GOLD DUPONT":"Eastern Europe","GOLD NIAGARA":"Eastern Europe",
    "LUNAR SPIDER":"Eastern Europe","MUMMY SPIDER":"Eastern Europe",
    "GRACEFUL SPIDER":"Eastern Europe","DAGGER PANDA":"Eastern Europe",
    # ── Vietnam ───────────────────────────────────────────────
    "APT32":"Vietnam","OceanLotus":"Vietnam","APT-C-00":"Vietnam",
    "Canvas Cyclone":"Vietnam","Bismuth":"Vietnam",
    # ── Pakistan ──────────────────────────────────────────────
    "Sidewinder":"Pakistan","Transparent Tribe":"Pakistan",
    "APT36":"Pakistan","ProjectM":"Pakistan","Mythic Leopard":"Pakistan",
    # ── India ─────────────────────────────────────────────────
    "Donot Team":"India","APT-C-35":"India","Viceroy Tiger":"India",
    # ── Turkey ────────────────────────────────────────────────
    "Sea Turtle":"Turkey","PROMETHIUM":"Turkey","StrongPity":"Turkey",
    # ── Lebanon ───────────────────────────────────────────────
    "Dark Caracal":"Lebanon","Lebanese Cedar":"Lebanon","Volatile Cedar":"Lebanon",
    # ── Palestinian Territory ─────────────────────────────────
    "Gaza Cybergang":"Palestinian Territory","Molerats":"Palestinian Territory",
    "Arid Viper":"Palestinian Territory","Desert Falcon":"Palestinian Territory",
    # ── Saudi Arabia / UAE ────────────────────────────────────
    "Stealth Falcon":"UAE","Project Raven":"UAE",
    # ── Belarus ───────────────────────────────────────────────
    "UNC1151":"Belarus","Ghostwriter":"Belarus",
    # ── Multiple / Unattributed ───────────────────────────────
    "Unknown":"Unknown",
}

COUNTRY_COORDS = {
    "Russia":                (61.52,  105.32),
    "China":                 (35.86,  104.19),
    "North Korea":           (40.34,  127.51),
    "Iran":                  (32.43,   53.69),
    "USA":                   (37.09,  -95.71),
    "Vietnam":               (14.06,  108.28),
    "Pakistan":              (30.38,   69.35),
    "India":                 (20.59,   78.96),
    "Turkey":                (38.96,   35.24),
    "Lebanon":               (33.85,   35.86),
    "Palestinian Territory": (31.95,   35.23),
    "Eastern Europe":        (50.40,   30.52),   # Kyiv centroid
    "UAE":                   (23.42,   53.85),
    "Belarus":               (53.71,   27.95),
    "Unknown":               (20.0,     0.0),
}

COUNTRY_COLOR = {
    "Russia":                "#E8451A",
    "China":                 "#FF6B1A",
    "North Korea":           "#D84315",
    "Iran":                  "#F4511E",
    "USA":                   "#4FC3F7",   # blue — stands out from orange cluster
    "Vietnam":               "#FF7043",
    "Pakistan":              "#FFAB40",
    "India":                 "#FFD180",
    "Turkey":                "#FF9800",
    "Lebanon":               "#FFB74D",
    "Palestinian Territory": "#FFCC02",
    "Eastern Europe":        "#CE93D8",   # purple — distinct from state actors
    "UAE":                   "#80CBC4",
    "Belarus":               "#EF9A9A",
    "Unknown":               "#444",
}

PRESETS = {
    "APT28 — Fancy Bear (Russia)":   ['T1001', 'T1003', 'T1005', 'T1014', 'T1021', 'T1025', 'T1027', 'T1030', 'T1033', 'T1036', 'T1037', 'T1039', 'T1040', 'T1048', 'T1056', 'T1057', 'T1059', 'T1068', 'T1070', 'T1071', 'T1074', 'T1078', 'T1082', 'T1083', 'T1090', 'T1091', 'T1092', 'T1098', 'T1102', 'T1105', 'T1110', 'T1113', 'T1114', 'T1119', 'T1120', 'T1133', 'T1134', 'T1137', 'T1140', 'T1189', 'T1190', 'T1199', 'T1203', 'T1204', 'T1210', 'T1211', 'T1213', 'T1218', 'T1221', 'T1498', 'T1505', 'T1528', 'T1546', 'T1547', 'T1550', 'T1557', 'T1559', 'T1560', 'T1564', 'T1566', 'T1567', 'T1573', 'T1583', 'T1584', 'T1586', 'T1588', 'T1589', 'T1591', 'T1595', 'T1596', 'T1598', 'T1669', 'T1684', 'T1685'],
    "Lazarus Group (North Korea)":     ['T1001', 'T1003', 'T1005', 'T1008', 'T1010', 'T1012', 'T1016', 'T1021', 'T1027', 'T1033', 'T1036', 'T1041', 'T1046', 'T1047', 'T1048', 'T1049', 'T1053', 'T1055', 'T1056', 'T1057', 'T1059', 'T1070', 'T1071', 'T1074', 'T1078', 'T1082', 'T1083', 'T1087', 'T1090', 'T1098', 'T1102', 'T1104', 'T1105', 'T1106', 'T1110', 'T1112', 'T1124', 'T1132', 'T1140', 'T1189', 'T1202', 'T1203', 'T1204', 'T1218', 'T1220', 'T1221', 'T1485', 'T1489', 'T1491', 'T1496', 'T1497', 'T1529', 'T1534', 'T1542', 'T1543', 'T1547', 'T1553', 'T1557', 'T1560', 'T1561', 'T1562', 'T1564', 'T1566', 'T1567', 'T1571', 'T1573', 'T1574', 'T1583', 'T1584', 'T1585', 'T1587', 'T1588', 'T1589', 'T1591', 'T1593', 'T1608', 'T1614', 'T1620', 'T1680', 'T1685', 'T1686'],
    "APT29 — Cozy Bear (Russia)":   ['T1001', 'T1003', 'T1005', 'T1016', 'T1018', 'T1021', 'T1027', 'T1036', 'T1037', 'T1047', 'T1048', 'T1057', 'T1059', 'T1068', 'T1069', 'T1070', 'T1071', 'T1074', 'T1078', 'T1082', 'T1083', 'T1087', 'T1090', 'T1095', 'T1098', 'T1102', 'T1110', 'T1114', 'T1133', 'T1136', 'T1140', 'T1190', 'T1195', 'T1199', 'T1203', 'T1204', 'T1213', 'T1218', 'T1482', 'T1484', 'T1505', 'T1528', 'T1539', 'T1546', 'T1547', 'T1548', 'T1550', 'T1552', 'T1553', 'T1555', 'T1556', 'T1558', 'T1560', 'T1562', 'T1566', 'T1568', 'T1573', 'T1583', 'T1584', 'T1586', 'T1587', 'T1588', 'T1589', 'T1595', 'T1606', 'T1621', 'T1649', 'T1651', 'T1665', 'T1685'],
    "APT41 — Winnti (China)":      ['T1003', 'T1005', 'T1008', 'T1012', 'T1016', 'T1018', 'T1021', 'T1027', 'T1030', 'T1033', 'T1036', 'T1037', 'T1046', 'T1047', 'T1049', 'T1053', 'T1055', 'T1056', 'T1059', 'T1069', 'T1070', 'T1071', 'T1078', 'T1082', 'T1083', 'T1087', 'T1090', 'T1098', 'T1102', 'T1104', 'T1105', 'T1110', 'T1112', 'T1133', 'T1135', 'T1136', 'T1190', 'T1195', 'T1197', 'T1203', 'T1213', 'T1218', 'T1480', 'T1484', 'T1486', 'T1496', 'T1542', 'T1543', 'T1546', 'T1547', 'T1550', 'T1553', 'T1555', 'T1560', 'T1562', 'T1566', 'T1568', 'T1569', 'T1570', 'T1574', 'T1588', 'T1589', 'T1595', 'T1596', 'T1599', 'T1684', 'T1685'],
    "MuddyWater (Iran)":           ['T1003', 'T1016', 'T1027', 'T1033', 'T1036', 'T1041', 'T1047', 'T1049', 'T1053', 'T1057', 'T1059', 'T1071', 'T1074', 'T1082', 'T1083', 'T1087', 'T1090', 'T1102', 'T1104', 'T1105', 'T1113', 'T1132', 'T1137', 'T1140', 'T1190', 'T1203', 'T1204', 'T1210', 'T1218', 'T1219', 'T1518', 'T1534', 'T1547', 'T1548', 'T1552', 'T1555', 'T1559', 'T1560', 'T1562', 'T1566', 'T1567', 'T1571', 'T1573', 'T1574', 'T1583', 'T1588', 'T1589', 'T1590', 'T1684', 'T1685'],
    "Minimal incident — 2 techniques":["T1059","T1003"],
}

# empirical reliability floor from find_min_coverage.py sweep — used only
# to show an honest confidence caveat, never to alter predictions
MIN_RELIABLE_TECHNIQUES = 18

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

/* Mode toggle */
div[data-testid="stRadio"] > label { display:none; }
div[data-testid="stRadio"] > div { gap: 8px; }
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

# ── resolve country for every group ──────────────────────────────────────────
def resolve_country(grp_name):
    # exact match first
    if grp_name in APT_COUNTRY:
        return APT_COUNTRY[grp_name]
    # substring match (case-insensitive)
    grp_lower = grp_name.lower()
    for k, v in APT_COUNTRY.items():
        if k.lower() in grp_lower or grp_lower in k.lower():
            return v
    return "Unknown"

# Build per-country groups dict
country_groups = {}
for grp, cnt in group_counts.items():
    country = resolve_country(grp)
    country_groups.setdefault(country, []).append((grp, cnt))

# ── spiral jitter: spread groups within a country in a tight circle ───────────
def spiral_positions(center_lat, center_lon, n, radius=4.5):
    """Return n (lat, lon) positions spiralled around a centre."""
    if n == 1:
        return [(center_lat, center_lon)]
    positions = []
    for i in range(n):
        angle  = 2 * np.pi * i / n
        r      = radius * (0.4 + 0.6 * (i / max(n-1, 1)))
        positions.append((
            center_lat + r * np.cos(angle) * 0.7,
            center_lon + r * np.sin(angle),
        ))
    return positions

np.random.seed(42)  # deterministic layout

lats, lons, texts, colors_list, sizes = [], [], [], [], []

for country, grps in country_groups.items():
    if country == "Unknown":
        # Scatter unknown groups across the bottom of the map
        for i, (grp, cnt) in enumerate(grps):
            lats.append(-40 + np.random.uniform(-5, 5))
            lons.append(-140 + i * (280 / max(len(grps), 1)) + np.random.uniform(-5, 5))
            texts.append(f"<b>{grp}</b><br>Country: Unattributed<br>Techniques: {cnt}")
            colors_list.append(COUNTRY_COLOR["Unknown"])
            sizes.append(max(7, min(16, cnt // 4 + 6)))
        continue

    coords   = COUNTRY_COORDS.get(country, (0, 0))
    positions = spiral_positions(coords[0], coords[1], len(grps))

    for (grp, cnt), (jlat, jlon) in zip(grps, positions):
        lats.append(jlat)
        lons.append(jlon)
        texts.append(
            f"<b>{grp}</b><br>"
            f"<span style='color:#FF6B1A'>Country:</span> {country}<br>"
            f"<span style='color:#FF6B1A'>Techniques:</span> {cnt}"
        )
        colors_list.append(COUNTRY_COLOR.get(country, "#FF6B1A"))
        sizes.append(max(8, min(24, cnt // 3 + 7)))

fig_map = go.Figure()

# ── country halo rings ────────────────────────────────────────────────────────
for country, grps in country_groups.items():
    if country == "Unknown":
        continue
    coords     = COUNTRY_COORDS.get(country, (0, 0))
    n_grps     = len(grps)
    halo_size  = max(22, min(55, n_grps * 9))
    clr        = COUNTRY_COLOR.get(country, "#FF6B1A")
    fig_map.add_trace(go.Scattergeo(
        lat=[coords[0]], lon=[coords[1]],
        mode="markers+text",
        marker=dict(size=halo_size, color=clr, opacity=0.12, line=dict(width=0)),
        text=[f"{country} ({n_grps})"],
        textposition="top center",
        textfont=dict(size=9, color=clr, family="IBM Plex Mono"),
        hoverinfo="skip",
        showlegend=False,
    ))

# ── individual group dots ─────────────────────────────────────────────────────
fig_map.add_trace(go.Scattergeo(
    lat=lats, lon=lons,
    mode="markers",
    marker=dict(
        size=sizes,
        color=colors_list,
        opacity=0.88,
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
legend_entries = [
    (c, COUNTRY_COLOR[c], len(country_groups.get(c, [])))
    for c in COUNTRY_COLOR
    if len(country_groups.get(c, [])) > 0
]
legend_entries.sort(key=lambda x: -x[2])  # sort by group count desc

legend_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;">'
for c, col, cnt in legend_entries:
    legend_html += (
        f'<div style="display:flex;align-items:center;gap:5px;font-size:11px;'
        f'color:#8A7A6A;font-family:IBM Plex Mono,monospace;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{col};'
        f'display:inline-block;flex-shrink:0;"></span>{c} ({cnt})</div>'
    )
legend_html += '</div>'
st.markdown(legend_html, unsafe_allow_html=True)

st.divider()

# ─── shared results renderer (used by BOTH input modes) ─────────────────────
def render_results(results):
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


# ─── ATTRIBUTION ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

parsed = []

with col_left:
    st.markdown('<div class="sec-lbl">Incident Input</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "input_mode",
        ["Technique IDs", "Describe What Happened"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── shared state for whichever mode fires ──
    run_results = None
    run_error = None
    mapped_details = None
    coverage_warning = None

    if input_mode == "Technique IDs":
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
**1.** Enter ATT&CK technique IDs from your incident (eg. from an EDR alert, SIEM rule, or threat intel report)
**2.** Click **Attribute →**
**3.** Review top-3 candidate threat groups with confidence scores

Sub-technique IDs (e.g. `T1059.001`) are automatically collapsed to root (`T1059`).
Use the sidebar presets to explore known APT signatures.
            """)

        if run:
            if not parsed:
                run_error = "empty"
            else:
                try:
                    run_results = predict_top3(parsed)
                    if len(parsed) < MIN_RELIABLE_TECHNIQUES:
                        coverage_warning = (
                            f"Only {len(parsed)} technique ID(s) provided — the model's "
                            f"empirically-measured reliability floor is ~{MIN_RELIABLE_TECHNIQUES}. "
                            f"Predictions on sparse input may be less reliable."
                        )
                except ValueError as e:
                    run_error = str(e)
                except Exception as e:
                    run_error = f"Unexpected error: {e}"

    else:  # ── Describe What Happened ──
        description_input = st.text_area(
            "description", height=200,
            placeholder=(
                "Example: The attacker sent a spearphishing email with a malicious "
                "attachment. Once opened, it ran an obfuscated PowerShell script that "
                "used Mimikatz to dump credentials, then moved laterally via RDP and "
                "established persistence through a scheduled task..."
            ),
            help="Describe the incident in plain language — tools, behaviors, and techniques observed. More technical detail improves accuracy.",
            label_visibility="collapsed",
        )

        run = st.button("Analyze Description →", type="primary", use_container_width=True)

        with st.expander("ℹ How to use"):
            st.markdown(f"""
**1.** Describe what happened — the tools used, how access was gained, persistence, C2 behavior, etc.
**2.** Click **Analyze Description →**
**3.** The description is mapped to MITRE ATT&CK technique IDs (via semantic matching against
MITRE's own technique text, plus known tool/malware relationships from MITRE's data), then
attributed to the top-3 candidate groups.

The more specific technical detail you provide, the more reliable the result — the model's
measured reliability floor is around **{MIN_RELIABLE_TECHNIQUES} techniques**; very short
descriptions may map to fewer than that.
            """)

        if run:
            if not description_input.strip():
                run_error = "empty"
            else:
                try:
                    with st.spinner("Loading NLP pipeline (first run only)…"):
                        model, corpus, alias_table, tids, technique_embeddings, idf_table = load_description_pipeline()
                    from description_mapper import attribute_from_description
                    technique_ids, mapped_details, run_results, coverage_warning = attribute_from_description(
                        description_input, corpus, alias_table, tids, technique_embeddings, model, idf_table=idf_table,
                    )
                except ValueError as e:
                    run_error = str(e)
                except Exception as e:
                    run_error = f"Unexpected error: {e}"

with col_right:
    st.markdown('<div class="sec-lbl">Attribution Results</div>', unsafe_allow_html=True)

    if run_error == "empty":
        st.warning("Enter at least one technique ID." if input_mode == "Technique IDs"
                    else "Enter a description of the incident.")
    elif run_error:
        st.error(run_error)
    elif run_results:
        if coverage_warning:
            st.warning(coverage_warning)

        render_results(run_results)

        if mapped_details is not None:
            with st.expander(f"Mapped technique IDs ({len(mapped_details)})"):
                for m in mapped_details:
                    st.markdown(
                        f"**{m['technique_id']}** — {m['name']}  "
                        f"<span style='color:#6A5A4A;font-size:11px;'>score={m['score']} · {m['reason']}</span>",
                        unsafe_allow_html=True,
                    )
        else:
            with st.expander("Matched / Unknown technique IDs"):
                matched_ids = [t.split(".")[0] if "." in t else t for t in parsed]
                known   = [t for t in matched_ids if t in feature_vocab]
                unknown = [t for t in parsed if (t.split(".")[0] if "." in t else t) not in feature_vocab]
                if known:
                    st.success(f"**{len(known)} known:** {', '.join(sorted(set(known)))}")
                if unknown:
                    st.warning(f"**{len(unknown)} skipped (not in vocab):** {', '.join(unknown)}")
    else:
        st.markdown("""
<div class="empty-state">
  <div style="font-size:36px;margin-bottom:10px;"></div>
  <div style="font-size:14px;">Enter technique IDs or describe the incident, then click Attribute.</div>
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

    ("", "Description → Technique Mapping",
     "The free-text input mode maps incident descriptions to ATT&CK technique IDs using two layers, "
     "both sourced from MITRE's own data — no hand-typed technique mappings:<br><br>"
     "• <b>Semantic matching:</b> a pretrained sentence-embedding model compares the description against "
     "MITRE's own technique name + description text.<br>"
     "• <b>Software/tool matching:</b> tool and malware names (e.g. Mimikatz, Cobalt Strike) mentioned in "
     "the text are matched to techniques via MITRE's own STIX <code>software \"uses\" technique</code> "
     "relationship data.<br><br>"
     f"Results below ~{MIN_RELIABLE_TECHNIQUES} mapped techniques are flagged with a reliability caveat, "
     "based on empirical testing of the minimum technique count needed for consistent attribution."),

    ("", "Known Limitations",
     "• Groups with fewer than 3 recorded techniques are excluded from training — too few signal points for reliable attribution.<br>"
     "• Single-sample classes (groups with only 1 training example after augmentation) never appear in the test set — metrics are computed only on verifiable classes.<br>"
     "• The training augmentation keeps 75-100% of a group's techniques per sample — sparse real-world input (10-15 techniques) falls outside this range for large groups, which is why the reliability floor exists.<br>"
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