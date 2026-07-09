"""
pages/analytics.py — Analytics & Reports Page
MITRE ATT&CK Orange Dark Theme
"""

import json, pathlib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Analytics | APT Attribution Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT      = pathlib.Path(__file__).parent.parent.resolve()
ARTIFACTS = ROOT / "artifacts"

@st.cache_resource(show_spinner=False)
def load_all():
    ev = json.loads((ARTIFACTS/"eval_summary.json").read_text())    if (ARTIFACTS/"eval_summary.json").exists()    else {}
    sh = json.loads((ARTIFACTS/"shap_importance.json").read_text()) if (ARTIFACTS/"shap_importance.json").exists() else []
    gc = json.loads((ARTIFACTS/"group_tech_counts.json").read_text()) if (ARTIFACTS/"group_tech_counts.json").exists() else {}
    md = json.loads((ARTIFACTS.parent/"models"/"training_metadata.json").read_text()) if (ARTIFACTS.parent/"models"/"training_metadata.json").exists() else {}
    return ev, sh, gc, md

eval_summary, shap_table, group_counts, train_meta = load_all()

# ─── CSS (shared theme) ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif;background-color:#0D0D0D!important;color:#E8E0D8!important;}
section[data-testid="stSidebar"]{background-color:#111111!important;border-right:1px solid #1E1E1E!important;}
section[data-testid="stSidebar"] *{color:#E8E0D8!important;}
.sec-lbl{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:#FF6B1A;margin-bottom:10px;display:block;}
.card{background:#111;border:1px solid #1E1E1E;border-radius:10px;padding:20px 22px;margin-bottom:16px;}
.card-title{font-size:12px;font-weight:600;color:#FF6B1A;letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px;font-family:'IBM Plex Mono',monospace;}
.big-num{font-family:'IBM Plex Mono',monospace;font-size:32px;font-weight:600;color:#FF6B1A;}
.big-lbl{font-size:11px;color:#6A5A4A;margin-top:2px;text-transform:uppercase;letter-spacing:.08em;}
.tag{display:inline-block;background:rgba(255,107,26,.12);border:1px solid rgba(255,107,26,.25);border-radius:4px;padding:2px 8px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#FF6B1A;margin:2px;}
.stButton>button{background:#FF6B1A!important;color:#000!important;font-weight:600!important;border:none!important;border-radius:6px!important;}
hr{border-color:#1E1E1E!important;}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#111", plot_bgcolor="#111",
    font=dict(color="#8A7A6A", size=11, family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#1E1E1E", linecolor="#2A2A2A", color="#6A5A4A"),
    yaxis=dict(gridcolor="#1E1E1E", linecolor="#2A2A2A", color="#6A5A4A"),
    margin=dict(l=10,r=20,t=30,b=30),
)

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px;border-bottom:1px solid #1E1E1E;margin-bottom:8px;'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#FF6B1A;letter-spacing:.15em;'> APT ATTRIBUTION</div>
        <div style='font-size:13px;font-weight:600;color:#FFF;margin-top:4px;'>Navigation</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
<div style='display:flex;flex-direction:column;gap:6px;'>
  <a href='/' target='_self' style='display:block;padding:8px 12px;border-radius:6px;
     background:rgba(255,255,255,0.04);border:1px solid #2A2A2A;
     color:#8A7A6A;text-decoration:none;font-size:13px;font-weight:500;'>
     &nbsp; Home — Attribution</a>
  <a href='/analytics' target='_self' style='display:block;padding:8px 12px;border-radius:6px;
     background:rgba(255,107,26,0.12);border:1px solid rgba(255,107,26,0.25);
     color:#FF6B1A;text-decoration:none;font-size:13px;font-weight:500;'>
     &nbsp; Analytics &amp; Reports</a>
</div>""", unsafe_allow_html=True)

# ─── PAGE HEADER ──────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#111 0%,#1A0E00 60%,#200A00 100%);
     border:1px solid #FF6B1A33;border-radius:12px;padding:22px 28px;margin-bottom:24px;'>
  <div style='font-family:IBM Plex Mono,monospace;font-size:10px;letter-spacing:.18em;color:#FF6B1A;margin-bottom:6px;'>ANALYTICS & REPORTS</div>
  <div style='font-size:22px;font-weight:700;color:#FFF;'>Model Performance · Explainability · Data Overview</div>
  <div style='font-size:12px;color:#6A5A4A;font-family:IBM Plex Mono,monospace;margin-top:4px;'>Full evaluation metrics · SHAP feature importance · Training data statistics</div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Model Performance",
    "  Explainability (SHAP)",
    "  Data Overview",
    "  Training Config",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("")

    # Top metric row
    f1v  = eval_summary.get("macro_f1",        0)
    t3v  = eval_summary.get("top3_accuracy",   0)
    prec = eval_summary.get("macro_precision", 0)
    rec  = eval_summary.get("macro_recall",    0)
    tgt_f1 = eval_summary.get("target_macro_f1", 0.62)
    tgt_t3 = eval_summary.get("target_top3_acc", 0.80)

    c1,c2,c3,c4 = st.columns(4)
    def big_card(col, val, lbl, target=None):
        ok_str = ""
        if target is not None:
            ok_str = f'<div style="font-size:10px;color:{"#1D9E75" if val>=target else "#E8451A"};margin-top:4px;">{"✓ Target met" if val>=target else "✗ Below target"} ({target:.2f})</div>'
        col.markdown(f"""
<div class="card" style="text-align:center;">
  <div class="big-num">{val:.3f}</div>
  <div class="big-lbl">{lbl}</div>
  {ok_str}
</div>""", unsafe_allow_html=True)

    big_card(c1, f1v,  "Macro F1",        tgt_f1)
    big_card(c2, t3v,  "Top-3 Accuracy",  tgt_t3)
    big_card(c3, prec, "Macro Precision")
    big_card(c4, rec,  "Macro Recall")

    st.markdown("")

    col_a, col_b = st.columns(2)

    # ── Radar chart ──
    with col_a:
        st.markdown('<span class="sec-lbl">Metric Radar</span>', unsafe_allow_html=True)
        cats = ["Macro F1","Top-3 Acc","Precision","Recall"]
        vals_r = [f1v, t3v, prec, rec]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals_r + [vals_r[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(255,107,26,0.15)",
            line=dict(color="#FF6B1A", width=2),
            name="Model",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=[tgt_f1, tgt_t3, tgt_f1, tgt_f1, tgt_f1],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(255,255,255,0.03)",
            line=dict(color="#444", width=1, dash="dot"),
            name="Target",
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="#111",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#2A2A2A", color="#6A5A4A"),
                angularaxis=dict(gridcolor="#2A2A2A", color="#8A7A6A"),
            ),
            paper_bgcolor="#111",
            font=dict(color="#8A7A6A", family="IBM Plex Mono"),
            showlegend=True,
            legend=dict(bgcolor="#111", bordercolor="#2A2A2A", borderwidth=1),
            height=300,
            margin=dict(l=40,r=40,t=20,b=20),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Metric gauge bars ──
    with col_b:
        st.markdown('<span class="sec-lbl">Target vs Achieved</span>', unsafe_allow_html=True)
        metrics_bar = [
            ("Macro F1",        f1v,  tgt_f1),
            ("Top-3 Accuracy",  t3v,  tgt_t3),
            ("Macro Precision", prec, tgt_f1),
            ("Macro Recall",    rec,  tgt_f1),
        ]
        fig_bar = go.Figure()
        for i, (name, val, tgt) in enumerate(metrics_bar):
            color = "#1D9E75" if val >= tgt else "#E8451A"
            fig_bar.add_trace(go.Bar(
                name=name, x=[val], y=[name],
                orientation="h",
                marker_color=color,
                text=[f"{val:.3f}"],
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
            ))
            # Target line
            fig_bar.add_shape(type="line",
                x0=tgt, x1=tgt, y0=i-0.4, y1=i+0.4,
                line=dict(color="#FF6B1A", width=2, dash="dot"),
            )
        fig_bar.update_layout(
            barmode="overlay",
            xaxis=dict(range=[0,1.1], gridcolor="#1E1E1E", color="#6A5A4A"),
            yaxis=dict(color="#6A5A4A"),
            height=300,
            **{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
            annotations=[dict(x=tgt_f1+0.01, y=3.5, text="target", showarrow=False,
                              font=dict(size=9, color="#FF6B1A", family="IBM Plex Mono"))],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Eval summary detail ──
    st.markdown('<span class="sec-lbl">Evaluation Detail</span>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="card">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:13px;">
    <div><span style="color:#6A5A4A;">Method</span><br><code style="color:#FF6B1A;">{eval_summary.get('eval_method','—')}</code></div>
    <div><span style="color:#6A5A4A;">Test Samples</span><br><code style="color:#FF6B1A;">{eval_summary.get('n_test_samples','—')}</code></div>
    <div><span style="color:#6A5A4A;">Groups in Test</span><br><code style="color:#FF6B1A;">{eval_summary.get('n_classes_in_test','—')}</code></div>
    <div><span style="color:#6A5A4A;">Groups with F1=0</span><br><code style="color:#E8451A;">{eval_summary.get('n_failing_groups','—')}</code></div>
  </div>
  <div style="margin-top:12px;padding-top:12px;border-top:1px solid #1E1E1E;font-size:12px;color:#6A5A4A;">
    ⚠&nbsp;{eval_summary.get('note','Single-sample classes excluded from test set.')}
  </div>
</div>""", unsafe_allow_html=True)

    # Failing groups
    failing = eval_summary.get("failing_groups", [])
    if failing:
        with st.expander(f"Groups with F1 = 0 on test set ({len(failing)} groups)"):
            tags = "".join(f'<span class="tag">{g}</span>' for g in failing)
            st.markdown(f'<div style="line-height:2;">{tags}</div>', unsafe_allow_html=True)
            st.caption("These are typically sparse groups with very few or highly shared techniques.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — EXPLAINABILITY (SHAP)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("")
    if not shap_table:
        st.info("Run notebook `03_evaluation.ipynb` to generate `shap_importance.json`.")
    else:
        st.markdown('<span class="sec-lbl">SHAP Feature Importance — XGBoost Base Estimator</span>', unsafe_allow_html=True)
        st.markdown("""
<div class="card" style="margin-bottom:16px;">
<div class="card-title">What is SHAP?</div>
SHAP (SHapley Additive exPlanations) assigns each feature a contribution score to the model's output.
<b>Mean |SHAP|</b> across all classes and samples tells us which ATT&CK technique IDs are most
discriminative for attributing groups — i.e. which techniques most uniquely identify a specific APT.
</div>""", unsafe_allow_html=True)

        col_s, _ = st.columns([1, 3])
        with col_s:
            top_n = st.slider("Top N techniques", 5, min(50, len(shap_table)), 20, key="shap_n")

        rows  = shap_table[:top_n]
        tids  = [r["technique_id"]  for r in rows]
        vals  = [r["mean_abs_shap"] for r in rows]

        # Horizontal bar — colour by rank
        norm  = [(v - min(vals))/(max(vals)-min(vals)+1e-9) for v in vals]
        colors_shap = [f"rgba(255,{int(107+100*n)},{int(26+100*n)},0.85)" for n in norm]

        fig_shap = go.Figure(go.Bar(
            x=vals[::-1], y=tids[::-1],
            orientation="h",
            marker_color=colors_shap[::-1],
            text=[f"{v:.4f}" for v in vals[::-1]],
            textposition="outside", cliponaxis=False,
        ))
        fig_shap.update_layout(
            xaxis_title="Mean |SHAP value|",
            height=max(350, top_n*22),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown('<span class="sec-lbl">Cumulative Importance</span>', unsafe_allow_html=True)
        all_vals = [r["mean_abs_shap"] for r in shap_table]
        cumsum   = np.cumsum(all_vals) / (sum(all_vals)+1e-9) * 100
        fig_cum  = go.Figure(go.Scatter(
            x=list(range(1, len(cumsum)+1)), y=cumsum,
            mode="lines", fill="tozeroy",
            fillcolor="rgba(255,107,26,0.10)",
            line=dict(color="#FF6B1A", width=2),
        ))
        fig_cum.add_hline(y=80, line_dash="dot", line_color="#555",
                          annotation_text="80% importance", annotation_font_color="#8A7A6A")
        fig_cum.update_layout(
            xaxis_title="Number of techniques", yaxis_title="Cumulative importance (%)",
            height=280, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        idx80 = next((i for i,v in enumerate(cumsum) if v>=80), len(cumsum))
        st.caption(f"Top **{idx80+1}** techniques account for **80%** of total model importance.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("")
    if not group_counts:
        st.info("Run notebook `01_ingestion_feature_matrix.ipynb` to generate `group_tech_counts.json`.")
    else:
        vals_g = sorted(group_counts.values())

        # Summary stats
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="card" style="text-align:center;"><div class="big-num">{len(group_counts)}</div><div class="big-lbl">Total Groups</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card" style="text-align:center;"><div class="big-num">{int(np.mean(vals_g))}</div><div class="big-lbl">Mean Techniques</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card" style="text-align:center;"><div class="big-num">{max(vals_g)}</div><div class="big-lbl">Max Techniques</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="card" style="text-align:center;"><div class="big-num">{sum(1 for v in vals_g if v<3)}</div><div class="big-lbl">Thin Groups (&lt;3)</div></div>', unsafe_allow_html=True)

        st.markdown("")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown('<span class="sec-lbl">Technique Count Distribution</span>', unsafe_allow_html=True)
            fig_hist = go.Figure(go.Histogram(
                x=vals_g, nbinsx=30,
                marker_color="#FF6B1A", opacity=0.8,
                marker_line=dict(color="#0D0D0D", width=0.5),
            ))
            fig_hist.add_vline(x=np.mean(vals_g), line_dash="dot", line_color="#FFD180",
                               annotation_text=f"Mean={np.mean(vals_g):.0f}",
                               annotation_font_color="#FFD180")
            fig_hist.update_layout(
                xaxis_title="Techniques per group", yaxis_title="Group count",
                height=280, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_d2:
            st.markdown('<span class="sec-lbl">Top 20 Groups by Technique Count</span>', unsafe_allow_html=True)
            top20 = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            names_t20 = [g for g,_ in top20]
            vals_t20  = [c for _,c in top20]
            norm20    = [(v-min(vals_t20))/(max(vals_t20)-min(vals_t20)+1e-9) for v in vals_t20]
            cols20    = [f"rgba(255,{int(107+80*n)},{int(26+80*n)},0.85)" for n in norm20]

            fig_top = go.Figure(go.Bar(
                x=vals_t20[::-1], y=names_t20[::-1],
                orientation="h",
                marker_color=cols20[::-1],
                text=vals_t20[::-1], textposition="outside", cliponaxis=False,
            ))
            fig_top.update_layout(
                xaxis_title="Technique count",
                height=420, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_top, use_container_width=True)

        # Cumulative coverage
        st.markdown('<span class="sec-lbl">Cumulative Technique Coverage</span>', unsafe_allow_html=True)
        vals_desc   = sorted(vals_g, reverse=True)
        cumsum_grp  = np.cumsum(vals_desc)
        pct_coverage = cumsum_grp / cumsum_grp[-1] * 100

        fig_cov = go.Figure(go.Scatter(
            x=list(range(1, len(pct_coverage)+1)), y=pct_coverage,
            mode="lines", fill="tozeroy",
            fillcolor="rgba(255,107,26,0.08)",
            line=dict(color="#FF6B1A", width=2),
        ))
        fig_cov.add_hline(y=80, line_dash="dot", line_color="#555",
                          annotation_text="80% coverage", annotation_font_color="#8A7A6A")
        fig_cov.update_layout(
            xaxis_title="Number of groups (ranked by technique count)",
            yaxis_title="Cumulative technique coverage (%)",
            height=260, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_cov, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — TRAINING CONFIG
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("")

    if not train_meta:
        st.info("Run notebook `02_training_pipeline.ipynb` to generate `training_metadata.json`.")
    else:
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown('<span class="sec-lbl">Training Configuration</span>', unsafe_allow_html=True)
            config_items = [
                ("Random State",     train_meta.get("random_state","—")),
                ("Test Size",        train_meta.get("test_size","—")),
                ("SMOTE k",          train_meta.get("smote_k","—")),
                ("Optuna Trials",    train_meta.get("n_trials","—")),
                ("Best CV F1",       f"{train_meta.get('optuna_best_value',0):.4f}"),
                ("Classes",          train_meta.get("n_classes","—")),
                ("Features",         train_meta.get("n_features","—")),
            ]
            rows_html = "".join(f"""
<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1E1E1E;font-size:13px;">
  <span style="color:#6A5A4A;">{k}</span>
  <code style="color:#FF6B1A;">{v}</code>
</div>""" for k,v in config_items)
            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

        with col_t2:
            st.markdown('<span class="sec-lbl">Best Optuna Hyperparameters</span>', unsafe_allow_html=True)
            bp = train_meta.get("optuna_best_params", {})
            if bp:
                bp_html = "".join(f"""
<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1E1E1E;font-size:13px;">
  <span style="color:#6A5A4A;">{k}</span>
  <code style="color:#FF6B1A;">{f"{v:.4f}" if isinstance(v, float) else v}</code>
</div>""" for k, v in bp.items())
                st.markdown(f'<div class="card">{bp_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No Optuna params found in metadata.")

        st.markdown('<span class="sec-lbl">Model Architecture</span>', unsafe_allow_html=True)
        st.markdown("""
<div class="card">
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;text-align:center;">
  <div>
    <div style="font-size:11px;color:#FF6B1A;font-family:IBM Plex Mono,monospace;margin-bottom:8px;">XGBoost</div>
    <div style="font-size:12px;color:#6A5A4A;line-height:1.8;">hist tree method<br>mlogloss objective<br>Weight: 2–4×</div>
  </div>
  <div style="border-left:1px solid #1E1E1E;border-right:1px solid #1E1E1E;">
    <div style="font-size:11px;color:#FF6B1A;font-family:IBM Plex Mono,monospace;margin-bottom:8px;">Random Forest</div>
    <div style="font-size:12px;color:#6A5A4A;line-height:1.8;">balanced_subsample<br>class weighting<br>Weight: 1–3×</div>
  </div>
  <div>
    <div style="font-size:11px;color:#FF6B1A;font-family:IBM Plex Mono,monospace;margin-bottom:8px;">Calibrated SVM</div>
    <div style="font-size:12px;color:#6A5A4A;line-height:1.8;">linear kernel<br>isotonic calibration<br>Weight: 1×</div>
  </div>
</div>
<div style="margin-top:16px;padding-top:12px;border-top:1px solid #1E1E1E;text-align:center;font-size:12px;color:#4A3A2A;">
  Soft-voting ensemble — probabilities averaged with learned weights
</div>
</div>""", unsafe_allow_html=True)