from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
from pipeline import (ARCHETYPE_DESC, CLUSTER_FEATURES, build_model, cluster_quality, fit_cluster, k_scan,
                      load_data, project_2d)
from reliability import (bootstrap_gpi_ci, classification_confidence, cluster_separation_tests,
                         data_quality_report, generate_insights, quality_score, weight_sensitivity)
from report import build_exec_summary
from scoring import (DEFAULT_WEIGHTS, KPI_COLUMNS, PILLARS, STRATEGIES, STRATEGY_COLORS, channel_table, recommend,
                     score_and_classify, simulate_shift)

st.set_page_config(page_title="SkyCity Growth Intelligence", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    .badge {display:inline-block; padding:4px 12px; border-radius:14px; color:white; font-weight:600; font-size:0.9rem;}
    .conf-badge {display:inline-block; padding:3px 10px; border-radius:10px; font-weight:500; font-size:0.8rem; margin-left:6px;}
    .archetype {background:#F2F6F8; border-left:5px solid #0E7C86; padding:10px 14px; border-radius:6px; margin-bottom:8px;}
    .insight-card {background:#F8F9FA; border-left:4px solid #0E7C86; padding:14px 18px; border-radius:8px; margin-bottom:12px;}
    .insight-card.warning {border-left-color:#C0392B;}
    .insight-card.moderate {border-left-color:#E6A100;}
    .insight-card.good {border-left-color:#1B9E77;}
    .evidence {background:#EDF2F4; padding:8px 12px; border-radius:6px; font-size:0.88rem; margin-top:8px; color:#495057;}
    div[data-testid="stMetric"] {background:#F2F6F8; padding:10px 14px; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_CSV = Path(__file__).parent / "data" / "skycity_restaurants.csv"

@st.cache_data(show_spinner=False)
def _load_default() -> pd.DataFrame:
    return load_data(DEFAULT_CSV)


@st.cache_data(show_spinner=False)
def _load_upload(raw: bytes) -> pd.DataFrame:
    return load_data(io.BytesIO(raw))


@st.cache_data(show_spinner="Fitting PCA + clustering model...")
def get_model(df, algo, k, include_cat, cat_weight, eps_mult, min_samples):
    return build_model(df, algo, k, include_cat, cat_weight, eps_mult, min_samples)


@st.cache_data(show_spinner="Projecting to 2D...")
def get_projection(_X, _Z, method, key):
    return project_2d(_X, _Z, method)


@st.cache_data(show_spinner=False)
def get_kscan(_Z, key):
    return k_scan(_Z)


@st.cache_data(show_spinner="Training driver model...")
def driver_importance(d: pd.DataFrame, target: str, feats: tuple):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    X, y = d[list(feats)], d[target]
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    r2 = float(cross_val_score(rf, X, y, cv=3, scoring="r2").mean())
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=list(feats)), r2


@st.cache_data(show_spinner=False)
def get_data_quality(raw_bytes):
    import io as _io
    df = pd.read_csv(_io.BytesIO(raw_bytes)) if isinstance(raw_bytes, bytes) else raw_bytes
    return data_quality_report(df)


@st.cache_data(show_spinner="Computing GPI confidence intervals...")
def get_ci(feat_bytes, weights_tuple):
    weights = dict(weights_tuple)
    return bootstrap_gpi_ci(pd.read_pickle(feat_bytes), weights, n_boot=150)


@st.cache_data(show_spinner="Analysing rank stability...")
def get_stability(feat_bytes, weights_tuple):
    weights = dict(weights_tuple)
    return weight_sensitivity(pd.read_pickle(feat_bytes), weights, n_perturb=50)


st.sidebar.title("📈 Growth Intelligence")
st.sidebar.caption("SkyCity Auckland Restaurants & Bars")

up = st.sidebar.file_uploader("Upload dataset (CSV)", type="csv", help="Leave empty to use the bundled dataset.")
try:
    raw = _load_upload(up.getvalue()) if up else _load_default()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

with st.sidebar.expander("⚙️ Clustering model", expanded=False):
    algo = st.radio("Algorithm", ["K-Means", "Hierarchical (Ward)", "DBSCAN"], horizontal=False)
    k = st.slider("Number of clusters (k)", 2, 9, 5, disabled=(algo == "DBSCAN"))
    eps_mult, min_samples = 1.0, 12
    if algo == "DBSCAN":
        eps_mult = st.slider("eps multiplier", 0.3, 2.0, 0.6, 0.05,
                             help="Scales the auto-selected eps (90th pct of k-distance).")
        min_samples = st.slider("min_samples", 4, 40, 12)
    include_cat = st.checkbox("Include categorical features", True, help="Cuisine, segment and subregion (one-hot).")
    cat_weight = st.slider("Categorical weight", 0.05, 1.0, 0.35, 0.05, disabled=not include_cat)

with st.sidebar.expander("🎚️ Growth Potential Index weights", expanded=False):
    weights = {p: st.slider(p, 0, 100, DEFAULT_WEIGHTS[p], 5) for p in PILLARS}
    if sum(weights.values()) == 0:
        st.warning("All weights are zero; using equal weights.")
        weights = {p: 25 for p in PILLARS}
    hold_q = st.slider("'Hold' below GPI percentile", 10, 60, 35, 5) / 100
    agg_floor = st.slider("Min. healthy aggregator margin (%)", -10.0, 10.0, 0.0, 0.5) / 100

model = get_model(raw, algo, k, include_cat, cat_weight, eps_mult, min_samples)
scored, hold_thr = score_and_classify(model.feat, weights, hold_q, agg_floor)
cmap = ch.cluster_colors(scored["Cluster"].unique())

st.sidebar.markdown("### Filters")
sub_opts, cui_opts, seg_opts = (sorted(scored[c].unique()) for c in ["Subregion", "CuisineType", "Segment"])
f_sub = st.sidebar.multiselect("Subregion", sub_opts, default=sub_opts)
f_cui = st.sidebar.multiselect("Cuisine", cui_opts, default=cui_opts)
f_seg = st.sidebar.multiselect("Segment", seg_opts, default=seg_opts)
f_clu = st.sidebar.multiselect("Archetype", sorted(scored["Cluster"].unique()), default=sorted(scored["Cluster"].unique()))
f_str = st.sidebar.multiselect("Strategy", STRATEGIES, default=STRATEGIES)

mask = (scored.Subregion.isin(f_sub) & scored.CuisineType.isin(f_cui) & scored.Segment.isin(f_seg)
        & scored.Cluster.isin(f_clu) & scored.Strategy.isin(f_str))
F = scored[mask]
if F.empty:
    st.warning("No restaurants match the current filters.")
    st.stop()

st.sidebar.markdown("### Focus restaurant")
focus = st.sidebar.selectbox(
    "Used on the map, scorecard, radar and strategy tabs",
    options=list(F.sort_values("GPI", ascending=False).index),
    format_func=lambda i: f"{scored.at[i, 'RestaurantName']} (#{scored.at[i, 'RestaurantID']}) · GPI {scored.at[i, 'GPI']:.0f}",
)
R = scored.loc[focus]

dq_report = data_quality_report(raw)
dq_score = quality_score(dq_report)
class_conf = classification_confidence(scored, hold_thr, agg_floor)

st.sidebar.markdown("### 🔒 Reliability indicators")
dq_color = "#1B9E77" if dq_score >= 90 else "#E6A100" if dq_score >= 70 else "#C0392B"
st.sidebar.markdown(
    f'Data quality: <span class="conf-badge" style="background:{dq_color};color:white">{dq_score:.0f}/100</span>',
    unsafe_allow_html=True)
focus_conf = class_conf.loc[focus]
fc_color = "#1B9E77" if focus_conf >= 70 else "#E6A100" if focus_conf >= 40 else "#C0392B"
st.sidebar.markdown(
    f'Classification confidence: <span class="conf-badge" style="background:{fc_color};color:white">{focus_conf:.0f}/100</span>',
    unsafe_allow_html=True)
st.sidebar.caption(f"Showing **{len(F):,}** of {len(scored):,} restaurants. Scores are always computed against the full portfolio.")

st.title("Restaurant Growth Potential & Strategic Classification")
st.caption("Which restaurants are structurally positioned for sustainable growth, and how should they be classified? "
           "All insights below are backed by statistical analysis and quantified evidence.")

tabs = st.tabs(["📊 Overview", "🗺️ Cluster map", "🎯 Scorecards", "🕸️ Radar & compare", "💡 Strategy",
                "🔍 Growth drivers", "🔬 Insights & Evidence", "🧪 Methodology & data", "📄 Executive summary"])


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


def pct_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c] * 100
    return out


with tabs[0]:
    c = st.columns(5)
    c[0].metric("Restaurants", f"{len(F):,}")
    c[1].metric("Mean GPI", f"{F.GPI.mean():.1f}", f"{F.GPI.mean() - scored.GPI.mean():+.1f} vs portfolio")
    c[2].metric("Ready to optimize", f"{(F.Strategy == 'Optimize').mean():.0%}")
    c[3].metric("Portfolio net margin", f"{F.TotalNetProfit.sum() / F.TotalRevenue.sum():.1%}")
    c[4].metric("Aggregator dependence", f"{F.AggregatorDependence.mean():.0%}",
                help="Uber Eats + DoorDash share of orders")

    a, b = st.columns(2)
    a.plotly_chart(ch.strategy_donut(F), width="stretch")
    b.plotly_chart(ch.cluster_size_bar(F, cmap), width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(ch.share_heatmap(F, "Subregion", "Cluster", "Archetype mix by subregion (row %)"), width="stretch")
    b.plotly_chart(ch.stacked_strategy(F, "Segment", "Strategy mix by segment"), width="stretch")

    st.subheader("Archetype profiles")
    prof = (F.groupby("Cluster")
              .agg(Restaurants=("RestaurantID", "count"), GPI=("GPI", "mean"), GPI_std=("GPI", "std"),
                   GrowthFactor=("GrowthFactor", "mean"),
                   MonthlyOrders=("MonthlyOrders", "mean"), AOV=("AOV", "mean"), NetMargin=("NetMargin", "mean"),
                   CostRate=("CostRate", "mean"), AggregatorDependence=("AggregatorDependence", "mean"),
                   AggregatorMargin=("AggregatorMargin", "mean"), SD_Margin=("SD_Margin", "mean"),
                   DeliveryRadiusKM=("DeliveryRadiusKM", "mean"))
              .sort_values("GPI", ascending=False))
    show = pct_cols(prof, ["NetMargin", "CostRate", "AggregatorDependence", "AggregatorMargin", "SD_Margin"])
    st.dataframe(
        show.round(2), width="stretch",
        column_config={
            "GPI": st.column_config.ProgressColumn("GPI", min_value=0, max_value=100, format="%.1f"),
            "GPI_std": st.column_config.NumberColumn("GPI σ", format="%.1f", help="Standard deviation of GPI within the archetype"),
            "NetMargin": st.column_config.NumberColumn("Net margin", format="%.1f%%"),
            "CostRate": st.column_config.NumberColumn("COGS+OPEX", format="%.1f%%"),
            "AggregatorDependence": st.column_config.NumberColumn("Aggregator share", format="%.0f%%"),
            "AggregatorMargin": st.column_config.NumberColumn("Aggregator margin", format="%.1f%%"),
            "SD_Margin": st.column_config.NumberColumn("Self-delivery margin", format="%.1f%%"),
            "MonthlyOrders": st.column_config.NumberColumn("Orders / mo", format="%.0f"),
            "AOV": st.column_config.NumberColumn("AOV", format="$%.2f"),
            "DeliveryRadiusKM": st.column_config.NumberColumn("Radius km", format="%.1f"),
        })
    with st.expander("🔎 What do the archetypes mean? (detailed evidence-based descriptions)", expanded=False):
        for name in prof.index:
            base = name.split(" (")[0]
            desc = ARCHETYPE_DESC.get(base, "")
            n_in_cluster = int(prof.loc[name, "Restaurants"])
            avg_gpi = prof.loc[name, "GPI"]
            st.markdown(
                f'<div class="insight-card">'
                f'<b>{name}</b> &nbsp; <span style="color:#6C757D">({n_in_cluster:,} restaurants, mean GPI {avg_gpi:.1f})</span><br><br>'
                f'{desc}'
                f'</div>',
                unsafe_allow_html=True)
    st.plotly_chart(ch.gpi_box(F, cmap), width="stretch")

with tabs[1]:
    st.subheader("Restaurant cluster map")
    m1, m2, m3, m4 = st.columns(4)
    proj = m1.selectbox("Projection", ["PCA", "t-SNE", "UMAP"], help="UMAP requires `pip install umap-learn`.")
    color_by = m2.selectbox("Colour by", ["Cluster", "Strategy", "GPI", "Subregion", "CuisineType", "Segment",
                                          "GrowthFactor", "NetMargin", "AggregatorDependence"])
    size_by = m3.selectbox("Size by", ["None", "MonthlyOrders", "TotalRevenue", "Scale"])
    dims = m4.radio("Dimensions", [2, 3], horizontal=True, disabled=(proj != "PCA"))
    if proj != "PCA":
        dims = 2
    key = f"{algo}-{k}-{include_cat}-{cat_weight}-{len(raw)}"
    try:
        xy = get_projection(model.X, model.Z, proj, key)
        var = model.pca.explained_variance_ratio_ * 100
        labels = ((f"PC1 ({var[0]:.0f}%)", f"PC2 ({var[1]:.0f}%)", f"PC3 ({var[2]:.0f}%)")
                  if proj == "PCA" else (f"{proj} 1", f"{proj} 2", ""))
        fig = ch.cluster_scatter(F, xy, color_by, size_by, cmap, highlight_idx=focus, dims=dims,
                                 xyz=model.Z[:, :3] if dims == 3 else None, axis_labels=labels)
        st.plotly_chart(fig, width="stretch")
    except ImportError:
        st.error("UMAP is not installed. Run `pip install umap-learn` or choose PCA / t-SNE.")

    st.markdown("### Latent factors (PCA)")
    a, b = st.columns([1, 1])
    a.plotly_chart(ch.scree(model.pca), width="stretch")
    b.markdown("**What the leading components represent**")
    b.dataframe(model.comp_table, hide_index=True, width="stretch")
    st.caption(f"{model.n_use} components (≥85% of variance) feed the clustering. Components are auto-named from "
               "their dominant loadings: cost pressure, channel leverage, growth momentum and logistics reach.")
    st.plotly_chart(ch.loadings_heatmap(model.loadings), width="stretch")

with tabs[2]:
    st.subheader(f"Scorecard: {R.RestaurantName}")
    fc = class_conf.loc[focus]
    fc_label = "High confidence" if fc >= 70 else "Moderate confidence" if fc >= 40 else "Near boundary"
    fc_col = "#1B9E77" if fc >= 70 else "#E6A100" if fc >= 40 else "#C0392B"
    st.markdown(
        f"{badge(R.Strategy, STRATEGY_COLORS[R.Strategy])} &nbsp; {badge(R.Cluster, cmap[R.Cluster])} &nbsp; "
        f'<span class="conf-badge" style="background:{fc_col};color:white">🔒 {fc_label}</span> &nbsp; '
        f"**{R.CuisineType}** · {R.Segment} · {R.Subregion} · ID {R.RestaurantID} · "
        f"Rank **#{R.GPI_Rank}** of {len(scored):,}",
        unsafe_allow_html=True,
    )
    if fc < 40:
        st.info(f"⚠️ This restaurant's GPI ({R.GPI:.0f}) is close to a classification boundary. "
                f"Small changes in performance could shift it to a different strategy tier. "
                f"Treat the classification as indicative and review the underlying metrics.")

    lo, hi = scored.GPI.quantile(0.35), scored.GPI.quantile(0.70)
    g = st.columns([1.3, 1, 1, 1, 1])
    g[0].plotly_chart(ch.gauge(R.GPI, "Growth Potential Index", lo, hi, 260), width="stretch")
    for col, p in zip(g[1:], PILLARS):
        col.plotly_chart(ch.gauge(R[p], p, lo, hi, 260), width="stretch")

    peers = scored[scored.ClusterID == R.ClusterID]
    st.markdown("**Key performance indicators** (delta = versus archetype median, backed by peer comparison)")
    k_ = st.columns(5)
    k_[0].metric("Scale (orders × growth)", f"{R.Scale:,.0f}", f"{R.Scale - peers.Scale.median():+,.0f}")
    k_[1].metric("Cost discipline (COGS+OPEX)", f"{R.CostRate:.1%}",
                 f"{(R.CostRate - peers.CostRate.median()) * 100:+.1f} pts", delta_color="inverse")
    k_[2].metric("Aggregator dependence", f"{R.AggregatorDependence:.0%}",
                 f"{(R.AggregatorDependence - peers.AggregatorDependence.median()) * 100:+.0f} pts", delta_color="inverse")
    k_[3].metric("Expansion headroom (orders/km)", f"{R.ExpansionHeadroom:,.0f}",
                 f"{R.ExpansionHeadroom - peers.ExpansionHeadroom.median():+,.0f}")
    k_[4].metric("Revenue quality", f"{R.RevenueQuality:.2f}", f"{R.RevenueQuality - peers.RevenueQuality.median():+.2f}")

    with st.expander("📐 How reliable is this scorecard?"):
        st.markdown(
            f"- **Peer group size:** {len(peers):,} restaurants in the *{R.Cluster}* archetype\n"
            f"- **Classification confidence:** {fc:.0f}/100 — {'well separated from boundaries' if fc >= 70 else 'moderately separated' if fc >= 40 else 'close to a classification boundary — interpret with caution'}\n"
            f"- **GPI percentile within archetype:** {(peers['GPI'] <= R.GPI).mean():.0%}\n"
            f"- **Deltas** compare to the median of the {len(peers):,} peers in this archetype, providing a fair benchmark\n"
            f"- All pillar scores are 0-100 percentile-based; the radar and comparison views use the same scale"
        )

    a, b = st.columns(2)
    a.plotly_chart(ch.pillar_bars(R, PILLARS, weights), width="stretch")
    tbl = channel_table(R)
    b.plotly_chart(ch.channel_profit_bar(tbl), width="stretch")
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "Orders": st.column_config.NumberColumn(format="%d"),
        "Revenue": st.column_config.NumberColumn(format="$%.0f"),
        "Net profit": st.column_config.NumberColumn(format="$%.0f"),
        "Margin": st.column_config.NumberColumn(format="percent"),
        "Profit / order": st.column_config.NumberColumn(format="$%.2f"),
    })

    st.markdown("---")
    st.subheader("Leaderboard")
    l1, l2, l3 = st.columns(3)
    sort_by = l1.selectbox("Sort by", ["GPI"] + PILLARS + ["GrowthFactor", "NetMargin", "Scale"])
    order = l2.radio("Order", ["Highest first", "Lowest first"], horizontal=True)
    topn = l3.slider("Rows", 10, 200, 25, 5)
    lb = F.sort_values(sort_by, ascending=(order == "Lowest first")).head(topn)
    lb = lb[["RestaurantName", "RestaurantID", "Cluster", "Strategy", "Subregion", "CuisineType", "Segment", "GPI"]
            + PILLARS + ["GrowthFactor", "NetMargin", "AggregatorDependence"]]
    lb = pct_cols(lb, ["NetMargin", "AggregatorDependence"])
    st.dataframe(lb, hide_index=True, width="stretch", column_config={
        "GPI": st.column_config.ProgressColumn("GPI", min_value=0, max_value=100, format="%.0f"),
        **{p: st.column_config.NumberColumn(p, format="%.0f") for p in PILLARS},
        "GrowthFactor": st.column_config.NumberColumn("Growth ×", format="%.2f"),
        "NetMargin": st.column_config.NumberColumn("Net margin", format="%.1f%%"),
        "AggregatorDependence": st.column_config.NumberColumn("Aggregator share", format="%.0f%%"),
    })

with tabs[3]:
    st.subheader("Feature contribution radar & comparison")
    r1, r2 = st.columns([1, 2])
    mode = r1.radio("Compare", ["Focus restaurant vs its archetype", "Archetypes", "Restaurants head-to-head"])
    spokes = r1.radio("Radar spokes", ["Strategic KPIs (percentile)", "GPI pillars"])
    if spokes.startswith("Strategic"):
        cats, cols_ = list(KPI_COLUMNS.keys()), list(KPI_COLUMNS.values())
    else:
        cats, cols_ = PILLARS, PILLARS

    if mode.startswith("Focus"):
        peers_mean = scored[scored.ClusterID == R.ClusterID][cols_].mean()
        series = {R.RestaurantName: R[cols_].astype(float).values,
                  f"{R.Cluster} (avg)": peers_mean.values,
                  "Portfolio (avg)": scored[cols_].mean().values}
        r2.plotly_chart(ch.radar(cats, series, title="Focus restaurant vs archetype and portfolio"), width="stretch")
        gap = pd.DataFrame({"Spoke": cats, "Restaurant": R[cols_].astype(float).values,
                            "Archetype avg": peers_mean.values})
        gap["Gap vs archetype"] = gap["Restaurant"] - gap["Archetype avg"]
        r1.dataframe(gap.round(1), hide_index=True, width="stretch")
    elif mode == "Archetypes":
        g = F.groupby("Cluster")[cols_].mean()
        r2.plotly_chart(ch.radar(cats, {n: g.loc[n].values for n in g.index},
                                 colors=[cmap[n] for n in g.index], title="Archetype profiles", height=480),
                        width="stretch")
    else:
        opts = list(F.sort_values("GPI", ascending=False).index)
        picks = r1.multiselect("Choose up to 4 restaurants", opts, default=[focus], max_selections=4,
                               format_func=lambda i: f"{scored.at[i, 'RestaurantName']} (#{scored.at[i, 'RestaurantID']})")
        if picks:
            r2.plotly_chart(ch.radar(cats, {scored.at[i, "RestaurantName"] + f" #{scored.at[i, 'RestaurantID']}":
                                            scored.loc[i, cols_].astype(float).values for i in picks},
                                     title="Head-to-head", height=480), width="stretch")
            cmp_cols = ["Cluster", "Strategy", "GPI"] + PILLARS + ["GrowthFactor", "AOV", "MonthlyOrders", "NetMargin",
                                                                   "CostRate", "AggregatorDependence", "AggregatorMargin",
                                                                   "SD_Margin", "DeliveryRadiusKM"]
            st.dataframe(scored.loc[picks, cmp_cols].T.rename(
                columns=lambda i: scored.at[i, "RestaurantName"]).astype(str), width="stretch")
        else:
            r2.info("Pick at least one restaurant.")

    st.markdown("### Compare across clusters")
    metrics = ["GPI", "GrowthFactor", "NetMargin", "CostRate", "AggregatorDependence", "AggregatorMargin", "SD_Margin",
               "AOV", "MonthlyOrders", "DeliveryRadiusKM", "ExpansionHeadroom", "RevenueQuality"]
    a, b = st.columns([1, 2])
    mt = a.selectbox("Metric", metrics)
    import plotly.express as px

    fig = px.box(F, x="Cluster", y=mt, color="Cluster", color_discrete_map=cmap, points=False)
    fig.update_layout(height=380, showlegend=False, xaxis_title=None, margin=dict(l=10, r=10, t=20, b=10))
    b.plotly_chart(fig, width="stretch")
    st.plotly_chart(ch.channel_margin_bar(F), width="stretch")
    with st.expander("Parallel coordinates: trace restaurants across dimensions"):
        pc = st.multiselect("Dimensions", CLUSTER_FEATURES + ["GPI"],
                            default=["GPI", "GrowthFactor", "NetMargin", "CostRate", "AggregatorDependence", "SD_Margin",
                                     "DeliveryRadiusKM"])
        if len(pc) >= 2:
            st.plotly_chart(ch.parallel(F.sample(min(len(F), 600), random_state=1), pc), width="stretch")

with tabs[4]:
    st.subheader("Strategy recommendation panel")
    fc_s = class_conf.loc[focus]
    fc_label_s = "High" if fc_s >= 70 else "Moderate" if fc_s >= 40 else "Low"
    fc_col_s = "#1B9E77" if fc_s >= 70 else "#E6A100" if fc_s >= 40 else "#C0392B"
    st.markdown(
        f"**{R.RestaurantName}** &nbsp; {badge(R.Strategy, STRATEGY_COLORS[R.Strategy])} &nbsp; "
        f'<span class="conf-badge" style="background:{fc_col_s};color:white">🔒 {fc_label_s} confidence</span> &nbsp; '
        f"GPI **{R.GPI:.0f}** · archetype *{R.Cluster}*",
        unsafe_allow_html=True)
    a, b = st.columns([3, 2])
    with a:
        st.markdown("#### Evidence-backed recommended actions")
        st.caption("Each recommendation cites specific numbers, peer benchmarks, and estimated financial impact.")
        for act in recommend(R, scored[scored.ClusterID == R.ClusterID]):
            st.markdown(f"- {act}")
    with b:
        st.markdown("#### Classification rules")
        st.markdown(
            f"- **Hold / Stabilize**: GPI < **{hold_thr:.1f}** (bottom {hold_q:.0%})\n"
            f"- **Rebalance channels**: GPI above that, but aggregator margin < **{agg_floor:.1%}**\n"
            f"- **Optimize**: GPI above threshold with healthy aggregator margin")
        st.markdown("#### Classification confidence")
        st.markdown(
            f"This restaurant's classification confidence is **{fc_s:.0f}/100**. "
            f"{'The GPI is well above/below the threshold, making the classification robust.' if fc_s >= 70 else 'The GPI is moderately separated from the boundary.' if fc_s >= 40 else 'The GPI is close to a classification boundary — small performance changes could shift the strategy tier.'}")

    st.markdown("#### What-if simulator: channel shift")
    s1, s2, s3 = st.columns(3)
    shift = s1.slider("Move X% of aggregator orders to self-delivery", 0, 60, 20, 5)
    keep = s2.slider("Customer retention through the shift (%)", 40, 100, 80, 5)
    cut = s3.slider("Commission reduction on remaining aggregator revenue (pts)", 0.0, 10.0, 0.0, 0.5)
    res = simulate_shift(R, shift, keep, cut)
    w1, w2 = st.columns([2, 1])
    w1.plotly_chart(ch.whatif_waterfall(res), width="stretch")
    w2.metric("Orders moved / month", f"{res['orders_moved']:,.0f}")
    w2.metric("Net profit impact / month", f"${res['total_delta']:,.0f}",
              f"{res['total_delta'] / abs(res['baseline_profit']) * 100:+.1f}% vs current" if res["baseline_profit"] else None)
    w2.caption("First-order estimate using each channel's current profit per order.")

    st.markdown("---")
    st.markdown("#### Portfolio playbook")
    a, b = st.columns(2)
    play = pd.crosstab(F.Cluster, F.Strategy).reindex(columns=STRATEGIES, fill_value=0)
    a.markdown("**Restaurants by archetype and strategy**")
    a.dataframe(play, width="stretch")
    fig = px.bar(F.groupby(["Cluster", "Strategy"]).size().reset_index(name="n"), x="n", y="Cluster", color="Strategy",
                 orientation="h", color_discrete_map=STRATEGY_COLORS)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Restaurants", yaxis_title=None)
    b.plotly_chart(fig, width="stretch")

    st.markdown("#### Priority list")
    ps = st.radio("Show", STRATEGIES, horizontal=True)
    pl = F[F.Strategy == ps].copy()
    if ps == "Optimize":
        pl = pl.sort_values("GPI", ascending=False)
    elif ps == "Rebalance channels":
        pl["Aggregator loss/mo ($)"] = -(pl.UberEatsNetProfit + pl.DoorDashNetProfit).clip(upper=0)
        pl = pl.sort_values("Aggregator loss/mo ($)", ascending=False)
    else:
        pl = pl.sort_values("GPI")
    cols_show = ["RestaurantName", "RestaurantID", "Cluster", "Subregion", "CuisineType", "Segment", "GPI", "NetMargin",
                 "AggregatorMargin", "SD_Margin"] + (["Aggregator loss/mo ($)"] if ps == "Rebalance channels" else [])
    view = pct_cols(pl[cols_show].head(100), ["NetMargin", "AggregatorMargin", "SD_Margin"])
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "GPI": st.column_config.ProgressColumn("GPI", min_value=0, max_value=100, format="%.0f"),
        "NetMargin": st.column_config.NumberColumn("Net margin", format="%.1f%%"),
        "AggregatorMargin": st.column_config.NumberColumn("Aggregator margin", format="%.1f%%"),
        "SD_Margin": st.column_config.NumberColumn("Self-delivery margin", format="%.1f%%"),
    })
    st.caption(f"{len(pl):,} restaurants in this group; top 100 shown.")

with tabs[5]:
    st.subheader("Which levers matter most?")
    DRIVERS = ["GrowthFactor", "AOV", "MonthlyOrders", "COGSRate", "OPEXRate", "CommissionRate", "DeliveryRadiusKM",
               "DeliveryCostPerOrder", "SD_DeliveryTotalCost", "InStoreShare", "UE_share", "DD_share", "SD_share",
               "AggregatorDependence", "ChannelBalance", "ExpansionHeadroom"]
    tgt = st.selectbox("Outcome to explain", ["GPI", "NetMargin", "GrowthFactor", "TotalNetProfit", "AggregatorMargin"])
    feats = tuple(x for x in DRIVERS if x != tgt)
    imp, r2 = driver_importance(scored[list(feats) + [tgt]], tgt, feats)
    a, b = st.columns(2)
    a.plotly_chart(ch.importance_bar(imp, f"Random-forest importance for {tgt}"), width="stretch")
    corr = F[list(feats)].corrwith(F[tgt], method="spearman").dropna()
    b.plotly_chart(ch.corr_bar(corr, f"Correlation with {tgt} (filtered view)"), width="stretch")
    st.caption(f"Random forest 3-fold cross-validated R² = **{r2:.2f}**. Importance shows predictive relevance, not causation; "
               "GPI and margins are partly built from these same inputs, so treat this as a sensitivity map.")

    st.markdown("### Drill into a driver")
    d1, d2 = st.columns([1, 3])
    xv = d1.selectbox("Driver (x-axis)", list(imp.sort_values(ascending=False).index))
    d1.markdown(f"Spearman ρ = **{corr.get(xv, np.nan):+.2f}**")
    d2.plotly_chart(ch.driver_scatter(F.sample(min(len(F), 1200), random_state=2), xv, tgt, cmap=cmap), width="stretch")

    st.markdown("### GPI by business identity")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(ch.group_bar(F, "Subregion"), width="stretch")
    c2.plotly_chart(ch.group_bar(F, "CuisineType"), width="stretch")
    c3.plotly_chart(ch.group_bar(F, "Segment"), width="stretch")

with tabs[6]:
    st.subheader("🔬 Insights & Evidence")
    st.caption("Data quality auditing, statistical validation, confidence intervals, and evidence-backed insights — "
               "everything you need to trust the analysis.")

    st.markdown("### 📋 Data Quality Audit")
    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.plotly_chart(ch.data_quality_gauge(dq_score), width="stretch")
    dq2.metric("Restaurants", f"{dq_report['rows']:,}")
    dq3.metric("Completeness", f"{dq_report['mean_completeness']:.1%}")
    dq4.metric("Statistical outliers", f"{dq_report['total_outliers']:,}")

    a, b = st.columns(2)
    a.plotly_chart(ch.outlier_bar(dq_report["outlier_counts"]), width="stretch")
    b.plotly_chart(ch.completeness_bar(dq_report["completeness"]), width="stretch")

    if dq_report["highly_skewed"]:
        st.info(f"⚠️ **Highly skewed features** (|skew| > 2): {', '.join(dq_report['highly_skewed'])}. "
                f"These are handled by the Yeo-Johnson transform in the pipeline, so they do not bias the clustering or scoring.")
    if dq_report["duplicate_ids"] > 0:
        st.warning(f"⚠️ Found {dq_report['duplicate_ids']} duplicate RestaurantIDs — duplicates were removed during loading.")

    st.markdown("---")

    st.markdown("### 🧪 Statistical Validation of Archetypes")
    pairwise_tests, overall_test = cluster_separation_tests(scored, "GPI")
    v1, v2 = st.columns([1, 2])
    with v1:
        st.metric("Kruskal-Wallis H", f"{overall_test['H-statistic']:.2f}")
        st.metric("p-value", f"{overall_test['p-value']:.2e}")
        if overall_test["Significant"]:
            st.success("✅ **Significant.** Archetypes have genuinely different GPI distributions (p < 0.05). "
                       "The clustering captures real structural differences.")
        else:
            st.warning("⚠️ **Not significant.** Archetype GPI distributions overlap substantially — "
                       "treat group labels as tendencies, not hard categories.")
    with v2:
        st.markdown("**Pairwise archetype separation (Mann-Whitney U tests)**")
        st.dataframe(pairwise_tests, hide_index=True, width="stretch", column_config={
            "p-value": st.column_config.NumberColumn(format="%.4f"),
            "Effect size (r)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
        })
        n_sig = pairwise_tests["Significant (α=0.05)"].sum()
        n_total = len(pairwise_tests)
        st.caption(f"{n_sig} of {n_total} pairwise comparisons are statistically significant (α=0.05). "
                   f"Effect size (r) indicates practical significance: > 0.3 = medium, > 0.5 = large.")

    st.markdown("---")

    st.markdown("### 🔒 Classification Confidence")
    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(ch.classification_confidence_hist(class_conf), width="stretch")
    n_low_conf = (class_conf < 40).sum()
    cc2.metric("Near-boundary restaurants", f"{n_low_conf:,}",
               help="Restaurants with classification confidence < 40 — close to a strategy threshold")
    cc2.metric("Mean confidence", f"{class_conf.mean():.0f}/100")
    cc2.metric("Median confidence", f"{class_conf.median():.0f}/100")
    if n_low_conf > 0:
        cc2.caption(f"{n_low_conf:,} restaurants are close to a classification boundary. "
                    f"Their strategy assignment could change with small performance shifts. "
                    f"Review these manually before making investment decisions.")

    st.markdown("---")

    st.markdown("### 💡 Evidence-Backed Insights")
    insights = generate_insights(scored, dq_report, model.quality)
    for ins in insights:
        sev_color = ch.insight_severity_colors(ins["severity"])
        st.markdown(
            f'<div class="insight-card {ins["severity"]}">'
            f'<b>{ins["icon"]} {ins["title"]}</b>'
            f'<span class="conf-badge" style="background:{sev_color};color:white;float:right">{ins["category"]}</span>'
            f'<br><br>{ins["finding"]}'
            f'<div class="evidence"><b>Evidence:</b> {ins["evidence"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### ⚖️ Rank Stability Analysis")
    st.caption("How much would the rankings change if the GPI pillar weights were different? "
               "We perturb the weights ±15 points 50 times and measure rank volatility.")
    stab = weight_sensitivity(model.feat, weights)
    st.plotly_chart(ch.stability_scatter(stab, scored, cmap), width="stretch")
    s1, s2, s3 = st.columns(3)
    s1.metric("Mean rank stability", f"{stab['RankStability'].mean():.0f}/100")
    s2.metric("Most volatile restaurant", scored.at[stab['RankStability'].idxmin(), 'RestaurantName'])
    s3.metric("Max rank shift observed", f"{stab['MaxRankShift'].max():,} positions")
    st.caption("High stability (close to 100) means the restaurant's ranking is robust regardless of exact weight choices. "
               "Low stability suggests the score is sensitive to how much weight you place on each pillar.")


with tabs[7]:
    st.subheader("Method and model diagnostics")
    st.markdown(
        """
        **Pipeline:** engineered KPIs → Yeo-Johnson transform (skew) + standardisation (scale) → one-hot categoricals
        (down-weighted) → PCA (latent factors) → clustering on the leading components → archetype labelling →
        Growth Potential Index → rule-based strategy.

        **Growth Potential Index (0-100)**: weighted mean of four percentile-based pillars

        | Pillar | Built from |
        |---|---|
        | Growth Signals | GrowthFactor (50%), Scale = orders × growth (30%), AOV (20%) |
        | Cost Resilience | low COGS+OPEX (50%), total net margin (50%) |
        | Channel Balance | order-mix evenness (50%), low aggregator dependence (30%), aggregator margin (20%) |
        | Logistics Scalability | self-delivery margin (35%), low delivery cost/AOV (25%), radius (20%), expansion headroom (20%) |

        **KPI definitions:** *Scale* = MonthlyOrders × GrowthFactor · *Cost discipline* = COGS + OPEX rate ·
        *Aggregator dependence* = (Uber Eats + DoorDash orders) / all orders · *Expansion headroom* = orders per km of
        delivery radius (demand pressing on current reach) · *Revenue quality* = AOV × net margin × (1 − dispersion of channel margins).

        **Reliability framework:** Data quality is audited (completeness, outliers, skewness). GPI confidence intervals
        are computed via bootstrap resampling. Classification confidence measures distance from strategy thresholds.
        Rank stability is tested under weight perturbation. Cluster separation is validated with Kruskal-Wallis and
        pairwise Mann-Whitney U tests. See the *Insights & Evidence* tab for full details.
        """
    )
    q = model.quality
    qc = st.columns(4)
    qc[0].metric("Clusters found", q["Clusters"])
    qc[1].metric("Silhouette", "n/a" if pd.isna(q["Silhouette"]) else f"{q['Silhouette']:.3f}")
    qc[2].metric("Davies-Bouldin", "n/a" if pd.isna(q["Davies-Bouldin"]) else f"{q['Davies-Bouldin']:.3f}")
    qc[3].metric("Noise points", f"{q['Noise %']}%")
    if algo == "DBSCAN" and q["Clusters"] < 2:
        st.info("DBSCAN found ≤1 dense cluster: restaurants form a continuum rather than isolated groups. "
                "Lower the eps multiplier for finer structure; K-Means / Ward are better for segmentation here.")

    a, b = st.columns(2)
    a.plotly_chart(ch.k_scan_fig(get_kscan(model.Z[:, :model.n_use], f"{include_cat}-{cat_weight}-{len(raw)}")), width="stretch")
    idx = np.random.RandomState(0).choice(len(model.Z), size=min(120, len(model.Z)), replace=False)
    b.plotly_chart(ch.dendrogram(model.Z[idx][:, :model.n_use]), width="stretch")

    st.markdown("#### Algorithm robustness check")
    from sklearn.metrics import adjusted_rand_score
    Zc = model.Z[:, :model.n_use]
    rows, base = [], model.labels
    for name in ["K-Means", "Hierarchical (Ward)", "DBSCAN"]:
        lab = fit_cluster(Zc, name, k, 0.6 if name == "DBSCAN" else 1.0, 12)
        qq = cluster_quality(Zc, lab)
        rows.append({"Algorithm": name, "Clusters": qq["Clusters"], "Silhouette": qq["Silhouette"],
                     "Davies-Bouldin": qq["Davies-Bouldin"], "Noise %": qq["Noise %"],
                     "Agreement with current (ARI)": adjusted_rand_score(base, lab)})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.caption("ARI = 1 means identical partitions, ~0 means unrelated. Moderate agreement between K-Means and Ward "
               "indicates the archetypes are reasonably robust to the algorithm.")

    st.markdown("---")
    st.subheader("Data explorer & downloads")
    export_cols = (["RestaurantID", "RestaurantName", "CuisineType", "Segment", "Subregion", "Cluster", "GPI", "GPI_Rank",
                    "Strategy"] + PILLARS + list(KPI_COLUMNS.values()) +
                   ["Scale", "CostRate", "AggregatorDependence", "ExpansionHeadroom", "RevenueQuality", "NetMargin",
                    "AggregatorMargin", "SD_Margin", "InStoreMargin"])
    out = F[export_cols].sort_values("GPI", ascending=False)
    st.dataframe(out, hide_index=True, width="stretch", height=380)
    st.download_button("⬇️ Download classification (CSV)", out.to_csv(index=False).encode(),
                       "restaurant_classification.csv", "text/csv")
    with st.expander("Raw dataset preview"):
        st.dataframe(raw.head(200), width="stretch")

with tabs[8]:
    md = build_exec_summary(scored, model.quality, weights, hold_thr, algo)
    st.markdown(md)
    st.download_button("⬇️ Download summary (Markdown)", md.encode(), "executive_summary.md", "text/markdown")
    st.caption("The summary reflects the full portfolio and the current model / weight settings in the sidebar. "
               "It now includes statistical validation, portfolio economics, and tiered recommendations.")
