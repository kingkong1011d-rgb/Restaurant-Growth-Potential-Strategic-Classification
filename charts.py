from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage

from scoring import STRATEGY_COLORS

PALETTE = ["#0E7C86", "#E6A100", "#7B4FBF", "#D1495B", "#2E86AB", "#6C757D", "#8AB17D", "#F4A261", "#264653"]
OUTLIER_GREY = "#9AA5AB"
LAYOUT = dict(margin=dict(l=10, r=10, t=45, b=10), font=dict(size=13), legend=dict(title=None))


def cluster_colors(labels) -> dict:
    labs = sorted(set(labels))
    cmap, i = {}, 0
    for l in labs:
        if l.startswith("Outliers"):
            cmap[l] = OUTLIER_GREY
        else:
            cmap[l] = PALETTE[i % len(PALETTE)]
            i += 1
    return cmap


def color_discrete_for(col: str, values, cmap_cluster: dict):
    if col == "Cluster":
        return cmap_cluster
    if col == "Strategy":
        return STRATEGY_COLORS
    return None


def cluster_scatter(d: pd.DataFrame, xy: np.ndarray, color_by: str, size_by: str | None,
                    cmap_cluster: dict, highlight_idx=None, dims=2, xyz=None, axis_labels=("Dim 1", "Dim 2", "Dim 3")):
    p = d.copy()
    p["x"], p["y"] = xy[p.index, 0], xy[p.index, 1]
    if dims == 3:
        p["z"] = xyz[p.index, 2]
    hover = {"RestaurantName": True, "Cluster": True, "Strategy": True, "GPI": ":.1f", "Subregion": True,
             "CuisineType": True, "Segment": True, "GrowthFactor": ":.2f", "NetMargin": ":.1%",
             "x": False, "y": False}
    if dims == 3:
        hover["z"] = False
    kw = dict(color=color_by, hover_name="RestaurantName", hover_data=hover)
    if color_by in ("Cluster", "Strategy"):
        kw["color_discrete_map"] = color_discrete_for(color_by, None, cmap_cluster)
    elif pd.api.types.is_numeric_dtype(p[color_by]):
        kw["color_continuous_scale"] = "Tealrose" if color_by == "GPI" else "Viridis"
    if size_by and size_by != "None":
        p["_size"] = p[size_by].clip(lower=p[size_by].quantile(0.01)) - p[size_by].min() + 1e-6
        kw["size"] = "_size"
        kw["size_max"] = 16 if dims == 2 else 10
    if dims == 3:
        fig = px.scatter_3d(p, x="x", y="y", z="z", **kw)
        fig.update_layout(scene=dict(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1], zaxis_title=axis_labels[2]))
    else:
        fig = px.scatter(p, x="x", y="y", **kw)
        fig.update_xaxes(title=axis_labels[0], zeroline=False)
        fig.update_yaxes(title=axis_labels[1], zeroline=False)
    fig.update_traces(marker=dict(opacity=0.75, line=dict(width=0.3, color="white")))
    if highlight_idx is not None and highlight_idx in p.index:
        r = p.loc[highlight_idx]
        if dims == 3:
            fig.add_trace(go.Scatter3d(x=[r.x], y=[r.y], z=[r.z], mode="markers", name="Focus",
                                       marker=dict(size=9, color="black", symbol="diamond"),
                                       hovertext=r.RestaurantName))
        else:
            fig.add_trace(go.Scatter(x=[r.x], y=[r.y], mode="markers+text", name="Focus", text=[r.RestaurantName],
                                     textposition="top center",
                                     marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=3, color="black"))))
    fig.update_layout(height=620, **LAYOUT)
    return fig


def scree(pca, n_show=12):
    ev = pca.explained_variance_ratio_[:n_show] * 100
    fig = go.Figure()
    fig.add_bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev, name="Variance %", marker_color="#0E7C86")
    fig.add_scatter(x=[f"PC{i+1}" for i in range(len(ev))], y=np.cumsum(ev), name="Cumulative %",
                    mode="lines+markers", line=dict(color="#E6A100"))
    fig.update_layout(title="Variance explained", yaxis_title="%", height=340, **LAYOUT)
    return fig


def loadings_heatmap(loadings: pd.DataFrame, top: int = 18):
    keep = loadings.abs().max(axis=1).sort_values(ascending=False).head(top).index
    L = loadings.loc[keep].iloc[:, :4]
    fig = px.imshow(L, color_continuous_scale="RdBu_r", zmin=-0.5, zmax=0.5, aspect="auto", text_auto=".2f")
    fig.update_layout(title="Feature loadings on the leading components", height=520, **LAYOUT)
    return fig


def radar(categories, series: dict, colors: list | None = None, height=430, title=None):
    fig = go.Figure()
    cols = colors or PALETTE
    theta = list(categories) + [categories[0]]
    for i, (name, vals) in enumerate(series.items()):
        vals = list(vals) + [list(vals)[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=theta, name=name, fill="toself", opacity=0.55,
                                      line=dict(color=cols[i % len(cols)], width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], visible=True, tickfont=dict(size=10))),
                      height=height, title=title, **LAYOUT)
    return fig


def gauge(value, title, low_thr=35, high_thr=70, height=230):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), title={"text": title, "font": {"size": 15}},
        number={"font": {"size": 34}},
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#1B2A34", thickness=0.28),
                   steps=[dict(range=[0, low_thr], color="#F5C6C0"),
                          dict(range=[low_thr, high_thr], color="#FBE7B0"),
                          dict(range=[high_thr, 100], color="#BFE5D6")])))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def pillar_bars(row, pillars, weights: dict):
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    contrib = [row[p] * w[p] for p in pillars]
    fig = go.Figure(go.Bar(x=contrib, y=pillars, orientation="h", marker_color="#0E7C86",
                           text=[f"{c:.1f} pts  (score {row[p]:.0f} x {w[p]:.0%})" for c, p in zip(contrib, pillars)],
                           textposition="auto"))
    fig.update_layout(title="GPI decomposition: contribution by pillar", height=270,
                      xaxis_title="Points contributed to GPI", yaxis=dict(autorange="reversed"), **LAYOUT)
    return fig


def strategy_donut(d):
    c = d["Strategy"].value_counts().reset_index()
    c.columns = ["Strategy", "Restaurants"]
    fig = px.pie(c, names="Strategy", values="Restaurants", hole=0.55, color="Strategy",
                 color_discrete_map=STRATEGY_COLORS)
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(title="Strategic classification", height=340, showlegend=False, **{k: v for k, v in LAYOUT.items() if k != "legend"})
    return fig


def cluster_size_bar(d, cmap):
    c = d.groupby("Cluster").agg(Restaurants=("RestaurantID", "count"), GPI=("GPI", "mean")).reset_index()
    fig = px.bar(c.sort_values("GPI"), x="GPI", y="Cluster", orientation="h", color="Cluster",
                 color_discrete_map=cmap, text=c.sort_values("GPI")["Restaurants"].map(lambda v: f"n={v}"))
    fig.update_layout(title="Mean Growth Potential Index by archetype", height=340, showlegend=False,
                      xaxis=dict(range=[0, 100]), **{k: v for k, v in LAYOUT.items() if k != "legend"})
    return fig


def gpi_box(d, cmap):
    fig = px.box(d, x="Cluster", y="GPI", color="Cluster", color_discrete_map=cmap, points="outliers")
    fig.update_layout(title="GPI distribution by archetype", height=380, showlegend=False, xaxis_title=None,
                      **{k: v for k, v in LAYOUT.items() if k != "legend"})
    return fig


def share_heatmap(d, row, col, title):
    pv = pd.crosstab(d[row], d[col], normalize="index")
    fig = px.imshow(pv, text_auto=".0%", aspect="auto", color_continuous_scale="Teal")
    fig.update_layout(title=title, height=340, coloraxis_showscale=False, **LAYOUT)
    return fig


def stacked_strategy(d, by, title):
    c = d.groupby([by, "Strategy"]).size().reset_index(name="n")
    fig = px.bar(c, x=by, y="n", color="Strategy", color_discrete_map=STRATEGY_COLORS, barmode="stack")
    fig.update_layout(title=title, height=340, xaxis_title=None, yaxis_title="Restaurants", **LAYOUT)
    return fig


def channel_margin_bar(d, group="Cluster"):
    cols = {"InStoreMargin": "In-Store", "UE_Margin": "Uber Eats", "DD_Margin": "DoorDash", "SD_Margin": "Self-Delivery"}
    g = d.groupby(group)[list(cols)].mean().rename(columns=cols).reset_index().melt(group, var_name="Channel", value_name="Margin")
    fig = px.bar(g, x=group, y="Margin", color="Channel", barmode="group",
                 color_discrete_sequence=["#264653", "#D1495B", "#E6A100", "#0E7C86"])
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title="Average net margin by channel", height=380, xaxis_title=None, **LAYOUT)
    return fig


def channel_profit_bar(tbl: pd.DataFrame):
    fig = px.bar(tbl, x="Channel", y="Margin", color="Channel", text=tbl["Margin"].map(lambda v: f"{v:.1%}"),
                 color_discrete_sequence=["#264653", "#D1495B", "#E6A100", "#0E7C86"])
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title="Net margin by channel", height=300, showlegend=False, xaxis_title=None,
                      **{k: v for k, v in LAYOUT.items() if k != "legend"})
    return fig


def parallel(d, cols, color="GPI"):
    fig = px.parallel_coordinates(d, dimensions=cols, color=color, color_continuous_scale="Tealrose")
    fig.update_layout(height=430, margin=dict(l=60, r=40, t=40, b=20))
    return fig


def k_scan_fig(scan: pd.DataFrame):
    fig = go.Figure()
    fig.add_scatter(x=scan.k, y=scan.Silhouette, name="Silhouette (higher better)", mode="lines+markers",
                    line=dict(color="#0E7C86"))
    fig.add_scatter(x=scan.k, y=scan["Davies-Bouldin"], name="Davies-Bouldin (lower better)", mode="lines+markers",
                    line=dict(color="#D1495B"), yaxis="y2")
    fig.update_layout(title="Choosing k for K-Means", xaxis_title="k", yaxis=dict(title="Silhouette"),
                      yaxis2=dict(title="Davies-Bouldin", overlaying="y", side="right"), height=340, **LAYOUT)
    return fig


def dendrogram(Z_sample: np.ndarray):
    fig = ff.create_dendrogram(Z_sample, linkagefun=lambda x: linkage(x, "ward"),
                               color_threshold=None)
    fig.update_layout(title="Ward dendrogram (random sample of restaurants)", height=380, **LAYOUT)
    fig.update_xaxes(showticklabels=False)
    return fig


def importance_bar(imp: pd.Series, title):
    s = imp.sort_values().tail(15)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color="#0E7C86"))
    fig.update_layout(title=title, height=440, xaxis_title="Importance", **LAYOUT)
    return fig


def corr_bar(corr: pd.Series, title):
    s = corr.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=["#D1495B" if v < 0 else "#0E7C86" for v in s.values]))
    fig.update_layout(title=title, height=440, xaxis_title="Spearman correlation", **LAYOUT)
    return fig


def driver_scatter(d, x, y, color="Cluster", cmap=None):
    fig = px.scatter(d, x=x, y=y, color=color, color_discrete_map=cmap, hover_name="RestaurantName", opacity=0.6)
    ok = d[[x, y]].dropna()
    if len(ok) > 2:
        m, b = np.polyfit(ok[x], ok[y], 1)
        xs = np.linspace(ok[x].min(), ok[x].max(), 20)
        fig.add_scatter(x=xs, y=m * xs + b, mode="lines", name="Linear trend", line=dict(color="black", dash="dash"))
    fig.update_layout(height=420, **LAYOUT)
    return fig


def group_bar(d, by, metric="GPI"):
    g = d.groupby(by)[metric].mean().sort_values().reset_index()
    fig = px.bar(g, x=metric, y=by, orientation="h", color=metric, color_continuous_scale="Teal")
    fig.update_layout(title=f"Mean {metric} by {by}", height=300, coloraxis_showscale=False, **LAYOUT)
    return fig


def whatif_waterfall(res: dict):
    fig = go.Figure(go.Waterfall(
        x=["Current profit", "Shift to self-delivery", "Lower commission", "Projected profit"],
        measure=["absolute", "relative", "relative", "total"],
        y=[res["baseline_profit"], res["shift_delta"], res["commission_delta"], 0],
        connector={"line": {"color": "#9AA5AB"}},
        increasing={"marker": {"color": "#1B9E77"}}, decreasing={"marker": {"color": "#C0392B"}},
        totals={"marker": {"color": "#0E7C86"}}))
    fig.update_layout(title="Estimated monthly net profit impact ($)", height=340, **LAYOUT)
    return fig


def confidence_interval_scatter(ci_df: pd.DataFrame, scored: pd.DataFrame, cmap: dict, highlight_idx=None):
    d = ci_df.join(scored[["RestaurantName", "Cluster", "Strategy", "GPI_Rank"]])
    d = d.sort_values("GPI", ascending=False).head(150)

    fig = go.Figure()
    for cluster in d["Cluster"].unique():
        mask = d["Cluster"] == cluster
        sub = d[mask]
        fig.add_trace(go.Scatter(
            x=sub["GPI_Rank"], y=sub["GPI"],
            error_y=dict(type="data", symmetric=False,
                         array=(sub["GPI_hi"] - sub["GPI"]).values,
                         arrayminus=(sub["GPI"] - sub["GPI_lo"]).values,
                         thickness=1.2, width=0, color="rgba(100,100,100,0.3)"),
            mode="markers", name=cluster,
            marker=dict(color=cmap.get(cluster, "#999"), size=7, opacity=0.8),
            text=sub["RestaurantName"],
            hovertemplate="%{text}<br>GPI: %{y:.1f}<br>95% CI: [%{customdata[0]:.1f}, %{customdata[1]:.1f}]"
                          "<br>CI width: %{customdata[2]:.1f}<extra></extra>",
            customdata=sub[["GPI_lo", "GPI_hi", "CI_width"]].values,
        ))

    if highlight_idx is not None and highlight_idx in ci_df.index:
        r = ci_df.loc[highlight_idx]
        name = scored.at[highlight_idx, "RestaurantName"]
        fig.add_trace(go.Scatter(
            x=[scored.at[highlight_idx, "GPI_Rank"]], y=[r["GPI"]],
            mode="markers+text", name="Focus", text=[name], textposition="top center",
            marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=3, color="black")),
        ))

    fig.update_layout(title="GPI with 95% bootstrap confidence intervals (top 150)",
                      xaxis_title="GPI rank", yaxis_title="Growth Potential Index",
                      height=480, **LAYOUT)
    return fig


def data_quality_gauge(score: float):
    if score >= 90:
        bar_color = "#1B9E77"
    elif score >= 70:
        bar_color = "#E6A100"
    else:
        bar_color = "#C0392B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": "Data Quality Score", "font": {"size": 16}},
        number={"font": {"size": 40}, "suffix": "/100"},
        gauge=dict(axis=dict(range=[0, 100]),
                   bar=dict(color=bar_color, thickness=0.3),
                   steps=[dict(range=[0, 50], color="#FDECEA"),
                          dict(range=[50, 75], color="#FFF3CD"),
                          dict(range=[75, 100], color="#D4EDDA")])))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def classification_confidence_hist(conf: pd.Series):
    fig = px.histogram(conf, nbins=25, labels={"value": "Classification confidence (0-100)", "count": "Restaurants"},
                       color_discrete_sequence=["#0E7C86"])
    fig.update_layout(title="How confident are the strategy classifications?",
                      xaxis_title="Distance from classification boundary (normalised)",
                      yaxis_title="Number of restaurants",
                      height=340, showlegend=False, **LAYOUT)
    return fig


def stability_scatter(stab_df: pd.DataFrame, scored: pd.DataFrame, cmap: dict):
    d = stab_df.join(scored[["RestaurantName", "Cluster"]])
    fig = px.scatter(d, x="GPI", y="RankStability", color="Cluster", color_discrete_map=cmap,
                     hover_name="RestaurantName", size="MaxRankShift", size_max=15,
                     hover_data={"StdRank": ":.1f", "MaxRankShift": True, "BaseRank": True})
    fig.update_layout(title="How stable is each restaurant's ranking under weight perturbation?",
                      xaxis_title="GPI", yaxis_title="Rank stability (0-100, higher = more stable)",
                      height=440, **LAYOUT)
    return fig


def insight_severity_colors(severity: str) -> str:
    return {"good": "#1B9E77", "moderate": "#E6A100", "warning": "#C0392B"}.get(severity, "#6C757D")


def outlier_bar(outlier_counts: dict, top_n: int = 15):
    s = pd.Series(outlier_counts).sort_values(ascending=True).tail(top_n)
    colors = ["#C0392B" if v > 50 else "#E6A100" if v > 20 else "#0E7C86" for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors))
    fig.update_layout(title="Statistical outliers by feature (IQR method)",
                      xaxis_title="Number of outliers", height=400, **LAYOUT)
    return fig


def completeness_bar(completeness: dict, top_n: int = 20):
    s = pd.Series(completeness).sort_values().head(top_n) * 100
    colors = ["#C0392B" if v < 95 else "#E6A100" if v < 99 else "#1B9E77" for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors,
                           text=[f"{v:.1f}%" for v in s.values], textposition="auto"))
    fig.update_layout(title="Data completeness by feature",
                      xaxis_title="Completeness (%)", xaxis=dict(range=[80, 101]),
                      height=400, **LAYOUT)
    return fig
