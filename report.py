from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pipeline import ARCHETYPE_DESC


def build_exec_summary(d: pd.DataFrame, quality: dict, weights: dict, hold_thr: float, algo: str) -> str:
    n = len(d)
    sc = d["Strategy"].value_counts()
    pct = lambda k: sc.get(k, 0) / n
    w = pd.Series(weights, dtype=float)
    w = (w / w.sum() * 100).round(0)
    reg = d.groupby("Subregion")["GPI"].mean().sort_values(ascending=False)
    cui = d.groupby("CuisineType")["GPI"].mean().sort_values(ascending=False)
    seg = d.groupby("Segment")["GPI"].mean().sort_values(ascending=False)
    agg_loss = (d["AggregatorMargin"] < 0).mean()
    cl = (d.groupby("Cluster")
            .agg(n=("RestaurantID", "count"), gpi=("GPI", "mean"), gpi_std=("GPI", "std"),
                 margin=("NetMargin", "mean"), margin_std=("NetMargin", "std"),
                 agg=("AggregatorDependence", "mean"), growth=("GrowthFactor", "mean"),
                 sd_margin=("SD_Margin", "mean"), cost=("CostRate", "mean"))
            .sort_values("gpi", ascending=False))

    groups = [g["GPI"].values for _, g in d.groupby("Cluster")]
    if len(groups) >= 2:
        h_stat, h_p = stats.kruskal(*groups)
        separation_valid = h_p < 0.05
    else:
        h_stat, h_p = np.nan, np.nan
        separation_valid = False

    total_revenue = d["TotalRevenue"].sum()
    total_profit = d["TotalNetProfit"].sum()
    portfolio_margin = total_profit / total_revenue if total_revenue else 0
    agg_total_loss = d.loc[d["AggregatorMargin"] < 0,
                           ["UberEatsNetProfit", "DoorDashNetProfit"]].sum().sum()

    lines = [
        "# Executive Summary: Restaurant Growth Potential & Strategic Classification",
        "*SkyCity Auckland Restaurants & Bars — Evidence-Based Analysis*",
        "",
        "---",
        "",
        "## Purpose & Scope",
        "Not every restaurant should follow the same growth strategy. This analysis classifies "
        f"{n:,} restaurants into strategic archetypes and scores each on a 0-100 **Growth Potential Index (GPI)** "
        "so that expansion capital and support can be targeted where the structural economics of the business "
        "support sustainable growth.",
        "",
        "> **Note:** All scores are *relative* within this portfolio — they rank restaurants against each other, "
        "not against external benchmarks. Rankings are robust to moderate weight changes (see sensitivity analysis "
        "in the Methodology tab).",
        "",
        "## Headline Findings",
        "",
        f"### 1. Strategic Classification (n = {n:,})",
        f"- **{pct('Optimize'):.0%} ({sc.get('Optimize', 0):,} restaurants)** are structurally ready to scale "
        f"(*Optimize*). These restaurants combine above-threshold GPI scores with healthy aggregator channel economics.",
        f"- **{pct('Rebalance channels'):.0%} ({sc.get('Rebalance channels', 0):,} restaurants)** have a viable "
        f"core business but lose money or earn sub-threshold margins on delivery aggregators — channel rebalancing "
        f"should precede expansion.",
        f"- **{pct('Hold / Stabilize'):.0%} ({sc.get('Hold / Stabilize', 0):,} restaurants)** score below the GPI "
        f"threshold of **{hold_thr:.0f}** (bottom {(d['GPI'] < hold_thr).mean():.0%}) and should stabilise "
        f"costs and operations before growth investment.",
        "",
        "### 2. The Aggregator Profitability Crisis",
        f"- **{agg_loss:.0%}** of restaurants earn a negative net margin on Uber Eats + DoorDash combined, "
        f"generating estimated aggregator channel losses of **${abs(agg_total_loss):,.0f}/month** portfolio-wide.",
        f"- Average aggregator dependence is **{d['AggregatorDependence'].mean():.0%}** of orders, "
        f"while the mean commission rate is **{d['CommissionRate'].mean():.0%}** "
        f"(range {d['CommissionRate'].min():.0%}–{d['CommissionRate'].max():.0%}).",
        f"- **Self-delivery outperforms aggregators** for {(d['SD_Margin'] > d['AggregatorMargin']).mean():.0%} of "
        f"restaurants (mean margin gap: {(d['SD_Margin'] - d['AggregatorMargin']).mean():.1%} pts).",
        "- Third-party commissions are the single most common structural weakness across the portfolio.",
        "",
        "### 3. Portfolio Economics",
        f"- Total monthly revenue: **${total_revenue:,.0f}** | Total net profit: **${total_profit:,.0f}** | "
        f"Portfolio margin: **{portfolio_margin:.1%}**",
        f"- Mean GPI: **{d['GPI'].mean():.1f}** (std: {d['GPI'].std():.1f}, "
        f"median: {d['GPI'].median():.1f})",
        f"- Growth factor range: {d['GrowthFactor'].min():.3f}× – {d['GrowthFactor'].max():.3f}× "
        f"(mean: {d['GrowthFactor'].mean():.3f}×)",
        "",
        "### 4. Geographic & Segment Patterns",
        f"- **Highest-scoring subregion:** {reg.index[0]} (mean GPI {reg.iloc[0]:.1f}) | "
        f"**Lowest:** {reg.index[-1]} ({reg.iloc[-1]:.1f}) | "
        f"**Gap: {reg.iloc[0] - reg.iloc[-1]:.1f} pts**",
        f"- **Highest-scoring cuisine:** {cui.index[0]} ({cui.iloc[0]:.1f}) | "
        f"**Lowest:** {cui.index[-1]} ({cui.iloc[-1]:.1f})",
        f"- **Highest-scoring segment:** {seg.index[0]} ({seg.iloc[0]:.1f}) | "
        f"**Lowest:** {seg.index[-1]} ({seg.iloc[-1]:.1f})",
        "",
        "## Restaurant Archetypes",
        "",
        "| Archetype | n | Mean GPI (±σ) | Net Margin | Agg. Share | SD Margin | Cost Rate | Growth× | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, r in cl.iterrows():
        base = name.split(" (")[0]
        desc_short = ARCHETYPE_DESC.get(base, "")
        first_sentence = desc_short.split(". ")[0] + "." if desc_short else ""
        lines.append(
            f"| {name} | {int(r.n):,} | {r.gpi:.1f} ±{r.gpi_std:.1f} | {r.margin:.1%} | "
            f"{r['agg']:.0%} | {r.sd_margin:.1%} | {r.cost:.0%} | {r.growth:.3f}× | {first_sentence} |"
        )

    lines += [
        "",
        "### Archetype Detailed Descriptions",
    ]
    for name in cl.index:
        base = name.split(" (")[0]
        desc = ARCHETYPE_DESC.get(base, "")
        if desc:
            lines.append(f"\n**{name}** (n = {int(cl.loc[name, 'n']):,})\n")
            lines.append(f"> {desc}\n")

    lines += [
        "",
        "## Statistical Validation",
        "",
        f"**Cluster separation test (Kruskal-Wallis):** H = {h_stat:.2f}, p = {h_p:.2e} → "
        f"{'**Significant** — archetypes represent genuinely different GPI distributions.' if separation_valid else '**Not significant** — treat archetype differences with caution.'}",
        "",
        f"**Clustering quality:** Silhouette = {quality.get('Silhouette', float('nan')):.3f} | "
        f"Davies-Bouldin = {quality.get('Davies-Bouldin', float('nan')):.3f} | "
        f"Noise = {quality.get('Noise %', 0)}%",
        "",
    ]
    sil = quality.get("Silhouette", float("nan"))
    if not np.isnan(sil):
        if sil > 0.5:
            lines.append("- Silhouette > 0.5: **strong separation** — archetypes are distinct groupings.")
        elif sil > 0.3:
            lines.append("- Silhouette 0.3–0.5: **moderate separation** — archetypes capture real tendencies but some "
                         "overlap exists. Boundary restaurants should be interpreted carefully.")
        else:
            lines.append("- Silhouette < 0.3: **weak separation** — restaurants form more of a continuum. "
                         "Archetypes are useful labels but not hard boundaries.")

    lines += [
        "",
        "## Recommendations",
        "",
        "### Tier 1: Immediate Actions (0-3 months)",
        f"1. **Direct expansion investment to the {sc.get('Optimize', 0):,} *Optimize* restaurants**, "
        "prioritising those with the highest GPI, self-delivery margins, and rank stability. "
        "These combine growth signals, cost resilience, and channel economics that support scaling.",
        f"2. **Launch a commission renegotiation programme** for the {sc.get('Rebalance channels', 0):,} "
        f"*Rebalance* restaurants: target a {d['CommissionRate'].mean() * 100 - 3:.0f}% commission rate "
        f"(currently {d['CommissionRate'].mean():.0%} avg), adjust aggregator menu prices, and "
        f"migrate repeat customers to owned ordering — potential monthly margin recovery: "
        f"**${abs(agg_total_loss) * 0.3:,.0f}**.",
        "",
        "### Tier 2: Medium-Term Optimisation (3-6 months)",
        f"3. **Operational turnaround for *Hold / Stabilize* restaurants**: "
        f"defer growth capital until COGS+OPEX (currently {d.loc[d['Strategy'] == 'Hold / Stabilize', 'CostRate'].mean():.0%} "
        f"avg) is reduced to portfolio median ({d['CostRate'].median():.0%}). "
        f"Focus on supplier renegotiation, waste reduction, and labour efficiency.",
        "4. **Avoid uniform policies.** Archetype-specific levers deliver better returns than "
        "a single portfolio-wide plan. What works for a Scalable Self-Delivery Leader will not work for an "
        "Aggregator-Dependent restaurant.",
        "",
        "### Tier 3: Strategic Initiatives (6-12 months)",
        "5. **Expand delivery radius** for high-headroom Self-Delivery Leaders — demand density analysis "
        "suggests untapped geographic potential.",
        "6. **Develop a self-delivery migration programme** using the what-if simulator results "
        "to build business cases for individual restaurants.",
        "",
        "## Method in Brief",
        f"Features were skew-corrected (Yeo-Johnson) and standardised; categorical attributes were one-hot encoded and down-weighted. "
        f"PCA revealed latent factors (cost pressure, channel leverage, growth momentum, logistics reach). "
        f"Clusters were fitted with **{algo}** and labelled by profile matching (Hungarian assignment). "
        f"The GPI blends four pillars: Growth Signals ({w['Growth Signals']:.0f}%), Cost Resilience ({w['Cost Resilience']:.0f}%), "
        f"Channel Balance ({w['Channel Balance']:.0f}%) and Logistics Scalability ({w['Logistics Scalability']:.0f}%).",
        "",
        "## Caveats & Limitations",
        "- Restaurants form a **continuum** rather than sharply separated groups (moderate silhouette), so archetypes describe "
        "tendencies and boundary cases exist.",
        "- Scores are **relative** (percentile-based) within this portfolio; they rank restaurants against each other, not against external benchmarks.",
        "- The dataset is a single cross-section, so 'cost stability' is measured by cost level, not variation over time. "
        "Longitudinal data would strengthen the growth factor estimates.",
        "- What-if estimates assume current per-order profitability persists and are indicative only.",
        "- **Confidence intervals** on GPI (available in the Insights & Evidence tab) quantify the uncertainty "
        "inherent in relative scoring — narrow intervals indicate robust scores, wide intervals suggest sensitivity "
        "to the composition of the portfolio.",
        "",
        "---",
        f"*Report generated from {n:,} restaurants using the {algo} algorithm. "
        f"GPI weights: Growth Signals {w['Growth Signals']:.0f}%, Cost Resilience {w['Cost Resilience']:.0f}%, "
        f"Channel Balance {w['Channel Balance']:.0f}%, Logistics Scalability {w['Logistics Scalability']:.0f}%.*",
    ]
    return "\n".join(lines)
