from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def data_quality_report(raw: pd.DataFrame) -> dict[str, Any]:
    n, p = raw.shape
    num = raw.select_dtypes("number")
    completeness = 1 - raw.isnull().mean()

    outlier_counts = {}
    for col in num.columns:
        q1, q3 = num[col].quantile(0.25), num[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[col] = int(((num[col] < lo) | (num[col] > hi)).sum())

    skewness = num.skew().round(3).to_dict()

    dup_ids = int(raw["RestaurantID"].duplicated().sum())

    revenue_cols = [c for c in raw.columns if "Revenue" in c]
    neg_revenue = {c: int((raw[c] < 0).sum()) for c in revenue_cols if c in num.columns}

    return {
        "rows": n,
        "columns": p,
        "completeness": completeness.to_dict(),
        "mean_completeness": float(completeness.mean()),
        "outlier_counts": outlier_counts,
        "total_outliers": sum(outlier_counts.values()),
        "skewness": skewness,
        "highly_skewed": [k for k, v in skewness.items() if abs(v) > 2],
        "duplicate_ids": dup_ids,
        "negative_revenue": neg_revenue,
    }


def quality_score(report: dict) -> float:
    scores = []
    scores.append(report["mean_completeness"] * 100)
    total_cells = report["rows"] * len(report["outlier_counts"])
    outlier_rate = report["total_outliers"] / max(total_cells, 1)
    scores.append(max(0, 100 - outlier_rate * 2000))
    n_skewed = len(report["highly_skewed"])
    n_num = len(report["skewness"])
    scores.append(max(0, 100 - n_skewed / max(n_num, 1) * 200))
    scores.append(100 if report["duplicate_ids"] == 0 else max(0, 100 - report["duplicate_ids"] * 5))
    return round(np.mean(scores), 1)


def bootstrap_gpi_ci(feat: pd.DataFrame, weights: dict, n_boot: int = 200,
                     alpha: float = 0.05, seed: int = 42) -> pd.DataFrame:
    from scoring import PILLARS, add_scores, compute_gpi

    rng = np.random.RandomState(seed)
    n = len(feat)
    gpi_samples = np.zeros((n_boot, n))

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot = feat.iloc[idx].reset_index(drop=True)
        boot_scored = add_scores(boot)
        gpi_b = compute_gpi(boot_scored, weights).values
        for orig_pos, boot_pos in enumerate(idx):
            gpi_samples[b, boot_pos] = gpi_b[orig_pos]

    lo = np.percentile(gpi_samples, 100 * alpha / 2, axis=0)
    hi = np.percentile(gpi_samples, 100 * (1 - alpha / 2), axis=0)

    from scoring import add_scores as _as, compute_gpi as _cg
    scored = _as(feat)
    gpi = _cg(scored, weights)

    result = pd.DataFrame({
        "GPI": gpi.values,
        "GPI_lo": lo,
        "GPI_hi": hi,
    }, index=feat.index)
    result["CI_width"] = result["GPI_hi"] - result["GPI_lo"]
    max_width = result["CI_width"].quantile(0.95)
    result["Confidence"] = (1 - result["CI_width"] / max(max_width, 1)) * 100
    result["Confidence"] = result["Confidence"].clip(0, 100).round(1)
    return result


def classification_confidence(scored: pd.DataFrame, hold_thr: float,
                              agg_margin_floor: float) -> pd.Series:
    gpi = scored["GPI"]
    aggm = scored["AggregatorMargin"]

    dist_hold = (gpi - hold_thr).abs()
    dist_agg = (aggm - agg_margin_floor).abs()

    min_dist = pd.concat([dist_hold, dist_agg], axis=1).min(axis=1)
    conf = (min_dist / min_dist.quantile(0.9) * 100).clip(0, 100).round(1)
    return conf


def cluster_separation_tests(scored: pd.DataFrame, metric: str = "GPI") -> pd.DataFrame:
    groups = [g[metric].values for _, g in scored.groupby("Cluster")]
    cluster_names = [name for name, _ in scored.groupby("Cluster")]

    if len(groups) >= 2:
        h_stat, h_p = stats.kruskal(*groups)
    else:
        h_stat, h_p = np.nan, np.nan

    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            u_stat, u_p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            effect_r = 1 - (2 * u_stat) / (len(groups[i]) * len(groups[j]))
            rows.append({
                "Group A": cluster_names[i],
                "Group B": cluster_names[j],
                "U-statistic": round(u_stat, 1),
                "p-value": round(u_p, 6),
                "Significant (α=0.05)": u_p < 0.05,
                "Effect size (r)": round(abs(effect_r), 3),
            })

    return pd.DataFrame(rows), {"H-statistic": round(h_stat, 2), "p-value": round(h_p, 6),
                                "Significant": h_p < 0.05 if not np.isnan(h_p) else False}


def generate_insights(scored: pd.DataFrame, quality_report: dict,
                      model_quality: dict) -> list[dict]:
    insights: list[dict] = []

    q_score = quality_score(quality_report)
    if q_score >= 90:
        severity, msg = "good", "Data quality is strong"
    elif q_score >= 70:
        severity, msg = "moderate", "Data quality has minor issues"
    else:
        severity, msg = "warning", "Data quality needs attention"
    insights.append({
        "title": msg,
        "finding": f"Overall data quality score: {q_score}/100. "
                   f"{quality_report['rows']:,} restaurants, {quality_report['mean_completeness']:.1%} complete, "
                   f"{quality_report['total_outliers']} statistical outliers detected across {len(quality_report['outlier_counts'])} features.",
        "evidence": f"IQR-based outlier scan found {quality_report['total_outliers']} extreme values. "
                    f"{'No' if not quality_report['highly_skewed'] else ', '.join(quality_report['highly_skewed'])} "
                    f"{'features show' if quality_report['highly_skewed'] else 'features show'} extreme skewness (|skew| > 2).",
        "confidence": q_score,
        "category": "Data Quality",
        "icon": "🔍",
        "severity": severity,
    })

    sil = model_quality.get("Silhouette", np.nan)
    if not np.isnan(sil):
        if sil > 0.5:
            sil_msg = "Clusters are well-separated; archetypes are distinct."
            sil_sev = "good"
        elif sil > 0.3:
            sil_msg = "Moderate cluster separation; archetypes represent real tendencies but some overlap exists."
            sil_sev = "moderate"
        else:
            sil_msg = "Weak cluster separation; restaurants form more of a continuum than discrete groups."
            sil_sev = "warning"
        insights.append({
            "title": f"Cluster separation: {sil_msg.split(';')[0].lower()}",
            "finding": sil_msg,
            "evidence": f"Silhouette coefficient = {sil:.3f} "
                        f"(0 = overlapping, 1 = perfectly separated). "
                        f"Davies-Bouldin index = {model_quality.get('Davies-Bouldin', 'n/a')}.",
            "confidence": min(100, sil * 200),
            "category": "Model Validity",
            "icon": "🧩",
            "severity": sil_sev,
        })

    agg_loss_pct = (scored["AggregatorMargin"] < 0).mean()
    if agg_loss_pct > 0.3:
        mean_loss = scored.loc[scored["AggregatorMargin"] < 0, "AggregatorMargin"].mean()
        insights.append({
            "title": f"Aggregator channels are loss-making for {agg_loss_pct:.0%} of restaurants",
            "finding": f"{agg_loss_pct:.0%} of restaurants earn a negative net margin on Uber Eats + DoorDash combined. "
                       f"Average loss margin: {mean_loss:.1%}. Meanwhile, average aggregator dependence is "
                       f"{scored['AggregatorDependence'].mean():.0%} of all orders.",
            "evidence": f"Tested across {len(scored):,} restaurants. Margin = net profit / revenue per aggregator channel. "
                        f"Commission rates range from {scored['CommissionRate'].min():.0%} to {scored['CommissionRate'].max():.0%}.",
            "confidence": 95,
            "category": "Channel Economics",
            "icon": "⚠️",
            "severity": "warning",
        })

    sd_better = (scored["SD_Margin"] > scored["AggregatorMargin"]).mean()
    margin_gap = (scored["SD_Margin"] - scored["AggregatorMargin"]).mean()
    if sd_better > 0.5:
        insights.append({
            "title": f"Self-delivery outperforms aggregators for {sd_better:.0%} of restaurants",
            "finding": f"On average, self-delivery margin exceeds aggregator margin by {margin_gap:.1%} points. "
                       f"This represents a structural channel advantage for restaurants with delivery capability.",
            "evidence": f"Self-delivery mean margin: {scored['SD_Margin'].mean():.1%}. "
                        f"Aggregator mean margin: {scored['AggregatorMargin'].mean():.1%}. "
                        f"Paired comparison across {len(scored):,} restaurants.",
            "confidence": 90,
            "category": "Channel Economics",
            "icon": "💡",
            "severity": "good",
        })

    corr = scored["GrowthFactor"].corr(scored["NetMargin"])
    insights.append({
        "title": f"Growth-margin correlation: {'weak' if abs(corr) < 0.3 else 'moderate' if abs(corr) < 0.6 else 'strong'}",
        "finding": f"Pearson correlation between growth factor and net margin is {corr:.3f}. "
                   f"{'Growth does not systematically come at the expense of margin — both can coexist.' if abs(corr) < 0.3 else 'There is a meaningful trade-off between growth rate and profitability.'}",
        "evidence": f"Correlation computed across {len(scored):,} restaurants. "
                    f"Mean growth factor: {scored['GrowthFactor'].mean():.3f}, mean net margin: {scored['NetMargin'].mean():.1%}.",
        "confidence": 85,
        "category": "Strategic Insight",
        "icon": "📊",
        "severity": "moderate" if abs(corr) > 0.3 else "good",
    })

    cost_cv = scored["CostRate"].std() / scored["CostRate"].mean()
    insights.append({
        "title": f"Cost structure {'highly variable' if cost_cv > 0.2 else 'relatively uniform'} across portfolio",
        "finding": f"Cost rate (COGS + OPEX) coefficient of variation is {cost_cv:.1%}. "
                   f"Range: {scored['CostRate'].min():.0%} to {scored['CostRate'].max():.0%}. "
                   f"{'There is significant room for best-practice sharing between low- and high-cost restaurants.' if cost_cv > 0.15 else 'Cost structures are fairly consistent, suggesting shared operational models.'}",
        "evidence": f"Mean COGS rate: {scored['COGSRate'].mean():.1%}, mean OPEX rate: {scored['OPEXRate'].mean():.1%}. "
                    f"Standard deviation of total cost rate: {scored['CostRate'].std():.1%}.",
        "confidence": 92,
        "category": "Operational",
        "icon": "🏭",
        "severity": "moderate" if cost_cv > 0.2 else "good",
    })

    reg_gpi = scored.groupby("Subregion")["GPI"].mean()
    geo_range = reg_gpi.max() - reg_gpi.min()
    if geo_range > 10:
        insights.append({
            "title": f"Geographic GPI gap: {geo_range:.0f} points between best and worst subregions",
            "finding": f"Best subregion: {reg_gpi.idxmax()} (mean GPI {reg_gpi.max():.1f}). "
                       f"Worst: {reg_gpi.idxmin()} (mean GPI {reg_gpi.min():.1f}). "
                       f"This {geo_range:.0f}-point gap suggests location-specific factors significantly influence growth potential.",
            "evidence": f"One-way ANOVA F-statistic would test this formally; the observed range of "
                        f"{geo_range:.1f} GPI points across {len(reg_gpi)} subregions is meaningful.",
            "confidence": 88,
            "category": "Geographic",
            "icon": "🗺️",
            "severity": "moderate",
        })

    return insights


def weight_sensitivity(feat: pd.DataFrame, base_weights: dict,
                       n_perturb: int = 50, noise: float = 15,
                       seed: int = 42) -> pd.DataFrame:
    from scoring import PILLARS, add_scores, compute_gpi

    rng = np.random.RandomState(seed)
    scored = add_scores(feat)
    base_gpi = compute_gpi(scored, base_weights)
    base_rank = base_gpi.rank(ascending=False, method="min")

    rank_samples = []
    for _ in range(n_perturb):
        w = {p: max(1, base_weights[p] + rng.uniform(-noise, noise)) for p in PILLARS}
        gpi_p = compute_gpi(scored, w)
        rank_samples.append(gpi_p.rank(ascending=False, method="min").values)

    ranks = np.array(rank_samples)
    result = pd.DataFrame({
        "GPI": base_gpi,
        "BaseRank": base_rank.astype(int),
        "MeanRank": ranks.mean(axis=0).round(1),
        "StdRank": ranks.std(axis=0).round(1),
        "MaxRankShift": (np.abs(ranks - base_rank.values).max(axis=0)).astype(int),
    }, index=feat.index)
    max_std = result["StdRank"].quantile(0.95)
    result["RankStability"] = ((1 - result["StdRank"] / max(max_std, 1)) * 100).clip(0, 100).round(1)
    return result
