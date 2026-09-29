from __future__ import annotations

import numpy as np
import pandas as pd

PILLARS = ["Growth Signals", "Cost Resilience", "Channel Balance", "Logistics Scalability"]
DEFAULT_WEIGHTS = {"Growth Signals": 30, "Cost Resilience": 30, "Channel Balance": 20, "Logistics Scalability": 20}

STRATEGIES = ["Optimize", "Rebalance channels", "Hold / Stabilize"]
STRATEGY_COLORS = {"Optimize": "#1B9E77", "Rebalance channels": "#E6A100", "Hold / Stabilize": "#C0392B"}

KPI_COLUMNS = {
    "Scale": "KPI_Scale",
    "Cost Discipline": "KPI_CostDiscipline",
    "Aggregator Independence": "KPI_AggIndependence",
    "Expansion Headroom": "KPI_Headroom",
    "Revenue Quality": "KPI_RevenueQuality",
    "Growth Momentum": "KPI_Growth",
}


def pct(s: pd.Series, invert: bool = False) -> pd.Series:
    r = s.rank(pct=True, method="average") * 100
    return 100 - r + (100 / len(s)) if invert else r


def add_scores(feat: pd.DataFrame) -> pd.DataFrame:
    d = feat.copy()

    d["KPI_Scale"] = pct(d["Scale"])
    d["KPI_CostDiscipline"] = pct(d["CostRate"], invert=True)
    d["KPI_AggIndependence"] = pct(d["AggregatorDependence"], invert=True)
    d["KPI_Headroom"] = pct(d["ExpansionHeadroom"])
    d["KPI_RevenueQuality"] = pct(d["RevenueQuality"])
    d["KPI_Growth"] = pct(d["GrowthFactor"])

    d["Growth Signals"] = 0.5 * pct(d["GrowthFactor"]) + 0.3 * pct(d["Scale"]) + 0.2 * pct(d["AOV"])
    d["Cost Resilience"] = 0.5 * pct(d["CostRate"], invert=True) + 0.5 * pct(d["NetMargin"])
    d["Channel Balance"] = (0.5 * pct(d["ChannelBalance"]) + 0.3 * pct(d["AggregatorDependence"], invert=True)
                            + 0.2 * pct(d["AggregatorMargin"]))
    d["Logistics Scalability"] = (0.35 * pct(d["SD_Margin"]) + 0.25 * pct(d["DeliveryCostPctAOV"], invert=True)
                                  + 0.20 * pct(d["DeliveryRadiusKM"]) + 0.20 * pct(d["ExpansionHeadroom"]))
    return d


def compute_gpi(d: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1 / len(w), index=w.index)
    return (d[PILLARS] * w).sum(axis=1).round(2)


def classify(d: pd.DataFrame, hold_q: float = 0.35, agg_margin_floor: float = 0.0):
    hold_thr = d["GPI"].quantile(hold_q)
    strat = np.where(
        d["GPI"] < hold_thr, "Hold / Stabilize",
        np.where(d["AggregatorMargin"] < agg_margin_floor, "Rebalance channels", "Optimize"),
    )
    return pd.Series(strat, index=d.index), hold_thr


def score_and_classify(feat, weights, hold_q=0.35, agg_margin_floor=0.0):
    d = add_scores(feat)
    d["GPI"] = compute_gpi(d, weights)
    d["GPI_Rank"] = d["GPI"].rank(ascending=False, method="min").astype(int)
    d["Strategy"], thr = classify(d, hold_q, agg_margin_floor)
    d["GPI_Tier"] = pd.cut(d["GPI"], [-1, d["GPI"].quantile(0.35), d["GPI"].quantile(0.7), 101],
                           labels=["Low", "Medium", "High"])
    return d, thr


def recommend(row: pd.Series, peers: pd.DataFrame) -> list[str]:
    acts: list[str] = []
    weakest = row[PILLARS].astype(float).idxmin()
    strongest = row[PILLARS].astype(float).idxmax()
    med = peers.median(numeric_only=True)
    p25 = peers.quantile(0.25, numeric_only=True)
    p75 = peers.quantile(0.75, numeric_only=True)
    pillar_gap = {p: float(row[p] - med.get(p, row[p])) for p in PILLARS}

    if row["Strategy"] == "Optimize":
        rank_in_peers = int((peers["GPI"] <= row["GPI"]).sum())
        acts.append(
            f"**✅ Invest to scale.** GPI **{row['GPI']:.0f}** ranks in the top "
            f"**{(1 - rank_in_peers / len(peers)) * 100:.0f}%** of its archetype ({len(peers):,} peers). "
            f"Strongest pillar: *{strongest}* ({row[strongest]:.0f}/100, "
            f"+{pillar_gap[strongest]:+.0f} vs archetype median). "
            f"This restaurant's structural economics support further investment in capacity and marketing."
        )
    elif row["Strategy"] == "Rebalance channels":
        monthly_agg_loss = (row.get("UberEatsNetProfit", 0) + row.get("DoorDashNetProfit", 0))
        if monthly_agg_loss < 0:
            acts.append(
                f"**⚠️ Rebalance channels before expanding.** Aggregator channels lose "
                f"**${abs(monthly_agg_loss):,.0f}/month** (margin {row['AggregatorMargin']:.1%}) while "
                f"self-delivery earns {row['SD_Margin']:.1%} and in-store {row['InStoreMargin']:.1%}. "
                f"Annualised, this channel imbalance costs ~**${abs(monthly_agg_loss) * 12:,.0f}/year** — "
                f"fix before committing growth capital."
            )
        else:
            acts.append(
                f"**⚠️ Rebalance channels.** Aggregator margin ({row['AggregatorMargin']:.1%}) is below the "
                f"healthy threshold while self-delivery ({row['SD_Margin']:.1%}) and in-store ({row['InStoreMargin']:.1%}) "
                f"perform better. Shifting order mix can unlock margin."
            )
    else:
        below_med = {p: pillar_gap[p] for p in PILLARS if pillar_gap[p] < -5}
        deficit_text = ", ".join(f"*{p}* ({pillar_gap[p]:+.0f})" for p in below_med) if below_med else f"*{weakest}*"
        acts.append(
            f"**🔴 Hold and stabilise.** GPI **{row['GPI']:.0f}** is in the bottom tier. "
            f"Key deficits vs archetype median: {deficit_text}. "
            f"Fix operational fundamentals before committing growth capital — "
            f"every $1 invested here yields lower expected return than the portfolio average."
        )

    if row["AggregatorMargin"] < 0:
        agg_orders = row.get("UberEatsOrders", 0) + row.get("DoorDashOrders", 0)
        agg_loss_po = abs(row.get("UberEatsNetProfit", 0) + row.get("DoorDashNetProfit", 0)) / max(agg_orders, 1)
        acts.append(
            f"📉 **Aggregator orders are loss-making.** Commission rate of {row['CommissionRate']:.0%} on "
            f"{row['AggregatorDependence']:.0%} of all orders ({agg_orders:,.0f}/month) produces a loss of "
            f"**${agg_loss_po:.2f} per order**. Peer median commission: {med.get('CommissionRate', 0.30):.0%}. "
            f"*Three levers:* (1) renegotiate commission by ≥{max(1, (row['CommissionRate'] - 0.25) * 100):.0f} pts, "
            f"(2) raise aggregator menu prices by {max(5, row['CommissionRate'] * 100 - 20):.0f}%, "
            f"(3) enforce minimum basket size of ≥${row['AOV'] * 1.2:.0f}."
        )

    margin_gap_sd_agg = row["SD_Margin"] - row["AggregatorMargin"]
    if margin_gap_sd_agg > 0.05 and row["AggregatorDependence"] > med.get("AggregatorDependence", 0.5):
        agg_orders = row.get("UberEatsOrders", 0) + row.get("DoorDashOrders", 0)
        potential_monthly = agg_orders * 0.2 * margin_gap_sd_agg * row["AOV"]
        acts.append(
            f"💡 **Self-delivery margin advantage: +{margin_gap_sd_agg:.1%} pts vs aggregators.** "
            f"Self-delivery: {row['SD_Margin']:.1%}, aggregators: {row['AggregatorMargin']:.1%}. "
            f"Migrating just 20% of aggregator orders ({agg_orders * 0.2:,.0f}/month) at 80% retention "
            f"could add ~**${potential_monthly:,.0f}/month** in margin. "
            f"Use the what-if simulator below to model scenarios."
        )

    if row["CostRate"] > med.get("CostRate", 0.65) + 0.05:
        cost_excess = row["CostRate"] - med.get("CostRate", 0.65)
        saving_potential = cost_excess * row.get("TotalRevenue", 0)
        cogs_vs = "above" if row["COGSRate"] > med.get("COGSRate", 0.28) else "in line with"
        opex_vs = "above" if row["OPEXRate"] > med.get("OPEXRate", 0.40) else "in line with"
        acts.append(
            f"🏭 **Cost rate is {row['CostRate']:.0%} vs peer median {med.get('CostRate', 0.65):.0%}** "
            f"(+{cost_excess:.0%} pts excess). COGS ({row['COGSRate']:.0%}) is {cogs_vs} peers; "
            f"OPEX ({row['OPEXRate']:.0%}) is {opex_vs} peers. "
            f"Closing the gap to median could save ~**${saving_potential:,.0f}/month**. "
            f"*Priority levers:* {'supplier renegotiation and waste reduction' if row['COGSRate'] > med.get('COGSRate', 0.28) else 'labour rostering and overhead optimisation'}."
        )

    if row["DeliveryCostPctAOV"] > med.get("DeliveryCostPctAOV", 0.08) * 1.3:
        acts.append(
            f"🚚 **Delivery cost is high:** ${row['DeliveryCostPerOrder']:.2f}/order "
            f"({row['DeliveryCostPctAOV']:.1%} of AOV vs peer median {med.get('DeliveryCostPctAOV', 0.08):.1%}). "
            f"Consider tighter delivery zones, batched routes, or minimum order thresholds to improve unit economics."
        )

    if row["ExpansionHeadroom"] > med.get("ExpansionHeadroom", 100) * 1.5 and row["DeliveryRadiusKM"] < med.get("DeliveryRadiusKM", 10):
        acts.append(
            f"📍 **Untapped expansion potential.** Demand density ({row['ExpansionHeadroom']:,.0f} orders/km) is "
            f"{row['ExpansionHeadroom'] / med.get('ExpansionHeadroom', 100):.1f}× the archetype median "
            f"within a {row['DeliveryRadiusKM']:.0f} km radius (peer avg {med.get('DeliveryRadiusKM', 10):.0f} km). "
            f"Piloting a {row['DeliveryRadiusKM'] + 3:.0f} km radius could absorb latent demand."
        )

    if row["GrowthFactor"] >= 1.04 and row["NetMargin"] < med.get("NetMargin", 0):
        margin_deficit = med.get("NetMargin", 0) - row["NetMargin"]
        acts.append(
            f"📈 **Growth sustainability concern.** Growth factor {row['GrowthFactor']:.2f}× is strong but "
            f"net margin ({row['NetMargin']:.1%}) is {margin_deficit:.1%} pts below peers ({med.get('NetMargin', 0):.1%}). "
            f"Growth may be subsidised by unsustainable pricing or promotion. "
            f"*Action:* set margin guardrails and monitor unit economics monthly."
        )

    if pillar_gap[weakest] < -10:
        acts.append(
            f"🎯 **Biggest improvement opportunity: *{weakest}*** (score {row[weakest]:.0f}, "
            f"{pillar_gap[weakest]:+.0f} vs archetype median). "
            f"Improving this pillar to the archetype median would add ~{abs(pillar_gap[weakest]) * 0.25:.0f} pts "
            f"to GPI (at current weights)."
        )

    return acts


def channel_table(row: pd.Series) -> pd.DataFrame:
    ch = {"In-Store": "InStore", "Uber Eats": "UberEats", "DoorDash": "DoorDash", "Self-Delivery": "SelfDelivery"}
    rows = []
    for label, p in ch.items():
        orders, rev, prof = row[f"{p}Orders"], row[f"{p}Revenue"], row[f"{p}NetProfit"]
        rows.append({"Channel": label, "Orders": orders, "Revenue": rev, "Net profit": prof,
                     "Margin": prof / rev if rev else 0, "Profit / order": prof / orders if orders else 0})
    return pd.DataFrame(rows)


def simulate_shift(row: pd.Series, shift_pct: float, retention: float, commission_cut_pts: float):
    ch = channel_table(row).set_index("Channel")
    agg_orders = row["UberEatsOrders"] + row["DoorDashOrders"]
    agg_profit_po = (row["UberEatsNetProfit"] + row["DoorDashNetProfit"]) / agg_orders if agg_orders else 0
    sd_po = ch.loc["Self-Delivery", "Profit / order"]
    moved = agg_orders * shift_pct / 100
    shift_delta = moved * retention / 100 * sd_po - moved * agg_profit_po
    agg_rev = row["UberEatsRevenue"] + row["DoorDashRevenue"]
    remaining_rev = agg_rev * (1 - shift_pct / 100)
    comm_delta = remaining_rev * commission_cut_pts / 100
    return {"orders_moved": moved, "shift_delta": shift_delta, "commission_delta": comm_delta,
            "total_delta": shift_delta + comm_delta, "baseline_profit": row["TotalNetProfit"]}
