from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer

ALIASES = {
    "InStoreOrdersCount": "InStoreOrders",
    "UberEatsOrdersCount": "UberEatsOrders",
    "DoorDashOrdersCount": "DoorDashOrders",
    "SelfDeliveryOrdersCount": "SelfDeliveryOrders",
    "DeliveryCostOrder": "DeliveryCostPerOrder",
}

CATEGORICAL = ["CuisineType", "Segment", "Subregion"]

REQUIRED = [
    "RestaurantID", "RestaurantName", "CuisineType", "Segment", "Subregion",
    "GrowthFactor", "AOV", "MonthlyOrders",
    "InStoreOrders", "UberEatsOrders", "DoorDashOrders", "SelfDeliveryOrders",
    "InStoreRevenue", "UberEatsRevenue", "DoorDashRevenue", "SelfDeliveryRevenue",
    "COGSRate", "OPEXRate", "CommissionRate", "DeliveryRadiusKM",
    "DeliveryCostPerOrder", "SD_DeliveryTotalCost",
    "InStoreNetProfit", "UberEatsNetProfit", "DoorDashNetProfit", "SelfDeliveryNetProfit",
    "InStoreShare", "UE_share", "DD_share", "SD_share",
]


def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=ALIASES)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.drop_duplicates(subset="RestaurantID").reset_index(drop=True)
    num_cols = [c for c in REQUIRED if c not in CATEGORICAL + ["RestaurantName"]]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=num_cols).reset_index(drop=True)
    return df


def _safe_div(a, b):
    b = np.where(np.asarray(b) == 0, np.nan, b)
    return pd.Series(np.asarray(a) / b, index=getattr(a, "index", None)).fillna(0.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["TotalRevenue"] = d[["InStoreRevenue", "UberEatsRevenue", "DoorDashRevenue", "SelfDeliveryRevenue"]].sum(axis=1)
    d["TotalNetProfit"] = d[["InStoreNetProfit", "UberEatsNetProfit", "DoorDashNetProfit", "SelfDeliveryNetProfit"]].sum(axis=1)
    d["NetMargin"] = _safe_div(d["TotalNetProfit"], d["TotalRevenue"])

    d["Scale"] = d["MonthlyOrders"] * d["GrowthFactor"]
    d["ProjectedAnnualGrowth"] = d["GrowthFactor"] ** 12 - 1

    d["CostRate"] = d["COGSRate"] + d["OPEXRate"]

    d["AggregatorDependence"] = _safe_div(d["UberEatsOrders"] + d["DoorDashOrders"], d["MonthlyOrders"])

    d["InStoreMargin"] = _safe_div(d["InStoreNetProfit"], d["InStoreRevenue"])
    d["UE_Margin"] = _safe_div(d["UberEatsNetProfit"], d["UberEatsRevenue"])
    d["DD_Margin"] = _safe_div(d["DoorDashNetProfit"], d["DoorDashRevenue"])
    d["SD_Margin"] = _safe_div(d["SelfDeliveryNetProfit"], d["SelfDeliveryRevenue"])
    d["AggregatorMargin"] = _safe_div(
        d["UberEatsNetProfit"] + d["DoorDashNetProfit"],
        d["UberEatsRevenue"] + d["DoorDashRevenue"],
    )
    margins = d[["InStoreMargin", "UE_Margin", "DD_Margin", "SD_Margin"]]
    d["ChannelMarginStd"] = margins.std(axis=1, ddof=0)

    d["RevenueQuality"] = d["AOV"] * d["NetMargin"] * (1 - d["ChannelMarginStd"].clip(0, 1))

    d["ExpansionHeadroom"] = d["MonthlyOrders"] / d["DeliveryRadiusKM"]
    d["DeliveryCostPctAOV"] = d["DeliveryCostPerOrder"] / d["AOV"]

    ord_cols = ["InStoreOrders", "UberEatsOrders", "DoorDashOrders", "SelfDeliveryOrders"]
    shares = d[ord_cols].div(d[ord_cols].sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    hhi = (shares ** 2).sum(axis=1)
    d["ChannelBalance"] = (1 - hhi) / (1 - 0.25)
    return d


CLUSTER_FEATURES = [
    "GrowthFactor", "AOV", "MonthlyOrders", "Scale", "TotalRevenue", "TotalNetProfit",
    "NetMargin", "COGSRate", "OPEXRate", "CommissionRate", "CostRate",
    "DeliveryRadiusKM", "DeliveryCostPerOrder", "SD_DeliveryTotalCost",
    "InStoreShare", "UE_share", "DD_share", "SD_share",
    "AggregatorDependence", "AggregatorMargin", "InStoreMargin", "SD_Margin",
    "ChannelBalance", "ExpansionHeadroom", "RevenueQuality", "DeliveryCostPctAOV",
]

THEMES = {
    "Cost pressure": ["COGSRate", "OPEXRate", "CostRate", "NetMargin", "InStoreMargin", "SD_Margin",
                      "AggregatorMargin", "CommissionRate", "RevenueQuality"],
    "Channel leverage": ["InStoreShare", "UE_share", "DD_share", "SD_share", "AggregatorDependence",
                         "ChannelBalance"],
    "Growth momentum": ["GrowthFactor", "Scale", "MonthlyOrders", "TotalRevenue", "TotalNetProfit",
                        "AOV", "SD_DeliveryTotalCost"],
    "Logistics reach": ["DeliveryRadiusKM", "DeliveryCostPerOrder", "ExpansionHeadroom",
                        "DeliveryCostPctAOV"],
}


def preprocess(feat: pd.DataFrame, include_cat: bool = True, cat_weight: float = 0.35):
    num = feat[CLUSTER_FEATURES].astype(float)
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    Xn = pt.fit_transform(num)
    names = list(CLUSTER_FEATURES)
    blocks = [Xn]
    if include_cat:
        dummies = pd.get_dummies(feat[CATEGORICAL], dtype=float)
        dv = dummies.values
        dv = (dv - dv.mean(axis=0)) / (dv.std(axis=0) + 1e-9) * cat_weight
        blocks.append(dv)
        names += list(dummies.columns)
    return np.hstack(blocks), names


def fit_pca(X: np.ndarray, var_target: float = 0.85):
    pca = PCA(random_state=42).fit(X)
    Z = pca.transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_use = int(max(3, np.searchsorted(cum, var_target) + 1))
    return pca, Z, n_use


def name_components(pca: PCA, names: list[str], n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        load = pd.Series(pca.components_[i], index=names)
        theme_strength = {
            t: float((load.reindex(cols).dropna() ** 2).sum()) for t, cols in THEMES.items()
        }
        top = load.abs().sort_values(ascending=False).head(4).index
        rows.append({
            "Component": f"PC{i+1}",
            "Variance %": round(pca.explained_variance_ratio_[i] * 100, 1),
            "Dominant theme": max(theme_strength, key=theme_strength.get),
            "Top drivers": ", ".join(f"{f} ({load[f]:+.2f})" for f in top),
        })
    return pd.DataFrame(rows)


def project_2d(X: np.ndarray, Z: np.ndarray, method: str = "PCA", seed: int = 42) -> np.ndarray:
    if method == "PCA":
        return Z[:, :2]
    if method == "t-SNE":
        return TSNE(n_components=2, perplexity=35, init="pca", random_state=seed).fit_transform(X)
    if method == "UMAP":
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    raise ValueError(method)


def auto_eps(Z: np.ndarray, min_samples: int, pct: float = 90) -> float:
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)
    return float(np.percentile(d[:, -1], pct))


def fit_cluster(Z: np.ndarray, algo: str, k: int = 5, eps_mult: float = 1.0,
                min_samples: int = 12, seed: int = 42) -> np.ndarray:
    if algo == "K-Means":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z)
    if algo == "Hierarchical (Ward)":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    if algo == "DBSCAN":
        return DBSCAN(eps=auto_eps(Z, min_samples) * eps_mult, min_samples=min_samples).fit_predict(Z)
    raise ValueError(algo)


def k_scan(Z: np.ndarray, ks=range(2, 10), seed: int = 42) -> pd.DataFrame:
    out = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        out.append({"k": k, "Inertia": km.inertia_,
                    "Silhouette": silhouette_score(Z, km.labels_),
                    "Davies-Bouldin": davies_bouldin_score(Z, km.labels_)})
    return pd.DataFrame(out)


def cluster_quality(Z: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != -1
    n_cl = len(set(labels[mask]))
    if n_cl < 2 or mask.sum() < 10:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan, "Clusters": n_cl,
                "Noise %": round((~mask).mean() * 100, 1)}
    return {"Silhouette": silhouette_score(Z[mask], labels[mask]),
            "Davies-Bouldin": davies_bouldin_score(Z[mask], labels[mask]),
            "Clusters": n_cl, "Noise %": round((~mask).mean() * 100, 1)}


ARCHETYPES = [
    "High-Growth / High-Risk",
    "Stable Local Performers",
    "Aggregator-Dependent Low Margin",
    "Scalable Self-Delivery Leaders",
    "Overextended, Low Return",
]

ARCHETYPE_DESC = {
    "High-Growth / High-Risk": (
        "These restaurants show strong month-on-month order momentum (growth factor ≥ 1.04×) but thin or volatile "
        "economics. Margins are often compressed by aggressive promotions, high aggregator dependence, or "
        "under-optimised cost structures. **Why it matters:** growth is real, but it may be 'bought' through "
        "unsustainable pricing — every growth dollar must be stress-tested against unit economics. "
        "**Key risk:** if commissions rise or promotions end, profitability could reverse sharply. "
        "**Typical profile:** mid-to-high AOV, above-average order volume, below-peer net margin, "
        "elevated commission rates."
    ),
    "Stable Local Performers": (
        "Healthy margins, balanced channel economics, and moderate but steady growth characterise this archetype. "
        "These restaurants are reliable cash generators with diversified order sources — no single channel dominates "
        "beyond 40% of orders. **Why it matters:** stability makes them ideal candidates for cautious capacity "
        "expansion or franchise replication. **Strength:** low variance in returns across channels means less "
        "exposure to aggregator commission hikes. **Typical profile:** balanced order mix (Herfindahl < 0.3), "
        "above-median net margin, growth factor near the portfolio average, COGS+OPEX well-controlled."
    ),
    "Aggregator-Dependent Low Margin": (
        "Orders skew heavily to Uber Eats and DoorDash (aggregator dependence > 60%), where 25-33% commission "
        "rates erode profitability. Many restaurants in this group earn negative net margins on aggregator channels "
        "while remaining profitable overall only because of in-store or self-delivery revenue. "
        "**Why it matters:** structural dependence on a loss-making channel is a fragile foundation for growth. "
        "**The danger:** any volume growth through aggregators *amplifies* losses rather than building value. "
        "**Action path:** renegotiate commissions, raise platform-specific menu prices, shift repeat customers to "
        "self-delivery, or set minimum basket sizes to restore per-order profitability."
    ),
    "Scalable Self-Delivery Leaders": (
        "Profitable self-delivery operations with efficient logistics (low delivery cost per order relative to AOV), "
        "a strong self-delivery order share, and sufficient scale to absorb fixed delivery costs. "
        "**Why it matters:** owning the delivery channel means owning the customer relationship, the data, and the "
        "margin — the best structural position for geographic expansion. **Expansion signal:** high demand density "
        "(orders/km) suggests that extending the delivery radius could capture incremental revenue at minimal "
        "marginal cost. **Typical profile:** self-delivery margin > in-store margin, delivery cost < 8% of AOV, "
        "above-average radius, strong customer retention on the owned channel."
    ),
    "Overextended, Low Return": (
        "High cost bases (COGS + OPEX above 75th percentile), weak or negative net margins, and little organic "
        "growth define this archetype. These restaurants are spending more than they earn from each incremental "
        "order. **Why it matters:** expansion capital deployed here has the lowest expected ROI in the portfolio — "
        "any growth would compound structural losses. **Root cause:** typically a combination of elevated food "
        "costs (poor supplier terms or high waste) and excessive operating expenses. "
        "**Priority:** operational turnaround (cost reduction, menu engineering, labour optimisation) before any "
        "growth investment. Only after margins are stabilised should capacity or coverage be increased."
    ),
    "Outliers / Unclassified": (
        "Restaurants flagged as statistical outliers (DBSCAN noise points) or restaurants that don't fit neatly "
        "into any archetype. These are atypical operations that may have unique business models, niche cuisines, "
        "or data anomalies. **Action:** manual review is recommended — some may be hidden gems, others may have "
        "data quality issues that distort their profile. Check for missing data, unusual channel splits, or "
        "recently opened locations with insufficient operating history."
    ),
}


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def label_clusters(feat: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    z = pd.DataFrame({
        "growth": _z(feat["GrowthFactor"]), "margin": _z(feat["NetMargin"]),
        "aggdep": _z(feat["AggregatorDependence"]), "aggm": _z(feat["AggregatorMargin"]),
        "sdm": _z(feat["SD_Margin"]), "cost": _z(feat["CostRate"]),
        "scale": _z(feat["Scale"]), "sdshare": _z(feat["SD_share"]),
        "logi": _z(-feat["DeliveryCostPctAOV"]),
    })
    z["cluster"] = labels
    prof = z[z.cluster != -1].groupby("cluster").mean()

    def score(p: pd.Series) -> list[float]:
        return [
            p.growth - 0.5 * p.margin + 0.3 * p.cost,
            p.margin - 0.5 * p.aggdep - 0.4 * abs(p.growth) - 0.2 * p.cost,
            p.aggdep - p.aggm - 0.5 * p.margin,
            p.sdm + 0.5 * p.logi + 0.3 * p.scale - 0.3 * p.aggdep + 0.3 * p.sdshare,
            -p.margin + 0.6 * p.cost - 0.5 * p.growth - 0.2 * p.sdm,
        ]

    S = np.array([score(prof.loc[c]) for c in prof.index])
    mapping: dict[int, str] = {}
    rows, cols = linear_sum_assignment(-S)
    for r, c in zip(rows, cols):
        mapping[int(prof.index[r])] = ARCHETYPES[c]
    counts: dict[str, int] = {}
    for i, cid in enumerate(prof.index):
        if int(cid) not in mapping:
            best = ARCHETYPES[int(np.argmax(S[i]))]
            counts[best] = counts.get(best, 1) + 1
            mapping[int(cid)] = f"{best} ({['I','II','III','IV','V','VI'][counts[best]-1]})"
    if (labels == -1).any():
        mapping[-1] = "Outliers / Unclassified"
    return mapping


@dataclass
class ModelResult:
    feat: pd.DataFrame
    labels: np.ndarray
    label_map: dict
    pca: PCA
    Z: np.ndarray
    n_use: int
    names: list
    X: np.ndarray
    quality: dict
    comp_table: pd.DataFrame
    loadings: pd.DataFrame
    extra: dict = field(default_factory=dict)


def build_model(df: pd.DataFrame, algo="K-Means", k=5, include_cat=True, cat_weight=0.35,
                eps_mult=1.0, min_samples=12, seed=42) -> ModelResult:
    feat = engineer_features(df)
    X, names = preprocess(feat, include_cat, cat_weight)
    pca, Z, n_use = fit_pca(X)
    labels = fit_cluster(Z[:, :n_use], algo, k, eps_mult, min_samples, seed)
    label_map = label_clusters(feat, labels)
    feat = feat.copy()
    feat["ClusterID"] = labels
    feat["Cluster"] = [label_map[int(c)] for c in labels]
    loadings = pd.DataFrame(pca.components_[:6].T, index=names,
                            columns=[f"PC{i+1}" for i in range(6)])
    return ModelResult(
        feat=feat, labels=labels, label_map=label_map, pca=pca, Z=Z, n_use=n_use, names=names, X=X,
        quality=cluster_quality(Z[:, :n_use], labels),
        comp_table=name_components(pca, names), loadings=loadings,
    )
