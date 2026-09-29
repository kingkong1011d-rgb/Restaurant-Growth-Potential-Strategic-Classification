# Strategic Archetyping and Multi-Pillar Growth Potential Modeling for Hospitality Assets: A Machine Learning and Statistical Reliability Framework

**Author:** Antigravity Research Group / SkyCity Intelligence Initiative  
**Repository:** [kingkong1011d-rgb/Restaurant-Growth-Potential-Strategic-Classification](https://github.com/kingkong1011d-rgb/Restaurant-Growth-Potential-Strategic-Classification)  
**Interactive Application:** [https://restgrowth.streamlit.app/](https://restgrowth.streamlit.app/)  
**Keywords:** Machine Learning, K-Means Clustering, Principal Component Analysis, Growth Potential Index, Multi-Channel Unit Economics, Statistical Reliability, Kruskal-Wallis Test, Hospitality Analytics

---

## Abstract

The contemporary food service and restaurant sector operates within a multi-channel paradigm characterized by complex interactions among in-store dining, third-party delivery aggregator platforms (e.g., Uber Eats, DoorDash), and proprietary self-delivery logistics. Conventional evaluation approaches rely predominantly on top-line sales volume or simplistic customer review ratings, obscuring profound variations in commission friction, cost pressures, and true unit economics. In this paper, we introduce an end-to-end analytical and machine learning framework designed to evaluate, benchmark, and classify hospitality assets systematically. Utilizing an empirical dataset of $N = 1,696$ restaurant businesses across Auckland, New Zealand, our methodology integrates Yeo-Johnson power transformations, Principal Component Analysis (PCA), and $k$-means clustering to identify five distinct, statistically validated operational archetypes: *High-Growth / High-Risk* ($n = 437$), *Stable Local Performers* ($n = 426$), *Scalable Self-Delivery Leaders* ($n = 379$), *Aggregator-Dependent Low Margin* ($n = 255$), and *Overextended, Low Return* ($n = 199$). 

Furthermore, we construct a composite, multi-pillar **Growth Potential Index (GPI)** spanning four core operational dimensions: Growth Signals (30%), Cost Resilience (30%), Channel Balance (20%), and Logistics Scalability (20%). Archetype separation is validated via non-parametric Kruskal-Wallis ANOVA ($H = 887.46, p < 10^{-15}$) alongside pairwise Mann-Whitney $U$ tests. Uncertainty quantification is established via 1,000-iteration bootstrap percentile confidence intervals and continuous boundary distance estimation. Finally, we implement an interactive decision support engine deployed to production to generate granular, evidence-backed strategic actions—enabling capital allocators and enterprise operators to distinguish between scalable operations and structurally compromised margin structures.

---

## 1. Introduction

The economic architecture of the hospitality industry has undergone significant disruption over the past decade. The rapid rise of on-demand third-party delivery aggregators has expanded the geographic footprint of dining venues, yet it has simultaneously introduced high platform commission fees (typically ranging between 20% and 35% of gross order value). Consequently, high-volume dining establishments frequently encounter the "growth paradox": expanding top-line revenue accompanied by eroding operating margins or outright channel-level net losses.

Despite this operational reality, traditional commercial underwriting, franchising assessment, and precinct tenancy selection have continued to rely on blunt proxy metrics—chiefly gross sales velocity, average order value (AOV), or aggregate customer sentiment. Such approaches fail to capture critical operational nuances:
1. **Channel Cannibalization & Margin Dilution:** Higher order volumes channeled through high-commission third-party apps can depress in-store foot traffic and drive negative marginal unit profit.
2. **Logistics Asymmetry:** Proprietary self-delivery networks, while demanding upfront fleet coordination, often retain significantly superior unit margins compared to third-party marketplaces.
3. **Cost Inelasticity:** Variations in food cost of goods sold (COGS) and operational expenditures (OPEX) determine whether a business can withstand local economic fluctuations and wage inflation.

To address these challenges, this study presents a rigorous computational framework for **strategic classification and growth potential estimation**. We formulate an automated feature engineering pipeline, an unsupervised clustering architecture grounded in dimensionality reduction, a composite scoring index (GPI), and a non-parametric statistical reliability framework. The overall system is packaged as an interactive decision intelligence engine, made accessible to researchers, hospitality asset managers, and policymakers via an open-source web application.

---

## 2. Dataset & Data Quality Audit

### 2.1 Empirical Cohort Overview
The study analyzes an empirical dataset consisting of $N = 1,696$ commercial food service venues operating within the Greater Auckland metropolitan area. The cohort represents diverse culinary styles (30+ distinct cuisine types), geographic subregions (including Auckland CBD, Ponsonby, Newmarket, Takapuna, and Manukau), and pricing tiers (ranging from fast-casual to fine dining).

Each record contains 30 raw operational, logistical, and financial variables, including:
- **Order Volumes:** In-store, Uber Eats, DoorDash, and self-delivery monthly order counts ($O_{\text{in}}, O_{\text{ue}}, O_{\text{dd}}, O_{\text{sd}}$).
- **Revenue Distributions:** Channel-specific gross earnings ($R_{\text{in}}, R_{\text{ue}}, R_{\text{dd}}, R_{\text{sd}}$) and Average Order Value ($\text{AOV}$).
- **Cost & Margin Drivers:** Cost of Goods Sold rate ($\text{COGS}$), Operating Expense rate ($\text{OPEX}$), Aggregator Commission rate ($\text{Commission}$), and Delivery Radius ($\text{Radius}_{\text{KM}}$).
- **Channel Net Profits:** Net operating returns segmented across all four fulfillment modes ($\Pi_{\text{in}}, \Pi_{\text{ue}}, \Pi_{\text{dd}}, \Pi_{\text{sd}}$).

### 2.2 Data Quality Scoring Protocol
Prior to downstream modeling, we execute an automated four-stage data quality audit:
1. **Completeness Verification:** Evaluates missing value frequencies across all numeric and categorical attributes:
   $$\text{Completeness} = 1 - \frac{1}{N \cdot P} \sum_{i=1}^{N} \sum_{j=1}^{P} \mathbb{I}(x_{ij} \text{ is null}) = 100.0\%$$
2. **Interquartile Range (IQR) Outlier Scanning:** Flags extreme aberrations using Tukey's fences ($[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$).
3. **Distributional Skewness:** Quantifies Fisher-Pearson standardized moment coefficients:
   $$\gamma_1 = \frac{\mathbb{E}[(X - \mu)^3]}{\sigma^3}$$
   Features exhibiting $|\gamma_1| > 2.0$ are flagged for stabilizing non-linear power transformations.
4. **Duplicate & Logical Consistency Checks:** Enforces uniqueness of entity identifiers and validates non-negativity across revenue streams.

The composite **Data Quality Score** is formulated as:
$$\text{Score}_{\text{DQ}} = \frac{1}{4} \left( S_{\text{comp}} + S_{\text{outlier}} + S_{\text{skew}} + S_{\text{integrity}} \right) = 89.2 / 100$$
confirming a robust foundation for statistical modeling.

---

## 3. Methodology & Mathematical Formulation

The methodological architecture comprises four sequential computational stages: (1) feature synthesis, (2) Yeo-Johnson preprocessing and Principal Component Analysis, (3) unsupervised archetype clustering with Hungarian bipartite matching, and (4) multi-pillar Growth Potential Index scoring.

```
┌─────────────────────────┐
│ Raw Restaurant Records  │ (N = 1,696 records, 30 attributes)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Feature Synthesis       │ (26 multidimensional engineered metrics)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Yeo-Johnson + Scaling   │ (PowerTransformer, categorical dummy weighting)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Principal Component     │ (Explained variance threshold ≥ 85%)
│ Analysis (PCA)          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ K-Means Clustering      │ (k = 5, n_init = 20)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Hungarian Optimal Match │ (Assignment to 5 operational archetypes)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Multi-Pillar GPI Engine │ (Growth, Cost, Channel, Logistics)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Statistical Reliability │ (Bootstrap CIs, Kruskal-Wallis ANOVA, U tests)
└─────────────────────────┘
```

### 3.1 Feature Synthesis
We synthesize 26 domain-specific features designed to expose hidden unit economics:
- **Total Revenue & Net Margin:**
  $$R_{\text{total}} = \sum_{c \in \mathcal{C}} R_c, \quad \Pi_{\text{total}} = \sum_{c \in \mathcal{C}} \Pi_c, \quad \text{Margin}_{\text{net}} = \frac{\Pi_{\text{total}}}{R_{\text{total}}}$$
- **Aggregator Dependence & Channel Economics:**
  $$\text{AggDep} = \frac{O_{\text{ue}} + O_{\text{dd}}}{O_{\text{monthly}}}, \quad \text{Margin}_{\text{agg}} = \frac{\Pi_{\text{ue}} + \Pi_{\text{dd}}}{R_{\text{ue}} + R_{\text{dd}}}$$
- **Channel Dispersion (Herfindahl-Hirschman Index Derivative):**
  $$s_c = \frac{O_c}{\sum_{k} O_k}, \quad \text{HHI} = \sum_{c} s_c^2, \quad \text{Balance}_{\text{channel}} = \frac{1 - \text{HHI}}{1 - 0.25}$$
  where $\text{Balance}_{\text{channel}} \in [0, 1]$ achieves its maximum under perfectly uniform order distribution across all four channels.
- **Revenue Quality & Logistics Headroom:**
  $$\text{Quality}_{\text{rev}} = \text{AOV} \times \text{Margin}_{\text{net}} \times \left( 1 - \min(1, \sigma_{\text{margins}}) \right)$$
  $$\text{Headroom}_{\text{expansion}} = \frac{O_{\text{monthly}}}{\text{Radius}_{\text{KM}}}$$

### 3.2 Yeo-Johnson Power Transformation & PCA
To rectify severe skewness in financial and volume metrics without discarding zeros or negative net margins, each continuous variable is transformed via the Yeo-Johnson parameterized power family $\psi(\lambda, x)$:
$$\psi(\lambda, x) = \begin{cases}
\frac{(x + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, x \ge 0 \\
\log(x + 1) & \text{if } \lambda = 0, x \ge 0 \\
-\frac{(-x + 1)^{2 - \lambda} - 1}{2 - \lambda} & \text{if } \lambda \neq 2, x < 0 \\
-\log(-x + 1) & \text{if } \lambda = 2, x < 0
\end{cases}$$
followed by zero-mean, unit-variance standardization:
$$\mathbf{z}_j = \frac{\psi(\hat{\lambda}_j, \mathbf{x}_j) - \mu_j}{\sigma_j}$$

Categorical attributes (Cuisine, Segment, Subregion) are one-hot encoded, standardized, and scaled by a shrinkage factor $\omega_{\text{cat}} = 0.35$ to prevent high-cardinality nominal features from dominating numerical variance. Principal Component Analysis is then applied:
$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
Selecting $d$ components such that the cumulative explained variance satisfies $\sum_{i=1}^{d} \lambda_i / \sum_{j} \lambda_j \ge 0.85$.

### 3.3 Clustering & Bipartite Archetype Matching
Partitioning is conducted via $k$-means clustering on the reduced eigenspace $\mathbf{Z}_d$, optimizing the within-cluster sum of squares (WCSS):
$$\arg\min_{\mathcal{S}} \sum_{k=1}^{K} \sum_{\mathbf{z} \in S_k} \|\mathbf{z} - \boldsymbol{\mu}_k\|^2_2$$

To ensure repeatable, explainable business archetypes across varying model initializations, cluster centroids are matched to five pre-defined theoretical archetypes using the Kuhn-Munkres (Hungarian) algorithm on an operational affinity matrix $\mathbf{M} \in \mathbb{R}^{K \times 5}$:
$$\max_{\pi} \sum_{i=1}^{K} \mathbf{M}_{i, \pi(i)}$$
where affinity scores balance growth factors, net margins, aggregator dependence, cost discipline, and self-delivery logistics.

### 3.4 Multi-Pillar Growth Potential Index (GPI)
The **Growth Potential Index** is formulated as a percentile-ranked, weighted multi-attribute utility function:
$$\text{GPI}_i = \sum_{p=1}^{4} w_p \cdot \text{Pillar}_{p}(i)$$
subject to $\sum_{p=1}^4 w_p = 100$, where default weight distributions are parameterized as:
1. **Growth Signals ($w_1 = 30\%$):**
   $$\text{Pillar}_1 = 0.5 \cdot \text{Rank}(\text{GrowthFactor}) + 0.3 \cdot \text{Rank}(\text{Scale}) + 0.2 \cdot \text{Rank}(\text{AOV})$$
2. **Cost Resilience ($w_2 = 30\%$):**
   $$\text{Pillar}_2 = 0.5 \cdot \left(100 - \text{Rank}(\text{CostRate})\right) + 0.5 \cdot \text{Rank}(\text{NetMargin})$$
3. **Channel Balance ($w_3 = 20\%$):**
   $$\text{Pillar}_3 = 0.5 \cdot \text{Rank}(\text{Balance}_{\text{channel}}) + 0.3 \cdot \left(100 - \text{Rank}(\text{AggDep})\right) + 0.2 \cdot \text{Rank}(\text{Margin}_{\text{agg}})$$
4. **Logistics Scalability ($w_4 = 20\%$):**
   $$\text{Pillar}_4 = 0.35 \cdot \text{Rank}(\text{Margin}_{\text{sd}}) + 0.25 \cdot \left(100 - \text{Rank}(\text{DeliveryCostPctAOV})\right) + 0.2 \cdot \text{Rank}(\text{Radius}) + 0.2 \cdot \text{Rank}(\text{Headroom})$$

### 3.5 Strategic Tri-Classification Rule
Each restaurant is mapped into an actionable strategic tier:
$$\text{Strategy}(i) = \begin{cases}
\text{Hold / Stabilize} & \text{if } \text{GPI}_i < \mathcal{Q}_{0.35}(\text{GPI}) \\
\text{Rebalance channels} & \text{if } \text{GPI}_i \ge \mathcal{Q}_{0.35}(\text{GPI}) \land \text{Margin}_{\text{agg}}(i) < 0.0 \\
\text{Optimize} & \text{otherwise}
\end{cases}$$

---

## 4. Empirical Results & Findings

### 4.1 Portfolio Overview & Macro Metrics
Evaluating the entire $N = 1,696$ Auckland restaurant cohort reveals substantial aggregate economic activity alongside structural vulnerabilities:
- **Total Monthly Portfolio Revenue:** $\$77,739,307$ NZD
- **Total Monthly Net Operating Profit:** $\$7,866,306$ NZD
- **Aggregate Portfolio Margin:** $10.12\%$
- **Composite GPI Mean $\pm$ Standard Deviation:** $50.03 \pm 13.22$ (Range: $14.12 - 86.45$)

```
+---------------------------------------------------------------------------------+
| Strategy Tier          | Count (n) | Portfolio Share | Mean GPI | Primary Lever |
+========================+===========+=================+==========+===============+
| Optimize               | 954       | 56.25%          | 57.8     | Scale Capital |
| Hold / Stabilize       | 594       | 35.02%          | 37.1     | Cost Control  |
| Rebalance Channels     | 148       | 8.73%           | 51.9     | Channel Shift |
+---------------------------------------------------------------------------------+
```

### 4.2 Archetype Distribution & Profiles
Clustering analysis reveals five clearly differentiated operating realities:

```
+---------------------------------+-------+----------+------------+------------+-----------+
| Archetype                       | n     | Mean GPI | Net Margin | Agg. Share | SD Margin |
+=================================+=======+==========+============+============+===========+
| High-Growth / High-Risk         | 437   | 53.4     | 8.4%       | 58.2%      | 14.1%     |
| Stable Local Performers         | 426   | 52.8     | 14.6%      | 34.1%      | 18.2%     |
| Scalable Self-Delivery Leaders  | 379   | 58.9     | 13.9%      | 29.5%      | 23.4%     |
| Aggregator-Dependent Low Margin | 255   | 41.2     | 4.1%       | 68.7%      | 9.8%      |
| Overextended, Low Return        | 199   | 31.6     | -2.8%      | 51.4%      | 3.2%      |
+---------------------------------+-------+----------+------------+------------+-----------+
```

1. **High-Growth / High-Risk ($n = 437, 25.8\%$):** Marked by strong month-on-month order acceleration ($\text{GrowthFactor} \ge 1.04$), yet compromised by volatile marketing spend and elevated aggregator commission leakage.
2. **Stable Local Performers ($n = 426, 25.1\%$):** Balanced order profiles with well-controlled food and operational costs, exhibiting consistent cash generation and low reliance on discounts.
3. **Scalable Self-Delivery Leaders ($n = 379, 22.3\%$):** High-margin operators demonstrating superior delivery logistics efficiency, retaining full ownership of customer relationships.
4. **Aggregator-Dependent Low Margin ($n = 255, 15.0\%$):** Businesses with order profiles heavily dominated by Uber Eats and DoorDash, suffering severe margin compression due to 25–33% commission rates.
5. **Overextended, Low Return ($n = 199, 11.7\%$):** Financially strained entities where combined COGS and OPEX exceed 75% of revenues, yielding negative net margins.

### 4.3 The Aggregator Profitability Paradox
A critical finding of our empirical investigation is the widespread incidence of channel-level unprofitability. Across the entire portfolio:
- **$31.8\%$ of venues** operate at a net loss on third-party aggregator orders when accounting for commissions, dedicated packaging, and operational overhead.
- In contrast, proprietary self-delivery generated positive net margins for **$91.4\%$ of venues**, maintaining an average margin premium of **$+8.6$ percentage points** over aggregators.
- For venues in the *Rebalance Channels* tier, migrating merely $20\%$ of aggregator orders to self-delivery or direct collection produces an average monthly margin recovery of **$\$1,840 - \$3,420$ NZD per venue**.

---

## 5. Statistical Reliability & Validation

To establish methodological rigor, we deployed three complementary statistical verification tests:

### 5.1 Non-Parametric ANOVA (Kruskal-Wallis Test)
Because composite GPI scores across clusters depart moderately from strict normality, we performed the non-parametric Kruskal-Wallis test across the five archetype groups:
$$H = (N - 1) \frac{\sum_{i=1}^{k} n_i (\bar{r}_{i\cdot} - \bar{r})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (r_{ij} - \bar{r})^2}$$
The test yielded:
$$H = 887.46, \quad p = 2.14 \times 10^{-190} \quad (p < 10^{-15})$$
confirming with overwhelming statistical significance that the identified archetypes represent fundamentally distinct operational distributions rather than stochastic artifacts. Subsequent pairwise Mann-Whitney $U$ post-hoc tests with Bonferroni correction confirmed significant divergence across all pairs ($p_{\text{adj}} < 10^{-4}$).

### 5.2 Bootstrap Confidence Intervals & Rank Stability
To assess the sensitivity of individual restaurant GPI scores to sampling variation, we executed non-parametric bootstrapping with $B = 1,000$ iterations:
$$\text{CI}_{0.95} = \left[ \widehat{\text{GPI}}_{(0.025)}^{*}, \widehat{\text{GPI}}_{(0.975)}^{*} \right]$$
The mean $95\%$ confidence interval width across all $1,696$ venues was $\pm 3.12$ index points, indicating high stability. Individual confidence scores derived from interval widths demonstrated that over $88\%$ of classifications possess confidence metrics exceeding $80.0\%$.

### 5.3 Boundary Proximity & Classification Confidence
For strategic tri-classification, boundary proximity was evaluated by measuring minimum Euclidean distance to the hold threshold ($\mathcal{Q}_{0.35}$) and the aggregator margin zero-floor:
$$\text{Confidence}_{\text{class}}(i) = \min \left( \frac{|\text{GPI}_i - \text{Threshold}_{\text{hold}}|}{\delta_{\text{GPI}}}, \frac{|\text{Margin}_{\text{agg}}(i) - 0.0|}{\delta_{\text{agg}}} \right) \times 100$$
Restaurants near classification thresholds are automatically flagged in the interactive dashboard with targeted warning badges to prevent premature capital allocation decisions.

---

## 6. Software Architecture & Interactive Implementation

The computational pipeline is fully operationalized as a production-grade web application built on Streamlit, Plotly, and Scikit-Learn:
- **Real-Time Scenario Modeling:** An interactive what-if simulator allows decision-makers to model order migration from aggregators to self-delivery, calculating projected margin recovery dynamically.
- **Pillar Sensitivity Sweeps:** Users can alter pillar weightings in real-time, observing rank drift and Spearman rank correlation ($\rho > 0.92$ under $\pm 20\%$ weight shifts).
- **Automated Evidence-Backed Action Cards:** For any selected venue, the platform synthesizes personalized operational recommendations, contrasting entity metrics against peer archetype medians.

The complete codebase, configurations, and data artifacts are publicly accessible via the open repository:
- **GitHub Repository:** [https://github.com/kingkong1011d-rgb/Restaurant-Growth-Potential-Strategic-Classification](https://github.com/kingkong1011d-rgb/Restaurant-Growth-Potential-Strategic-Classification)
- **Production Dashboard:** [https://restgrowth.streamlit.app/](https://restgrowth.streamlit.app/)

---

## 7. Discussion & Strategic Implications

The findings carry direct implications for hospitality management and commercial real estate:
1. **Capital Allocation Selectivity:** Only $56.3\%$ of the portfolio (*Optimize*) possesses the financial resilience to absorb expansion capital effectively. Deploying capital to *Overextended* or *Aggregator-Dependent* venues without prior restructuring amplifies losses.
2. **Channel Sovereignty:** The reliance on third-party aggregators represents an acute vulnerability. Establishments must cultivate proprietary ordering channels and leverage menu-level price differentials to offset commission costs.
3. **Data-Driven Tenancy Selection:** Precinct operators (such as SkyCity Auckland) can employ the GPI framework to pre-screen prospective tenants, matching venue capabilities with appropriate precinct customer volume and pricing expectations.

---

## 8. Conclusion

This paper presented an integrated, statistically validated machine learning methodology for evaluating restaurant growth potential and classifying operational archetypes. By pairing Yeo-Johnson transformations, PCA, and $k$-means clustering with a multi-pillar scoring engine and rigorous non-parametric hypothesis testing ($H = 887.46, p < 10^{-15}$), our framework bridges the gap between academic analytics and practical hospitality decision-making. The open-source dashboard provides an intuitive, transparent interface for capital allocators, precinct managers, and restaurateurs navigating the multi-channel hospitality landscape.

---

## References

1. Anderson, C. K., & Xie, X. (2010). Improving restaurant revenue management through customer relationship management. *International Journal of Hospitality Management*, 29(3), 400-408.
2. Yeo, I. K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. *Biometrika*, 87(4), 954-959.
3. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281-297.
4. Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583-621.
5. Efron, B., & Tibshirani, R. J. (1994). *An introduction to the bootstrap*. CRC press.
6. Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1‐2), 83-97.
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
