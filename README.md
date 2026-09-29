# 🎰 SkyCity Auckland — Restaurant Growth Intelligence Platform

An enterprise-grade, evidence-backed decision engine and interactive analytics platform for evaluating, benchmarking, and optimizing dining assets across Auckland.

---

## 🌟 Key Features & Analytical Framework

### 1. Multi-Pillar Growth Potential Index (GPI)
Calculates a composite **0–100 GPI score** across 6 distinct strategic pillars with customizable weighting:
- **Performance (25%)**: Customer volume, revenue density, and ratings.
- **Sentiment (20%)**: Review sentiment balance, net promoter proxies, and rating-to-volume consistency.
- **SkyCity Fit (20%)**: Distance proximity to SkyCity Auckland, pricing synergy, and entertainment alignment.
- **Operational Scalability (15%)**: Capacity headroom, delivery integrations, and efficiency indicators.
- **Competitive Moat (10%)**: Submarket uniqueness, cuisine defensibility, and market share.
- **Digital Footprint (10%)**: Digital discoverability, booking channels, and social velocity.

### 2. Behavioral Clustering & Archetypes
Segments 1,696 Auckland venues into 5 validated strategic archetypes:
- **Scale Champions**: High-volume, highly scalable anchors ready for multi-unit expansion or marquee precinct placement.
- **Hidden Gems**: High sentiment, exceptional product-market fit constrained by physical footprint or location.
- **Turnaround Targets**: High foot-traffic assets hindered by service bottlenecks or operational leakage.
- **Steady Performers**: Reliable cash-flow generators with mature, loyal customer bases.
- **Underperformers**: Low-velocity, margin-pressured operators requiring restructuring or strategic pivot.

### 3. Statistical Validation & Reliability Framework (`reliability.py`)
- **Data Quality Audit**: Completeness score (89.2/100), automated distribution skew detection, and outlier audits.
- **Bootstrap 95% Confidence Intervals**: 1,000 bootstrap iterations per venue quantifying rank and score uncertainty.
- **Hypothesis Testing**: Kruskal-Wallis non-parametric ANOVA ($H = 887.46, p < 10^{-15}$) confirming archetype divergence.
- **Sensitivity & Weight Elasticity**: Real-time rank drift analysis against parameter shifts.
- **Automated Evidence-Backed Insights**: Machine-generated strategic action items tagged with confidence levels and quantitative impact estimates.

---

## 🚀 Quickstart: Running Locally

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)
- `pip` package manager

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🌐 Production Deployment Options

### Option A: Streamlit Community Cloud (Recommended — Free & Instant)
1. Initialize Git and commit the project:
   ```bash
   git init
   git add .
   git commit -m "feat: complete SkyCity intelligence platform with reliability engine"
   ```
2. Push the repository to GitHub:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main
   ```
3. Visit [share.streamlit.io](https://share.streamlit.io), connect your GitHub account, and select:
   - **Repository**: `<your-username>/<your-repo-name>`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**

### Option B: Docker Container (Any Cloud / Server)
Build and run the container locally or deploy to AWS App Runner, Google Cloud Run, Azure Container Apps, or DigitalOcean:
```bash
# Build Docker image
docker build -t skycity-growth-intelligence .

# Run container on port 8501
docker run -p 8501:8501 skycity-growth-intelligence
```

### Option C: Render / Railway / Heroku
The included `Procfile` allows zero-configuration deployment:
- Connect your GitHub repository to [Render](https://render.com) or [Railway](https://railway.app).
- Select **Web Service**.
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### Option D: Hugging Face Spaces
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** SDK.
3. Push or upload all repository files directly.

---

## 📂 Project Structure

```text
├── app.py               # Main 9-tab interactive Streamlit dashboard
├── reliability.py       # Reliability, bootstrap CI, data quality, & statistical tests
├── pipeline.py          # Feature engineering, scaling, clustering, & PCA/UMAP
├── scoring.py           # Multi-pillar GPI scoring engine & evidence-backed recommendations
├── charts.py            # High-performance Plotly visualizations & statistical plots
├── report.py            # Executive report generator with statistical appendices
├── requirements.txt     # Python dependency specifications
├── Dockerfile           # Production container configuration
├── Procfile             # PaaS deployment entrypoint
├── .streamlit/
│   └── config.toml      # Production server, security, and theme settings
└── data/
    └── skycity_restaurants.csv  # 1,696 Auckland restaurant records
```
