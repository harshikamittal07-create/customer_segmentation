
## PROJECT NAME- Customer Segmentation using RFM Analysis & Clustering

Created for- 6 week ML and GenAI with Python Internship

Team Members:

1.Harshika Mittal-07301172025, IGDTUW

2.Janvi Upreti- 07801172025, IGDTUW

3.Dhruvi Relan-05101172025, IGDTUW

Link to Dataset used: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci


**Live App:** [customersegmentation-forjsvg6fqqmit74k46lrk.streamlit.app](https://customersegmentation-forjsvg6fqqmit74k46lrk.streamlit.app/)

A machine learning project that segments e-commerce customers into actionable groups using RFM (Recency, Frequency, Monetary) analysis and unsupervised clustering, deployed as an interactive Streamlit dashboard.

---

## 📌 Problem Statement

Retail businesses often treat all customers the same way, applying blanket marketing strategies regardless of actual purchasing behavior. This leads to wasted marketing spend, lower customer retention, and missed opportunities to reward high-value customers. This project identifies meaningful customer segments from transaction data so that marketing efforts can be targeted, personalized, and more effective.

## ✨ What Makes This Project Different

Rather than applying a single clustering algorithm (the common approach in beginner projects), this project:

- Compares **four clustering algorithms** side by side: K-Means, Agglomerative Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models (GMM)
- Evaluates each using **three independent validation metrics**: Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score
- Determines the optimal number of clusters using the **Elbow Method** and **Silhouette Score**, with an explicit, justified override where the statistically "best" split (k=2) was rejected as business-uninformative in favor of k=4
- Visualizes cluster separation using **PCA** dimensionality reduction
- Translates raw clusters into **business-readable customer personas** with tailored marketing recommendations
- Ships as a **deployed, interactive Streamlit app** — not just a notebook

---

## 📊 Dataset

**Source:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI Machine Learning Repository)

- Real transaction data from a UK-based online retailer, Dec 2009 – Dec 2011
- ~1,067,000 raw transaction records
- After cleaning: **779,425 transactions** across **5,878 unique customers**

Cleaning steps: removed missing Customer IDs, cancelled orders, and non-positive quantity/price values.

---

## 🧠 Methodology

### 1. Feature Engineering (RFM)
Each customer's ~1M rows of transactions were collapsed into three behavioral features:
- **Recency** — days since their last purchase
- **Frequency** — number of distinct orders placed
- **Monetary** — total amount spent

Extreme outliers were capped at the 99th percentile, and all features were standardized (z-score) before clustering.

### 2. Choosing the Number of Clusters
The Elbow Method and Silhouette Score were computed for k = 2 to 10. Although Silhouette peaked at k=2, this split was rejected as too coarse to be marketing-actionable. **k=4** was selected based on the Elbow Method's inflection point and the goal of producing distinct, usable segments.

### 3. Clustering & Evaluation
All four algorithms were run on the same standardized RFM features and compared:

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Clusters Found |
|---|---|---|---|---|
| **K-Means** | **0.525** | 0.717 | **7360.8** | 4 |
| Hierarchical | 0.468 | 0.790 | 6286.4 | 4 |
| DBSCAN | 0.484 | 0.481 | 545.5 | 5 (incl. noise) |
| GMM | 0.346 | 1.683 | 2171.8 | 4 |

**K-Means was selected** as the best-performing algorithm — it won on 2 of 3 metrics decisively. DBSCAN's misleadingly strong Davies-Bouldin score was traced back to one dominant cluster (91% of customers) plus several trivially "tight" micro-clusters — a useful cautionary finding rather than a competitive result.

### 4. Visualization
PCA reduced the 3D RFM feature space to 2D for visual confirmation of K-Means's clean, well-separated cluster structure.

---

## 🎯 Customer Segments

| Segment | Recency | Frequency | Monetary | % of Customers |
|---|---|---|---|---|
| 🏆 **Champions** | 37.6 days | 34.4 orders | £20,500 | 3.3% |
| 💎 **Loyal Customers** | 44.5 days | 14.9 orders | £5,854 | 13.7% |
| 🛒 **Regular/Occasional Buyers** | 84.0 days | 3.8 orders | £1,191 | 51.2% |
| ⚠️ **At-Risk / Lost Customers** | 475.2 days | 2.0 orders | £653 | 31.8% |

**Key finding:** Champions make up just **3.3% of customers** but drive **29.7% of total revenue** — a strong Pareto-style insight that underlines the value of targeted retention for high-value customers.

### Recommended Actions
- **Champions:** VIP loyalty perks, early access to new arrivals, dedicated account support
- **Loyal Customers:** Cross-sell/upsell campaigns, loyalty-tier incentives to push toward Champion status
- **Regular/Occasional Buyers:** Re-engagement emails, seasonal promotions, bundle offers
- **At-Risk/Lost Customers:** Win-back campaigns with strong discounts, or deprioritize if reactivation cost exceeds historical value

---

## 🖥️ App Features

The deployed Streamlit app has two core capabilities:

1. **Single Customer Predictor** — enter (or randomly sample) a customer's RFM values and instantly see their predicted segment, a tailored marketing recommendation, and how they compare to each segment's average
2. **Interactive Dashboard** — live business metrics, segment profile tables, scatter plots, distribution charts, and a segment-size breakdown

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data handling:** Pandas, NumPy
- **Machine learning:** scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture, PCA, StandardScaler)
- **Visualization:** Plotly
- **App / deployment:** Streamlit, Streamlit Community Cloud
- **Model persistence:** joblib

---

## 📁 Project Structure

```
segmentation-app/
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── scaler.pkl                 # Fitted StandardScaler
├── kmeans_model.pkl           # Trained K-Means model (k=4)
├── cluster_labels.pkl         # Cluster number → segment name mapping
└── rfm_with_clusters.csv      # Customer-level RFM data with assigned clusters
```

The model itself was trained in a Google Colab notebook (data cleaning → RFM engineering → scaling → clustering → evaluation → export), with the trained artifacts exported via `joblib` for use in the Streamlit app.

---

## 🚀 Running Locally

```bash
git clone <your-repo-url>
cd segmentation-app
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---
