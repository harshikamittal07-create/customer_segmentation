import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Segmentation Explorer",
    page_icon="🛍️",
    layout="wide",
)

# ---------------------------------------------------------
# Load saved artifacts (must sit in the same folder as this file)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    cluster_labels = joblib.load("cluster_labels.pkl")
    return scaler, kmeans, cluster_labels

@st.cache_data
def load_data():
    return pd.read_csv("rfm_with_clusters.csv")

try:
    scaler, kmeans, cluster_labels = load_artifacts()
    rfm = load_data()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e}. Make sure scaler.pkl, kmeans_model.pkl, "
        "cluster_labels.pkl and rfm_with_clusters.csv sit in the same "
        "folder as app.py."
    )
    st.stop()

rfm["Segment"] = rfm["Cluster"].map(cluster_labels)

# ---------------------------------------------------------
# Segment styling — mirrors the species-info-card pattern
# ---------------------------------------------------------
SEGMENT_INFO = {
    "Champions": {
        "emoji": "🏆",
        "color": "#2ecc71",
        "desc": "Recent, frequent, high-spending customers. Your smallest group but your biggest revenue driver — protect this segment above all others.",
        "action": "VIP loyalty perks, early access to new arrivals, dedicated account support.",
    },
    "Loyal Customers": {
        "emoji": "💎",
        "color": "#3498db",
        "desc": "Recent and reasonably frequent buyers with solid spend. Strong candidates to upgrade into Champions.",
        "action": "Cross-sell/upsell campaigns, loyalty-tier incentives.",
    },
    "Regular/Occasional Buyers": {
        "emoji": "🛒",
        "color": "#f39c12",
        "desc": "Your largest segment — moderate recency, low purchase frequency. The biggest volume-based growth opportunity.",
        "action": "Re-engagement emails, seasonal promotions, bundle offers.",
    },
    "At-Risk / Lost Customers": {
        "emoji": "⚠️",
        "color": "#e74c3c",
        "desc": "Haven't purchased in a long time and historically spent the least. Needs a decision: win back or deprioritize.",
        "action": "Win-back discounts, \"we miss you\" emails, or deprioritize if reactivation cost outweighs value.",
    },
}
DEFAULT_INFO = {"emoji": "❓", "color": "#95a5a6", "desc": "", "action": ""}

cluster_profile = (
    rfm.groupby(["Cluster", "Segment"])
    .agg(
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
        Count=("CustomerID", "count"),
    )
    .reset_index()
    .sort_values("Monetary_mean", ascending=False)
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #2ecc71, #3498db, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🛍️ Customer Segmentation Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">RFM-based clustering to turn transactions into actionable customer segments</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar — inputs + model info
# ---------------------------------------------------------
st.sidebar.header("🔧 Customer RFM Values")
st.sidebar.caption("Adjust to match a customer, or pick a real one below")

r_min, r_max = float(rfm.Recency.min()), float(rfm.Recency.max())
f_min, f_max = float(rfm.Frequency.min()), float(rfm.Frequency.max())
m_min, m_max = float(rfm.Monetary.min()), float(rfm.Monetary.max())

recency = st.sidebar.slider("Recency (days since last purchase)", r_min, r_max, 30.0, 1.0)
frequency = st.sidebar.slider("Frequency (number of orders)", f_min, f_max, 5.0, 1.0)
monetary = st.sidebar.slider("Monetary (total spend, £)", m_min, m_max, 500.0, 10.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Or try a real customer")
if st.sidebar.button("Pick random customer from dataset"):
    sample = rfm.sample(1).iloc[0]
    recency, frequency, monetary = sample.Recency, sample.Frequency, sample.Monetary
    st.sidebar.success(f"Loaded a real customer — actual segment: {sample.Segment}")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.write(f"**Algorithm:** K-Means (k={kmeans.n_clusters})")
st.sidebar.write(f"**Customers in training set:** {len(rfm):,}")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
input_data = np.array([[recency, frequency, monetary]])
input_scaled = scaler.transform(input_data)
pred_cluster = int(kmeans.predict(input_scaled)[0])
pred_segment = cluster_labels.get(pred_cluster, f"Cluster {pred_cluster}")
info = SEGMENT_INFO.get(pred_segment, DEFAULT_INFO)

# distance to each cluster centroid -> pseudo "confidence" view
centroids = kmeans.cluster_centers_
dists = np.linalg.norm(centroids - input_scaled, axis=1)
closeness = 1 / (1 + dists)
closeness = closeness / closeness.sum()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Predicted Segment")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {info['color']}22, {info['color']}11);
                border: 2px solid {info['color']};
                border-radius: 16px; padding: 1.5rem; text-align: center;">
        <div style="font-size: 3rem;">{info['emoji']}</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: {info['color']};">{pred_segment}</div>
        <div style="color: #666; margin-top: 0.5rem;">{info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📣 Recommended Action")
    st.info(info["action"])

    st.markdown("### 📊 Closeness to Each Segment")
    close_df = pd.DataFrame({
        "Segment": [cluster_labels.get(i, f"Cluster {i}") for i in range(len(centroids))],
        "Closeness": closeness,
    }).sort_values("Closeness")

    fig_close = go.Figure(go.Bar(
        x=close_df["Closeness"], y=close_df["Segment"], orientation="h",
        marker_color=[SEGMENT_INFO.get(s, DEFAULT_INFO)["color"] for s in close_df["Segment"]],
        text=[f"{p*100:.1f}%" for p in close_df["Closeness"]],
        textposition="outside",
    ))
    fig_close.update_layout(
        xaxis_range=[0, max(close_df["Closeness"]) * 1.25], height=220,
        margin=dict(l=0, r=30, t=10, b=10),
        xaxis_title="Relative closeness", yaxis_title=""
    )
    st.plotly_chart(fig_close, use_container_width=True)

    st.markdown("### 📏 Your Input")
    st.dataframe(pd.DataFrame({
        "Feature": ["Recency (days)", "Frequency (orders)", "Monetary (£)"],
        "Value": [recency, frequency, monetary]
    }), hide_index=True, use_container_width=True)

with col2:
    st.subheader("Where this customer sits")
    fig_scatter = px.scatter(
        rfm, x="Frequency", y="Monetary", color="Segment",
        color_discrete_map={s: v["color"] for s, v in SEGMENT_INFO.items()},
        opacity=0.55, hover_data=["CustomerID", "Recency"],
        labels={"Frequency": "Frequency (orders)", "Monetary": "Monetary (£)"},
    )
    fig_scatter.add_trace(go.Scatter(
        x=[frequency], y=[monetary], mode="markers",
        marker=dict(size=20, color="black", symbol="star", line=dict(width=2, color="white")),
        name="This customer"
    ))
    fig_scatter.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📈 Segment Comparison")
    show_cols = cluster_profile[["Segment", "Recency_mean", "Frequency_mean", "Monetary_mean", "Count"]].rename(
        columns={"Recency_mean": "Avg Recency", "Frequency_mean": "Avg Frequency", "Monetary_mean": "Avg Monetary (£)"}
    )
    st.dataframe(
        show_cols.style.format({"Avg Recency": "{:.0f}", "Avg Frequency": "{:.1f}", "Avg Monetary (£)": "{:.2f}"})
                        .highlight_max(subset=["Avg Monetary (£)"], color="#d4f5dd"),
        hide_index=True, use_container_width=True
    )

st.markdown("---")

# ---------------------------------------------------------
# Business overview metrics
# ---------------------------------------------------------
st.subheader("💰 Business Overview")
total_customers = len(rfm)
total_revenue = rfm["Monetary"].sum()
champ_mask = rfm["Segment"] == "Champions"
champ_customers = champ_mask.sum()
champ_revenue = rfm.loc[champ_mask, "Monetary"].sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Customers", f"{total_customers:,}")
m2.metric("Total Revenue (capped)", f"£{total_revenue:,.0f}")
m3.metric("Champions — % of Customers", f"{champ_customers/total_customers:.1%}" if total_customers else "—")
m4.metric("Champions — % of Revenue", f"{champ_revenue/total_revenue:.1%}" if total_revenue else "—")

# ---------------------------------------------------------
# Explore dataset
# ---------------------------------------------------------
with st.expander("Explore the full customer dataset"):
    tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Pairwise Plot", "Distributions", "Segment Sizes"])

    with tab1:
        st.dataframe(rfm, use_container_width=True, height=300)
        st.caption(f"{len(rfm)} total customers, {rfm['Segment'].nunique()} segments")

    with tab2:
        rfm_features = ["Recency", "Frequency", "Monetary"]
        selected_features = st.multiselect(
            "Choose features to compare", rfm_features, default=["Frequency", "Monetary"]
        )
        if len(selected_features) == 2:
            fig = px.scatter(
                rfm, x=selected_features[0], y=selected_features[1], color="Segment",
                color_discrete_map={s: v["color"] for s, v in SEGMENT_INFO.items()},
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pick exactly 2 features to see a scatter plot.")

    with tab3:
        feature_for_hist = st.selectbox("Feature", ["Recency", "Frequency", "Monetary"])
        fig_hist = px.histogram(
            rfm, x=feature_for_hist, color="Segment", marginal="box",
            color_discrete_map={s: v["color"] for s, v in SEGMENT_INFO.items()},
            opacity=0.7, barmode="overlay"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab4:
        fig_pie = px.pie(
            cluster_profile, names="Segment", values="Count",
            color="Segment", color_discrete_map={s: v["color"] for s, v in SEGMENT_INFO.items()},
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown(
    '<p style="text-align:center; color:#aaa; margin-top:2rem;">Built with scikit-learn + Streamlit 🛍️</p>',
    unsafe_allow_html=True
)
