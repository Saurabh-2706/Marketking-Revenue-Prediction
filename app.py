import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="Marketing Revenue Prediction", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    /* Main Background - Soft Light Gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2D3436;
    }

    /* Floating Card Effect for Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }

    /* Custom Sidebar Title (The Circle Styled Title) */
    .sidebar-title {
        background-color: #6C5CE7;
        color: white;
        padding: 15px;
        border-radius: 50px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(108, 92, 231, 0.3);
    }

    /* Button Styling */
    .stButton>button {
        background-color: #6C5CE7;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        width: 100%;
        font-weight: bold;
    }

    /* Graph Explanation Styling */
    .graph-explanation {
        font-size: 14px;
        color: #636e72;
        font-style: italic;
        margin-bottom: 25px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('Marketing_Data_Clean.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month_Name'] = df['Date'].dt.month_name()
    return df


df = load_data()
COLORS = ['#6C5CE7', '#00CEC9', '#FAB1A0', '#0984E3', '#00B894']

# --- SIDEBAR ---
st.sidebar.markdown('<div class="sidebar-title">Marketing Revenue Prediction</div>', unsafe_allow_html=True)
menu = st.sidebar.selectbox("Navigate To", ["Predict Revenue", "Performance Analytics", "Platform Analysis"])

# --- PAGE 1: PREDICT REVENUE ---
if menu == "Predict Revenue":
    st.title("🔮 Marketing Revenue Prediction")

    try:
        model = joblib.load('marketing_model.pkl')
        model_features = joblib.load('model_features.pkl')

        st.markdown("""
            <div style="background: rgba(255,255,255,0.6); padding: 20px; border-radius: 15px; border: 1px solid white; margin-bottom: 20px;">
                <h3 style='margin-top:0;'>Configure Campaign Parameters</h3>
            </div>
        """, unsafe_allow_html=True)

        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                plat = st.selectbox("Select Platform", df['Platform'].unique())
                camp = st.selectbox("Campaign Type", df['Campaign_Name'].unique())
                budget = st.slider("Budget (INR)", 5000, 500000, 50000)
            with col2:
                imp = st.number_input("Target Impressions", value=100000)
                clk = st.number_input("Target Clicks", value=2000)
                conv = st.number_input("Target Conversions", value=100)

            submit = st.form_submit_button("Predict")

        if submit:
            input_row = pd.DataFrame(columns=model_features).fillna(0)
            input_row.loc[0] = 0
            ctr = (clk / imp) * 100 if imp > 0 else 0
            cr = (conv / clk) * 100 if clk > 0 else 0

            input_row.update(pd.DataFrame({
                'Impressions': [imp], 'Clicks': [clk], 'Cost_INR': [budget],
                'Conversions': [conv], 'CTR_%': [ctr],
                'Conversion_Rate_%': [cr], 'Month': [1], 'DayOfWeek': [0], 'Day': [1]
            }))

            if f'Platform_{plat}' in model_features: input_row[f'Platform_{plat}'] = 1
            if f'Campaign_Name_{camp}' in model_features: input_row[f'Campaign_Name_{camp}'] = 1

            pred = model.predict(input_row[model_features])[0]
            roi = ((pred - budget) / budget) * 100

            st.markdown("---")
            res1, res2 = st.columns(2)
            res1.metric("Predicted Revenue", f"₹{pred:,.2f}", delta=f"{roi:.1f}% ROI")

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=roi,
                title={'text': "Predicted ROI %"},
                gauge={'axis': {'range': [None, 500]}, 'bar': {'color': "#6C5CE7"}}
            ))
            fig_gauge.update_layout(template="plotly_white", height=300, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(
                '<p class="graph-explanation">This gauge visualizes the expected return on investment based on your selected campaign budget and platform parameters.</p>',
                unsafe_allow_html=True)

            # Historical Context Graph
            st.subheader("Historical Context: Cost vs. Revenue")
            fig_pred_context = px.scatter(df, x="Cost_INR", y="Revenue_INR", color="Platform",
                                          title="Current Prediction vs. Historical Data")
            fig_pred_context.add_trace(
                go.Scatter(x=[budget], y=[pred], mode='markers', marker=dict(size=15, color='red', symbol='star'),
                           name='Your Prediction'))
            fig_pred_context.update_layout(template="plotly_white")
            st.plotly_chart(fig_pred_context, use_container_width=True)
            st.markdown(
                '<p class="graph-explanation">The red star indicates where your simulated campaign sits compared to all previously recorded marketing campaigns.</p>',
                unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning("⚠️ Prediction files not found. Please run your model.py script first.")

# --- PAGE 2: PERFORMANCE Analytics ---
elif menu == "Performance Analytics":
    st.title("📊 Performance Analytics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"₹{df['Revenue_INR'].sum():,.0f}")
    m2.metric("Total Spend", f"₹{df['Cost_INR'].sum():,.0f}")
    m3.metric("Avg ROI", f"{df['ROI_%'].mean():.1f}%")
    m4.metric("Total Conversions", f"{df['Conversions'].sum():,}")

    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("🚀 Revenue vs Spend Growth")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Revenue_INR'], fill='tozeroy', name='Revenue', line=dict(color='#00B894')))
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Cost_INR'], fill='tonexty', name='Spend', line=dict(color='#D63031')))
        fig.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<p class="graph-explanation">This area chart tracks the daily balance between marketing expenditure and the resulting revenue generated over time.</p>',
            unsafe_allow_html=True)

    with c2:
        st.subheader("📱 Platform Share")
        fig_pie = px.pie(df, values='Revenue_INR', names='Platform', hole=0.5, color_discrete_sequence=COLORS)
        fig_pie.update_layout(template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown(
            '<p class="graph-explanation">This donut chart illustrates which advertising platforms are contributing the largest percentage to your total revenue.</p>',
            unsafe_allow_html=True)

    st.subheader("🎯 Conversion Efficiency by Campaign")
    fig_bubble = px.scatter(df, x="CTR_%", y="Conversion_Rate_%", size="Revenue_INR", color="Platform",
                            hover_name="Campaign_Name", size_max=60, color_discrete_sequence=COLORS)
    fig_bubble.update_layout(template="plotly_white")
    st.plotly_chart(fig_bubble, use_container_width=True)
    st.markdown(
        '<p class="graph-explanation">The bubble size shows revenue, helping you identify campaigns with high click-through rates and strong conversion efficiency.</p>',
        unsafe_allow_html=True)

# --- PAGE 3: PLATFORM ANALYSIS ---
else:
    st.title("🎯 Platform Efficiency Deep Dive")

    tabs = st.tabs(["ROI Distribution", "Monthly Trends", "Metric Correlations"])

    with tabs[0]:
        st.subheader("ROI Spread per Platform")
        fig_box = px.box(df, x="Platform", y="ROI_%", color="Platform", color_discrete_sequence=COLORS)
        fig_box.update_layout(template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown(
            '<p class="graph-explanation">This box plot highlights the statistical range and consistency of ROI across different platforms to identify the most stable performers.</p>',
            unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("Revenue Trends by Month")
        monthly_data = df.groupby(['Month_Name', 'Platform'])['Revenue_INR'].sum().reset_index()
        fig_bar = px.bar(monthly_data, x="Month_Name", y="Revenue_INR", color="Platform", barmode="group",
                         color_discrete_sequence=COLORS)
        fig_bar.update_layout(template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(
            '<p class="graph-explanation">This bar chart compares the total revenue generated by each platform month-over-month to reveal seasonal trends.</p>',
            unsafe_allow_html=True)

    with tabs[2]:
        st.subheader("Heatmap: Relationship between Metrics")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Viridis')
        fig_heatmap.update_layout(template="plotly_white")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown(
            '<p class="graph-explanation">The heatmap shows the strength of correlation between metrics; higher numbers indicate a stronger mathematical relationship.</p>',
            unsafe_allow_html=True)