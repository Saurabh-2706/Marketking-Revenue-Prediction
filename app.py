import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Marketing Revenue Prediction", layout="wide", page_icon="📈")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    color: #2D3436;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.8);
    padding: 20px;
    border-radius: 15px;
}
.sidebar-title {
    background-color: #6C5CE7;
    color: white;
    padding: 15px;
    border-radius: 50px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv('Marketing_Data_Clean.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month_Name'] = df['Date'].dt.month_name()
    return df

df = load_data()
COLORS = ['#6C5CE7', '#00CEC9', '#FAB1A0', '#0984E3', '#00B894']

# ---------------- SIDEBAR ----------------
st.sidebar.markdown('<div class="sidebar-title">Marketing Revenue Prediction</div>', unsafe_allow_html=True)
menu = st.sidebar.selectbox("Navigate To", ["Predict Revenue", "Performance Analytics", "Platform Analysis"])

# ===================== PAGE 1 =====================
if menu == "Predict Revenue":

    st.title("🔮 Marketing Revenue Prediction")

    try:
        model = joblib.load('marketing_model.pkl')
        model_features = joblib.load('model_features.pkl')

        with st.form("predict_form"):
            col1, col2 = st.columns(2)

            with col1:
                plat = st.selectbox("Platform", df['Platform'].unique())
                camp = st.selectbox("Campaign", df['Campaign_Name'].unique())
                budget = st.slider("Budget (INR)", 5000, 500000, 50000)

            with col2:
                imp = st.number_input("Impressions", value=100000)
                clk = st.number_input("Clicks", value=2000)
                conv = st.number_input("Conversions", value=100)

            submit = st.form_submit_button("Predict")

        if submit:
            # ----------- FEATURE ENGINEERING -----------
            ctr = (clk / imp) * 100 if imp > 0 else 0
            cr = (conv / clk) * 100 if clk > 0 else 0

            # ----------- SAFE INPUT CREATION -----------
            input_dict = {col: 0 for col in model_features}

            input_dict.update({
                'Impressions': imp,
                'Clicks': clk,
                'Cost_INR': budget,
                'Conversions': conv,
                'CTR_%': float(ctr),
                'Conversion_Rate_%': float(cr),
                'Month': 1,
                'DayOfWeek': 0,
                'Day': 1
            })

            # One-hot encoding
            if f'Platform_{plat}' in model_features:
                input_dict[f'Platform_{plat}'] = 1

            if f'Campaign_Name_{camp}' in model_features:
                input_dict[f'Campaign_Name_{camp}'] = 1

            input_row = pd.DataFrame([input_dict])

            # Ensure correct order & dtype
            input_row = input_row[model_features].astype(float)

            # ----------- PREDICTION -----------
            pred = model.predict(input_row)[0]
            roi = ((pred - budget) / budget) * 100

            st.markdown("---")

            col1, col2 = st.columns(2)
            col1.metric("Predicted Revenue", f"₹{pred:,.2f}", delta=f"{roi:.1f}% ROI")

            # ----------- GAUGE -----------
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=roi,
                title={'text': "ROI %"},
                gauge={'axis': {'range': [None, 500]}}
            ))

            st.plotly_chart(fig_gauge, use_container_width=True)

            # ----------- SCATTER -----------
            fig = px.scatter(df, x="Cost_INR", y="Revenue_INR", color="Platform")
            fig.add_scatter(x=[budget], y=[pred], mode='markers',
                            marker=dict(size=15, color='red'),
                            name='Prediction')

            st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("Model files not found. Upload marketing_model.pkl and model_features.pkl")

# ===================== PAGE 2 =====================
elif menu == "Performance Analytics":

    st.title("📊 Performance Analytics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Revenue", f"₹{df['Revenue_INR'].sum():,.0f}")
    col2.metric("Spend", f"₹{df['Cost_INR'].sum():,.0f}")
    col3.metric("Avg ROI", f"{df['ROI_%'].mean():.1f}%")
    col4.metric("Conversions", f"{df['Conversions'].sum():,}")

    st.markdown("---")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Revenue_INR'], name='Revenue', fill='tozeroy'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cost_INR'], name='Cost', fill='tonexty'))

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.pie(df, values='Revenue_INR', names='Platform', hole=0.5)
    st.plotly_chart(fig2, use_container_width=True)

# ===================== PAGE 3 =====================
else:

    st.title("🎯 Platform Analysis")

    fig = px.box(df, x="Platform", y="ROI_%", color="Platform")
    st.plotly_chart(fig, use_container_width=True)

    monthly = df.groupby(['Month_Name', 'Platform'])['Revenue_INR'].sum().reset_index()

    fig2 = px.bar(monthly, x="Month_Name", y="Revenue_INR", color="Platform", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    corr = df.select_dtypes(include=[np.number]).corr()
    fig3 = px.imshow(corr, text_auto=True)

    st.plotly_chart(fig3, use_container_width=True)
