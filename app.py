import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from ai_insights import load_data, build_data_summary, generate_all_insights, answer_question

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Load data (cached so it's fast on reruns) ───────────────────────────────
@st.cache_data
def get_data():
    monthly, forecast, features, superstore = load_data()
    summary = build_data_summary(monthly, forecast, superstore)
    return monthly, forecast, features, superstore, summary

@st.cache_data
def get_insights(_summary):
    return generate_all_insights(_summary)

@st.cache_resource
def get_models():
    xgb_model = joblib.load('models/xgb_model.pkl')
    rf_model  = joblib.load('models/rf_model.pkl')
    return xgb_model, rf_model

monthly, forecast, features, superstore, summary = get_data()
insights = get_insights(summary)
xgb_model, rf_model = get_models()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 AI-Powered Sales Forecasting Dashboard")
st.caption("Superstore sales data · XGBoost & Random Forest forecasting · AI-generated insights")

# ── KPI row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Revenue",
        f"${summary['total_sales']:,.0f}",
        f"{summary['margin']:.1f}% margin"
    )
with col2:
    st.metric(
        "6-Month Forecast",
        f"${summary['total_forecast']:,.0f}",
        f"{summary['forecast_vs_recent']:+.1f}% vs recent"
    )
with col3:
    st.metric(
        "Recent Trend",
        f"${summary['recent_3']:,.0f}/mo",
        f"{summary['trend_pct']:+.1f}%"
    )
with col4:
    top_cat = max(summary['cat_sales'], key=summary['cat_sales'].get)
    st.metric(
        "Top Category",
        top_cat,
        f"${summary['cat_sales'][top_cat]:,.0f}"
    )

st.divider()

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.header("Dashboard controls")

model_choice = st.sidebar.radio(
    "Forecast model",
    ["XGBoost", "Random Forest"],
    help="XGBoost generally performs better on this dataset"
)

category_filter = st.sidebar.multiselect(
    "Filter by category",
    options=superstore['Category'].unique(),
    default=superstore['Category'].unique()
)

region_filter = st.sidebar.multiselect(
    "Filter by region",
    options=superstore['Region'].unique(),
    default=superstore['Region'].unique()
)

filtered_df = superstore[
    (superstore['Category'].isin(category_filter)) &
    (superstore['Region'].isin(region_filter))
]

st.sidebar.divider()
st.sidebar.metric("Filtered records", f"{len(filtered_df):,}")
st.sidebar.metric("Filtered revenue", f"${filtered_df['Sales'].sum():,.0f}")

# ── Main chart: history + forecast ──────────────────────────────────────────
st.subheader("Sales history & forecast")

forecast_col = forecast  # using xgb_forecast.csv loaded earlier

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=monthly['YearMonth'], y=monthly['Sales'],
    mode='lines+markers', name='Actual sales',
    line=dict(color='#2a78d6', width=2.5),
    marker=dict(size=5)
))

fig.add_trace(go.Scatter(
    x=forecast_col['YearMonth'], y=forecast_col['Forecast'],
    mode='lines+markers', name=f'{model_choice} forecast',
    line=dict(color='#1baf7a', width=2.5, dash='dash'),
    marker=dict(size=6, symbol='diamond')
))

fig.add_trace(go.Scatter(
    x=pd.concat([forecast_col['YearMonth'], forecast_col['YearMonth'][::-1]]),
    y=pd.concat([forecast_col['Forecast']*1.12, (forecast_col['Forecast']*0.88)[::-1]]),
    fill='toself', fillcolor='rgba(27,175,122,0.12)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence band', showlegend=True
))

fig.update_layout(
    height=420,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title='Sales ($)'
)

st.plotly_chart(fig, use_container_width=True)

# ── Two-column section: category + region ───────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Revenue by category")
    cat_data = filtered_df.groupby('Category')['Sales'].sum().reset_index()
    fig_cat = px.bar(
        cat_data, x='Sales', y='Category', orientation='h',
        color='Category',
        color_discrete_sequence=['#2a78d6', '#1baf7a', '#eda100']
    )
    fig_cat.update_layout(height=300, showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_cat, use_container_width=True)

with col_right:
    st.subheader("Revenue by region")
    reg_data = filtered_df.groupby('Region')['Sales'].sum().reset_index()
    fig_reg = px.pie(
        reg_data, values='Sales', names='Region',
        color_discrete_sequence=['#2a78d6', '#1baf7a', '#eda100', '#4a3aa7']
    )
    fig_reg.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_reg, use_container_width=True)

st.divider()

# ── Model performance ────────────────────────────────────────────────────────
st.subheader("Model performance")

perf_col1, perf_col2 = st.columns([1, 1])

with perf_col1:
    st.markdown("**XGBoost vs Random Forest — feature importance**")
    importance_df = pd.DataFrame({
        'Feature': xgb_model.feature_names_in_,
        'XGBoost': xgb_model.feature_importances_,
        'Random Forest': rf_model.feature_importances_
    }).sort_values('XGBoost', ascending=True)

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        y=importance_df['Feature'], x=importance_df['XGBoost'],
        name='XGBoost', orientation='h', marker_color='#2a78d6'
    ))
    fig_imp.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

with perf_col2:
    st.markdown("**Forecast values**")
    st.dataframe(
        forecast_col.rename(columns={'YearMonth':'Month','Forecast':'Predicted Sales'}),
        use_container_width=True,
        hide_index=True
    )

st.divider()

# ── AI Insights section ──────────────────────────────────────────────────────
st.subheader("🤖 AI-generated insights")

tab1, tab2, tab3, tab4 = st.tabs(["Trend", "Category", "Forecast", "Risk & opportunity"])

with tab1:
    st.info(insights['trend'])
with tab2:
    st.info(insights['category'])
with tab3:
    st.info(insights['forecast'])
with tab4:
    st.warning(insights['risk'])

st.divider()

# ── AI Q&A chatbox ────────────────────────────────────────────────────────────
st.subheader("💬 Ask the AI analyst")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input(
    "Ask a question about your sales data",
    placeholder="e.g. Which region should we prioritize for expansion?"
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_clicked = st.button("Ask", type="primary")

if ask_clicked and user_question:
    answer = answer_question(user_question, summary)
    st.session_state.chat_history.append((user_question, answer))

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**Q: {q}**")
    st.write(a)
    st.markdown("---")

# ── Footer ───────────────────────────────────────────────────────────────────
st.caption("Built with Python, XGBoost, Random Forest, and Streamlit · Data: Superstore Sales Dataset")