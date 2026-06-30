# AI-Powered Sales Forecasting Dashboard

An interactive dashboard that predicts future retail sales using machine learning 
and explains trends in plain English using AI-generated insights.

## Features

- 📈 Sales forecasting with XGBoost and Random Forest models
- 🤖 AI-generated natural language insights on trends, categories, and risks
- 💬 Interactive Q&A — ask questions about the sales data
- 📊 Live filtering by category and region
- 📉 Model performance comparison with feature importance

## Tech stack

- **Python** — Pandas, NumPy for data processing
- **Machine Learning** — XGBoost, Random Forest (scikit-learn)
- **Dashboard** — Streamlit, Plotly
- **AI insights** — Rule-based natural language generation (Claude API ready)

## Dataset

[Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) 
— ~10,000 retail orders from 2014–2017 across the US.

## Project structure

sales-forecasting/
├── app.py                  # Streamlit dashboard
├── ai_insights.py          # AI insight generation module
├── data/                   # Cleaned data and forecasts
├── models/                 # Trained ML models
└── notebooks/              # Data exploration & model training

## How it works
1. Raw sales data is cleaned and aggregated by month
2. Features engineered: lag values, rolling averages, seasonality (sin/cos encoding)
3. XGBoost and Random Forest trained on historical data, evaluated on a held-out test period
4. Best model forecasts the next 6 months of sales
5. Insights module turns the numbers into plain-English business takeaways

## Run locally
```bash
git clone https://github.com/iamvikash28/sales-forecasting.git
cd sales-forecasting
pip install -r requirements.txt
streamlit run app.py
```

## Live demo
[sales-forecasting.streamlit.app](https://sales-forecasting.streamlit.app) *(update after deploying)*
