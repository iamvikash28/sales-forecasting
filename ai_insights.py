import os
import json
import pandas as pd
import numpy as np

# ── helpers ────────────────────────────────────────────────────────────────────

def load_data():
    monthly    = pd.read_csv('data/monthly_sales.csv',    parse_dates=['YearMonth'])
    forecast   = pd.read_csv('data/xgb_forecast.csv',     parse_dates=['YearMonth'])
    features   = pd.read_csv('data/monthly_features.csv', parse_dates=['YearMonth'])
    superstore = pd.read_csv('data/superstore_clean.csv', parse_dates=['Order Date'])
    return monthly, forecast, features, superstore


def build_data_summary(monthly, forecast, superstore):
    total_sales   = superstore['Sales'].sum()
    total_profit  = superstore['Profit'].sum()
    margin        = total_profit / total_sales * 100
    recent_3      = monthly.sort_values('YearMonth').tail(3)['Sales'].mean()
    previous_3    = monthly.sort_values('YearMonth').iloc[-6:-3]['Sales'].mean()
    trend_pct     = (recent_3 - previous_3) / previous_3 * 100
    best_month    = monthly.loc[monthly['Sales'].idxmax()]
    worst_month   = monthly.loc[monthly['Sales'].idxmin()]
    cat_sales     = superstore.groupby('Category')['Sales'].sum().to_dict()
    cat_profit    = superstore.groupby('Category')['Profit'].sum().to_dict()
    region_sales  = superstore.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    total_forecast = forecast['Forecast'].sum()
    forecast_vs_recent = (forecast['Forecast'].mean() - recent_3) / recent_3 * 100

    summary = {
        "total_sales"        : total_sales,
        "total_profit"       : total_profit,
        "margin"             : margin,
        "trend_pct"          : trend_pct,
        "recent_3"           : recent_3,
        "best_month"         : best_month['YearMonth'].strftime('%b %Y'),
        "best_month_sales"   : best_month['Sales'],
        "worst_month"        : worst_month['YearMonth'].strftime('%b %Y'),
        "worst_month_sales"  : worst_month['Sales'],
        "cat_sales"          : cat_sales,
        "cat_profit"         : cat_profit,
        "region_sales"       : region_sales.to_dict(),
        "total_forecast"     : total_forecast,
        "forecast_vs_recent" : forecast_vs_recent,
        "forecast_months"    : len(forecast),
    }
    return summary


def generate_all_insights(summary: dict) -> dict:
    """
    Generate insights from real data values — no API needed.
    Every sentence pulls actual numbers from the summary dict.
    """
    s = summary

    trend_dir   = "growing" if s['trend_pct'] > 0 else "declining"
    trend_word  = "up" if s['trend_pct'] > 0 else "down"
    fcast_dir   = "above" if s['forecast_vs_recent'] > 0 else "below"

    # Top category by sales
    top_cat     = max(s['cat_sales'], key=s['cat_sales'].get)
    top_cat_rev = s['cat_sales'][top_cat]
    top_cat_prf = s['cat_profit'][top_cat]
    top_cat_margin = top_cat_prf / top_cat_rev * 100

    # Most profitable category
    profit_cat  = max(s['cat_profit'], key=s['cat_profit'].get)

    # Top region
    top_region  = list(s['region_sales'].keys())[0]
    top_reg_sales = list(s['region_sales'].values())[0]

    # Weakest category by profit margin
    margins = {c: s['cat_profit'][c] / s['cat_sales'][c] * 100 for c in s['cat_sales']}
    weak_cat = min(margins, key=margins.get)
    weak_margin = margins[weak_cat]

    insights = {

        "trend": (
            f"The business is {trend_dir}, with average monthly sales "
            f"{trend_word} {abs(s['trend_pct']):.1f}% compared to the prior period. "
            f"The strongest single month on record was {s['best_month']} at "
            f"${s['best_month_sales']:,.0f}, while {s['worst_month']} was the "
            f"weakest at ${s['worst_month_sales']:,.0f} — a gap that reflects "
            f"strong seasonality in this business. "
            f"The overall profit margin sits at {s['margin']:.1f}%, which is "
            f"healthy for retail but worth monitoring as discounting increases."
        ),

        "category": (
            f"{top_cat} leads all categories with ${top_cat_rev:,.0f} in revenue "
            f"and a {top_cat_margin:.1f}% profit margin, making it the primary "
            f"growth engine to protect and invest in. "
            f"{weak_cat} is the weakest performer with only a {weak_margin:.1f}% "
            f"margin — pricing strategy or supplier costs likely need review. "
            f"If margins in {weak_cat} cannot be improved, reallocating marketing "
            f"spend toward {profit_cat} would improve overall profitability."
        ),

        "forecast": (
            f"The XGBoost model projects ${s['total_forecast']:,.0f} in revenue "
            f"across the next {s['forecast_months']} months — "
            f"{abs(s['forecast_vs_recent']):.1f}% {fcast_dir} recent monthly averages. "
            f"This forecast is based on lag features and seasonality patterns learned "
            f"from {s['best_month']} peaks and accounts for the cyclical slowdowns "
            f"seen historically. "
            f"Inventory and staffing should be planned around the peak months "
            f"to avoid stockouts during the highest-demand period."
        ),

        "risk": (
            f"The biggest opportunity is the {top_region} region, which generates "
            f"${top_reg_sales:,.0f} in sales — expanding customer acquisition "
            f"there with targeted campaigns could yield outsized returns. "
            f"The key risk is the {weak_cat} category, where a {weak_margin:.1f}% "
            f"margin leaves almost no buffer against cost increases or competitive "
            f"pricing pressure. "
            f"A 5% rise in {weak_cat} costs with no price adjustment could push "
            f"that segment into losses — a pricing audit is strongly recommended."
        ),
    }

    return insights


def answer_question(question: str, summary: dict) -> str:
    """
    Rule-based Q&A that matches keywords and returns data-driven answers.
    Works without any API.
    """
    q   = question.lower()
    s   = summary

    top_cat     = max(s['cat_sales'],   key=s['cat_sales'].get)
    profit_cat  = max(s['cat_profit'],  key=s['cat_profit'].get)
    top_region  = list(s['region_sales'].keys())[0]
    margins     = {c: s['cat_profit'][c] / s['cat_sales'][c] * 100 for c in s['cat_sales']}
    weak_cat    = min(margins, key=margins.get)

    if any(w in q for w in ['region', 'expand', 'geographic', 'location']):
        return (
            f"The {top_region} region is the strongest performer with "
            f"${list(s['region_sales'].values())[0]:,.0f} in total sales. "
            f"It would be the highest-confidence region for expansion investment "
            f"given its proven customer base and existing revenue momentum."
        )

    if any(w in q for w in ['profit', 'margin', 'profitable']):
        return (
            f"{profit_cat} is the most profitable category with "
            f"${s['cat_profit'][profit_cat]:,.0f} in total profit and a "
            f"{margins[profit_cat]:.1f}% margin. "
            f"In contrast, {weak_cat} has the thinnest margin at "
            f"{margins[weak_cat]:.1f}% — focus investment on {profit_cat} "
            f"for the best return."
        )

    if any(w in q for w in ['forecast', 'predict', 'future', 'next']):
        fcast_dir = "above" if s['forecast_vs_recent'] > 0 else "below"
        return (
            f"The forecast projects ${s['total_forecast']:,.0f} across the next "
            f"{s['forecast_months']} months, sitting "
            f"{abs(s['forecast_vs_recent']):.1f}% {fcast_dir} recent averages. "
            f"The model is most confident in near-term months; uncertainty "
            f"grows further out, so treat the later months as directional guidance."
        )

    if any(w in q for w in ['risk', 'danger', 'concern', 'worry', 'problem']):
        return (
            f"The main risk is {weak_cat} with a {margins[weak_cat]:.1f}% margin "
            f"— the thinnest of all categories. Any cost increases or discounting "
            f"pressure could push it into unprofitable territory quickly. "
            f"A pricing review for {weak_cat} SKUs is the most urgent action item."
        )

    if any(w in q for w in ['category', 'product', 'segment', 'best']):
        return (
            f"{top_cat} leads in revenue at ${s['cat_sales'][top_cat]:,.0f} "
            f"and {profit_cat} leads in profitability at a "
            f"{margins[profit_cat]:.1f}% margin. "
            f"These two categories should be the core focus of any growth strategy."
        )

    if any(w in q for w in ['trend', 'growing', 'declining', 'direction']):
        direction = "upward" if s['trend_pct'] > 0 else "downward"
        return (
            f"The overall trend is {direction}, with recent months averaging "
            f"${s['recent_3']:,.0f} in sales. "
            f"The business shows strong seasonality — plan resources around "
            f"the historically strong Q4 period to capture peak demand."
        )

    # Default fallback
    return (
        f"Based on the available data: total sales are ${s['total_sales']:,.0f} "
        f"with a {s['margin']:.1f}% profit margin. "
        f"{top_cat} leads in revenue, {top_region} leads by region, and the "
        f"6-month forecast projects ${s['total_forecast']:,.0f}. "
        f"Feel free to ask something more specific about categories, regions, "
        f"forecast, or risks."
    )