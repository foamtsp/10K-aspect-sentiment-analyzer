import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="10-K Filing Analysis Dashboard", layout="wide")

# --- MOCK DATA GENERATOR ---
# 7 aspects for the sentiment radar and table
ASPECTS = [
    "MD&A", 
    "Risk Factors", 
    "Financial Statements", 
    "Business Operations", 
    "Legal Proceedings", 
    "Internal Controls", 
    "Corporate Governance"
]

def generate_mock_scores():
    """Generates random scores bounded in [-1, 1] for the 7 aspects."""
    return [round(random.uniform(-1.0, 1.0), 2) for _ in range(7)]

# --- STYLING FUNCTIONS ---
def color_score(val):
    """Applies green/red/white coloring based on the score threshold."""
    if val > 0.1:
        return 'color: #00FF00;' # Green
    elif val < -0.1:
        return 'color: #FF0000;' # Red
    else:
        return 'color: #FFFFFF;' # White

# --- SIDEBAR: INPUT CONTRACT ---
st.sidebar.title("Configuration")

# 1. Ticker string
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()

# 2. Dropdown list of SEC filings
# In a real app, this would be populated dynamically based on the ticker via an API call
filing_options = ["2023 10-K (Filed Feb 2024)", "2022 10-K (Filed Feb 2023)", "2021 10-K (Filed Feb 2022)"]
selected_filing = st.sidebar.selectbox("Select SEC Filing", filing_options)

st.sidebar.divider()

# 3. Two buttons for Analyze and Force Refresh
col1, col2 = st.sidebar.columns(2)
analyze_clicked = col1.button("Analyze", type="primary", use_container_width=True)
force_clicked = col2.button("Force Refresh", use_container_width=True)

# --- MAIN DASHBOARD: OUTPUT CONTRACT ---
st.title(f"{ticker} 10-K Analysis")

# Trigger analysis based on button clicks
if analyze_clicked or force_clicked:
    
    # --- SIMULATE STATEFUL POLLING LOOP ---
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Simulate the GET /jobs/{id} polling every ~3s (scaled down for demo UX)
    for percent_complete in range(0, 101, 25):
        if percent_complete < 100:
            status_placeholder.info(f"Polling job status... Job is {percent_complete}% complete.")
            time.sleep(0.75) # Simulate network wait / inference time
            progress_bar.progress(percent_complete)
        else:
            status_placeholder.success("Scoring complete!")
            progress_bar.empty()
            time.sleep(0.5)
            status_placeholder.empty()

    # --- TOP ROW METRICS ---
    # Generate mock results
    is_up = random.choice([True, False])
    prediction_text = "UP 🔼" if is_up else "DOWN 🔽"
    confidence = random.randint(55, 98)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Filing Date", value="2024-02-02")
    m2.metric(label="Prediction", value=prediction_text)
    m3.metric(label="Calibrated Confidence", value=f"{confidence}%")
    m4.metric(label="Model Version", value="v2.1.4-beta")
    
    st.divider()

    # --- VISUALIZATIONS & TABULAR DATA ---
    viz_col, table_col = st.columns((3, 2))
    
    scores = generate_mock_scores()
    
    with viz_col:
        st.subheader("Aspect Sentiment Radar")
        
        # Plotly Radar Chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]], # Close the loop
            theta=ASPECTS + [ASPECTS[0]],
            fill='toself',
            name='Sentiment Score',
            line_color='royalblue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-1, 1],
                    tickfont=dict(size=10)
                )
            ),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        st.subheader("Tabular Aspect Scores")
        
        # Create DataFrame
        df = pd.DataFrame({
            "Aspect": ASPECTS,
            "Score": scores
        })
        
        # Sort values for better readability (optional, but good for analysts)
        df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        
        # Apply conditional formatting
        styled_df = df.style.map(color_score, subset=['Score']).format({"Score": "{:.2f}"})
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

else:
    # Empty state UI
    st.info("Select a ticker and filing from the sidebar, then click 'Analyze' to generate insights.")