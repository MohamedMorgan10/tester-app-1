import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
import time
warnings.filterwarnings('ignore')
 
# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="FMCG AI inventory optimization management Dashboard", layout="wide")
st.title("🏭 FMCG AI inventory optimization management Dashboard")
 
# ---------------------------------------------------------
# File Upload Widget
# ---------------------------------------------------------
st.markdown("### Data Input")
uploaded_file = st.file_uploader("Upload your 'Maintenance_KPIs_2026_Extended.xlsx' file", type=['xlsx'])
 
if uploaded_file is None:
    st.info("👆 Please upload the generated Excel file to initialize the dashboard.")
    st.stop()
 
# ---------------------------------------------------------
# 1. Data Preprocessing & Cleaning
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data(file):
    xls = pd.ExcelFile(file)
    df_breakdowns = pd.read_excel(xls, 'Breakdowns')
    df_open = pd.read_excel(xls, 'Open hours')
    df_planned = pd.read_excel(xls, 'Planned')
    df_parts = pd.read_excel(xls, 'Spare_Parts')
    df_usage = pd.read_excel(xls, 'Parts_Usage')
 
    for df in [df_breakdowns, df_open, df_planned, df_usage]:
        df['Date'] = pd.to_datetime(df['Date'])
 
    df_breakdowns['Effective DT reverted'] = pd.to_numeric(df_breakdowns['Effective DT reverted'], errors='coerce').fillna(0)
    df_breakdowns.dropna(subset=['Machine', 'Line'], inplace=True)
 
    df_usage_enriched = df_usage.merge(
        df_parts[['Part_ID', 'Name', 'Category', 'Unit_Cost']],
        on='Part_ID', how='left'
    )
 
    df_breakdowns['Month'] = df_breakdowns['Date'].dt.month_name()
    df_breakdowns['Day_of_Week'] = df_breakdowns['Date'].dt.day_name()
 
    return df_breakdowns, df_open, df_planned, df_parts, df_usage_enriched
 
try:
    df_breakdowns, df_open, df_planned, df_parts, df_usage = load_and_preprocess_data(uploaded_file)
except Exception as e:
    st.error(f"⚠️ Error reading the Excel file. Please ensure it has the correct sheets. Details: {e}")
    st.stop()
 
# ---------------------------------------------------------
# Sidebar Filtering
# ---------------------------------------------------------
st.sidebar.header("Filter Data")
selected_lines = st.sidebar.multiselect(
    "Select Production Lines",
    options=df_breakdowns['Line'].unique(),
    default=df_breakdowns['Line'].unique()
)
 
df_bd_filtered = df_breakdowns[df_breakdowns['Line'].isin(selected_lines)]
df_usg_filtered = df_usage[df_usage['Line'].isin(selected_lines)]
 
# ---------------------------------------------------------
# Analytics Functions & UI Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Descriptive",
    "🔍 Diagnostic",
    "⏱️ Predictive (Repair Time)",
    "⚠️ Predictive (Part Failure)",
    "💊 Prescriptive",
    "📈 AI Forecasting",
    "🤖 3D Static Routes",
    "▶️ Live Simulator"
])
 
# ==========================================
# TAB 1: Descriptive Analytics
# ==========================================
with tab1:
    st.header("Descriptive Analytics (What Happened)")
 
    total_downtime = df_bd_filtered['Effective DT reverted'].sum()
    total_cost = df_usg_filtered['Total_Cost'].sum()
    total_breakdowns = len(df_bd_filtered)
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Unplanned Downtime (mins)", f"{total_downtime:,.0f}")
    col2.metric("Total Breakdowns", f"{total_breakdowns}")
    col3.metric("Total Spare Parts Cost (EGP)", f"{total_cost:,.2f}")
 
    st.subheader("Downtime per Production Line")
    dt_per_line = df_bd_filtered.groupby('Line')['Effective DT reverted'].sum().reset_index()
    fig1 = px.bar(dt_per_line, x='Line', y='Effective DT reverted', color='Line', text_auto=True)
    st.plotly_chart(fig1, use_container_width=True)
 
    st.subheader("Mean Time To Repair (MTTR) by Machine")
    mttr_machine = df_bd_filtered.groupby('Machine')['Effective DT reverted'].mean().round(1).reset_index()
    fig2 = px.bar(mttr_machine, x='Effective DT reverted', y='Machine', orientation='h', color='Effective DT reverted', text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)
 
# ==========================================
# TAB 2: Diagnostic Analytics
# ==========================================
with tab2:
    st.header("Diagnostic Analytics (Why It Happened)")
 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pareto Analysis: Downtime by Fault Type")
        pareto = df_bd_filtered.groupby('Description')['Effective DT reverted'].sum().sort_values(ascending=False).reset_index()
        fig3 = px.bar(pareto, x='Description', y='Effective DT reverted', text_auto=True)
        st.plotly_chart(fig3, use_container_width=True)
 
    with col2:
        st.subheader("Machine vs. Fault Breakdown")
        fault_matrix = pd.crosstab(df_bd_filtered['Machine'], df_bd_filtered['Description'])
        st.dataframe(fault_matrix, use_container_width=True)
 
    st.subheader("Technician Performance Variance (Avg Downtime per Fault)")
    tech_var = df_bd_filtered.pivot_table(index='Tech', columns='Description', values='Effective DT reverted', aggfunc='mean').fillna(0).round(1)
    st.dataframe(tech_var.style.background_gradient(cmap='Reds'), use_container_width=True)
 
# ==========================================
# TAB 3: Predictive Analytics (Repair Time)
# ==========================================
with tab3:
    st.header("Predictive Analytics: Estimate Repair Time")
    st.markdown("Uses an underlying **Random Forest Regressor** to predict expected downtime based on historical data context.")
 
    @st.cache_resource
    def train_rf_regressor(df):
        X = df[['Line', 'Machine', 'Tech', 'Description']]
        y = df['Effective DT reverted']
        preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Line', 'Machine', 'Tech', 'Description'])])
        model = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        model.fit(X, y)
        return model
 
    rf_reg_model = train_rf_regressor(df_breakdowns)
 
    col1, col2 = st.columns(2)
    with col1:
        p_line = st.selectbox("Production Line", df_breakdowns['Line'].unique(), key='reg_line')
        p_machine = st.selectbox("Machine", df_breakdowns['Machine'].unique(), key='reg_mach')
    with col2:
        p_tech = st.selectbox("Assigned Technician", df_breakdowns['Tech'].unique(), key='reg_tech')
        p_desc = st.selectbox("Fault Description", df_breakdowns['Description'].unique(), key='reg_desc')
 
    if st.button("Predict Expected Downtime"):
        input_df = pd.DataFrame([[p_line, p_machine, p_tech, p_desc]], columns=['Line', 'Machine', 'Tech', 'Description'])
        prediction = rf_reg_model.predict(input_df)[0]
        st.success(f"**Estimated Repair Time:** {prediction:.0f} minutes")
 
# ==========================================
# TAB 4: Predictive Analytics (Part Failure)
# ==========================================
with tab4:
    st.header("Predictive Maintenance: Specific Part Failure Probability")
    st.markdown("Uses a **Random Forest Classifier** to map a part's current operational lifecycle against its historical Mean Time Between Failures (MTBF) to predict the likelihood of an imminent breakdown.")
 
    @st.cache_resource
    def train_part_failure_model(df_u):
        part_freq = df_u.groupby('Part_ID')['Date'].apply(lambda x: x.sort_values().diff().dt.days.mean()).fillna(45)
 
        data = []
        for part_id, mtbf in part_freq.items():
            for _ in range(50):
                days = np.random.uniform(0, mtbf * 0.8)
                data.append([part_id, days, 0])
            for _ in range(50):
                days = np.random.uniform(mtbf * 0.8, mtbf * 1.5)
                data.append([part_id, days, 1])
 
        df_synth = pd.DataFrame(data, columns=['Part_ID', 'Days_Since_Replacement', 'Failed'])
 
        X = df_synth[['Part_ID', 'Days_Since_Replacement']]
        y = df_synth['Failed']
 
        preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Part_ID'])], remainder='passthrough')
        model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
        model.fit(X, y)
 
        return model, part_freq
 
    rf_clf_model, part_mtbf_dict = train_part_failure_model(df_usage)
 
    col1, col2 = st.columns(2)
    with col1:
        selected_part_name = st.selectbox("Select Part to Analyze", df_parts['Name'].unique())
        selected_part_id = df_parts[df_parts['Name'] == selected_part_name]['Part_ID'].iloc[0]
 
        mtbf_val = part_mtbf_dict.get(selected_part_id, 45)
        st.info(f"Historical Mean Time Between Failures (MTBF) for this part: **{mtbf_val:.0f} days**")
 
    with col2:
        current_days = st.slider("Days Since Last Replacement", min_value=0, max_value=int(mtbf_val * 2), value=int(mtbf_val * 0.5))
 
    if st.button("Predict Failure Probability"):
        input_df = pd.DataFrame([[selected_part_id, current_days]], columns=['Part_ID', 'Days_Since_Replacement'])
        prob = rf_clf_model.predict_proba(input_df)[0][1] * 100
 
        col_gauge, col_text = st.columns([1, 1])
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Failure Probability %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [0, 35], 'color': "#00cc96"},
                        {'range': [35, 70], 'color': "#FFA15A"},
                        {'range': [70, 100], 'color': "#EF553B"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
 
        with col_text:
            st.write("<br><br>", unsafe_allow_html=True)
            if prob < 35:
                st.success(f"### Status: Healthy\nThe '{selected_part_name}' is operating within safe lifecycle parameters.")
            elif prob < 70:
                st.warning(f"### Status: Monitor Closely\nThe '{selected_part_name}' is approaching its average failure threshold. Add to inspection route.")
            else:
                st.error(f"### S
