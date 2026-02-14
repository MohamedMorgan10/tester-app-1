import streamlit as st
st.title("🎈Mohamed Morgan Creations ")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import datetime
from fpdf import FPDF
import io
 
# --- 1. Page Configuration ---
st.set_page_config(page_title="Maintenance Command Center 2026", page_icon="⚙️", layout="wide")
 
# --- 2. CSS STYLING (Industrial/Dark Theme - WHITE HEADERS & CHARTS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
 
    /* Global Background */
    .stApp {
        background-color: #0f172a; /* Deep Slate */
        font-family: 'Roboto', sans-serif;
        color: #e2e8f0;
    }
 
    /* --- FORCE ALL TEXT HEADERS TO PURE WHITE --- */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    [data-testid="stHeader"],
    .css-10trblm {
        color: #ffffff !important;
    }
   
    /* Sidebar Headers */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar .stMarkdown {
        color: #ffffff !important;
    }
 
    /* KPI Card Style */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: #3b82f6;
    }
   
    /* KPI Title - White */
    .kpi-title {
        color: #ffffff !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 8px;
    }
   
    .kpi-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 800;
    }
   
    .kpi-sub {
        font-size: 0.75rem;
        margin-top: 5px;
    }
   
    .status-good { color: #34d399; } /* Green */
    .status-bad { color: #f43f5e; } /* Red */
    .status-warning { color: #fbbf24; } /* Amber */
   
    /* Table Styling - White Headers */
    [data-testid="stDataFrame"] {
        background-color: rgba(30, 41, 59, 0.4);
    }
    [data-testid="stDataFrame"] th {
        color: #ffffff !important;
        font-weight: bold;
    }
   
    /* Tab Labels */
    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1);
        border-bottom-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)
 
# --- 3. HELPER FUNCTIONS ---
def card(col, title, value, sub_text, status="neutral"):
    color_class = f"status-{status}" if status in ["good", "bad", "warning"] else "text-white"
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub {color_class}">{sub_text}</div>
    </div>
    """
    with col:
        st.markdown(html, unsafe_allow_html=True)
 
def apply_chart_style(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", color="#ffffff"),  
        title=dict(font=dict(color="#ffffff", size=18)),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            title_font=dict(color="#ffffff"),
            tickfont=dict(color="#ffffff")
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            title_font=dict(color="#ffffff"),
            tickfont=dict(color="#ffffff")
        ),
        legend=dict(font=dict(color="#ffffff"))
    )
    return fig
 
# --- 4. DATA LOADING ENGINE ---
@st.cache_data
def load_maintenance_data(file):
    try:
        xls = pd.ExcelFile(file, engine='openpyxl')
       
        # Helper to read sheets safely
        def read_sheet(name, header=0):
            matches = [s for s in xls.sheet_names if name.lower() in s.lower()]
            if matches:
                return pd.read_excel(xls, sheet_name=matches[0], header=header)
            return pd.DataFrame()
 
        # Load specific sheets based on your file structure
        df_bd = read_sheet("Breakdowns")
        df_open = read_sheet("Open hours")
        df_plan = read_sheet("Planned")
        df_ops = read_sheet("Operational")
       
        # --- PRE-PROCESSING ---
       
        # 1. Breakdowns
        if not df_bd.empty:
            df_bd['Date'] = pd.to_datetime(df_bd['Date'], errors='coerce')
            df_bd['Effective DT reverted'] = pd.to_numeric(df_bd['Effective DT reverted'], errors='coerce').fillna(0)
            df_bd['Month_Name'] = df_bd['Date'].dt.month_name()
            df_bd['Week_Start'] = df_bd['Date'] - pd.to_timedelta(df_bd['Date'].dt.dayofweek, unit='d')
           
            # Extract Time Info if 'From' exists
            if 'From' in df_bd.columns:
                 # Try convert to string first to handle time objects
                 df_bd['Hour'] = pd.to_datetime(df_bd['From'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
                 # Fallback for mixed formats
                 if df_bd['Hour'].isnull().all():
                      df_bd['Hour'] = pd.to_datetime(df_bd['From'], errors='coerce').dt.hour
            else:
                 df_bd['Hour'] = 0 # Default
 
        # 2. Open Hours
        if not df_open.empty:
            df_open['Date'] = pd.to_datetime(df_open['Date'], errors='coerce')
            df_open['Available mins'] = pd.to_numeric(df_open['Available mins'], errors='coerce').fillna(0)
       
        # 3. Planned
        if not df_plan.empty:
            df_plan['Date'] = pd.to_datetime(df_plan['Date'], errors='coerce')
            df_plan['Duration'] = pd.to_numeric(df_plan['Duration'], errors='coerce').fillna(0)
 
        return {
            "breakdowns": df_bd,
            "open_hours": df_open,
            "planned": df_plan,
            "operational": df_ops
        }
 
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
 
# --- 5. SIDEBAR & FILTERS ---
st.sidebar.title("🛠️ Maint. Cockpit")
uploaded_file = st.sidebar.file_uploader("Upload 'Maintenance KPIs 2026.xlsx'", type=['xlsx'])
 
if not uploaded_file:
    st.info("👋 Please upload your Maintenance Excel file to begin.")
    st.stop()
 
data = load_maintenance_data(uploaded_file)
if not data: st.stop()
 
df_bd = data['breakdowns']
df_open = data['open_hours']
df_plan = data['planned']
 
# --- Global Filters ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter Scope")
 
# Date Filter
min_date = df_bd['Date'].min()
max_date = df_bd['Date'].max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
 
# Line Filter
all_lines = sorted(df_bd['Line'].dropna().unique().tolist())
selected_lines = st.sidebar.multiselect("Production Lines", all_lines, default=all_lines)
 
# Machine Filter (Dependent on Line)
available_machines = df_bd[df_bd['Line'].isin(selected_lines)]['Machine'].dropna().unique().tolist()
selected_machines = st.sidebar.multiselect("Machines", sorted(available_machines), default=sorted(available_machines))
 
# --- FILTER APPLICATION ---
# 1. Filter Breakdowns
mask_bd = (
    (df_bd['Date'] >= pd.to_datetime(date_range[0])) &
    (df_bd['Date'] <= pd.to_datetime(date_range[1])) &
    (df_bd['Line'].isin(selected_lines)) &
    (df_bd['Machine'].isin(selected_machines))
)
df_bd_filt = df_bd[mask_bd]
 
# 2. Filter Planned
if not df_plan.empty:
    mask_plan = (
        (df_plan['Date'] >= pd.to_datetime(date_range[0])) &
        (df_plan['Date'] <= pd.to_datetime(date_range[1])) &
        (df_plan['Line'].isin(selected_lines))
    )
    df_plan_filt = df_plan[mask_plan]
else:
    df_plan_filt = pd.DataFrame()
 
# 2. Filter Open Hours (For Availability Calc)
mask_open = (
    (df_open['Date'] >= pd.to_datetime(date_range[0])) &
    (df_open['Date'] <= pd.to_datetime(date_range[1])) &
    (df_open['Line'].isin(selected_lines))
)
df_open_filt = df_open[mask_open]
 
# --- 6. CORE CALCULATIONS (KPIs) ---
total_bd_mins = df_bd_filt['BD Duration Eff'].sum()
count_bd = len(df_bd_filt)
 
# Calculate Available Time
if not df_open_filt.empty:
    total_avail_mins = df_open_filt['Available mins'].sum()
else:
    days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days + 1
    total_avail_mins = days * 1440 * len(selected_machines)
 
# Standard KPIs
mttr = (total_bd_mins / count_bd) if count_bd > 0 else 0
operating_time = total_avail_mins - total_bd_mins
mtbf = (operating_time / count_bd) if count_bd > 0 else 0
availability = (operating_time / total_avail_mins * 100) if total_avail_mins > 0 else 0
dt_pct = 100 - availability
 
# --- 7. DASHBOARD LAYOUT ---
# Updated Tabs to include Fishbone Analysis
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Executive Overview", "🔧 Breakdown Analysis", "👨‍🔧 Tech & Asset Performance", "🤖 AI Insights", "🐟 Fishbone Analysis"])
 
# =================================================
# TAB 1: EXECUTIVE OVERVIEW (10 KPIs & 10 CHARTS)
# =================================================
with tab1:
    st.markdown("### 🏆 World Class Maintenance Performance")
   
    # KPIs
    planned_mins = df_plan_filt['Duration'].sum() if not df_plan_filt.empty else 0
    total_maint_mins = total_bd_mins + planned_mins
    reactive_ratio = (total_bd_mins / total_maint_mins * 100) if total_maint_mins > 0 else 0
   
    k1, k2, k3, k4, k5 = st.columns(5)
    card(k1, "Availability (OEE)", f"{availability:.1f}%", "Target: >95%", "good" if availability > 95 else "bad")
    card(k2, "MTBF", f"{mtbf/60:.1f} Hr", "Reliability Index", "good" if mtbf > 1440 else "warning")
    card(k3, "MTTR", f"{mttr:.0f} Min", "Repair Efficiency", "good" if mttr < 60 else "bad")
    card(k4, "Downtime Rate", f"{dt_pct:.2f}%", "Capacity Loss", "good" if dt_pct < 5 else "bad")
    card(k5, "Total Failures", f"{count_bd}", "Interventions", "neutral")
 
    k6, k7, k8, k9, k10 = st.columns(5)
    card(k6, "Total Downtime", f"{total_bd_mins/60:,.0f} Hr", "Production Hours Lost", "bad")
    card(k7, "Planned Maint.", f"{planned_mins/60:,.0f} Hr", "Scheduled Work", "neutral")
    card(k8, "Reactive Ratio", f"{reactive_ratio:.1f}%", "Unplanned Work %", "good" if reactive_ratio < 20 else "bad")
    card(k9, "Avg Daily Loss", f"{(total_bd_mins/(days if 'days' in locals() else 30)):.0f} Min", "Burn Rate", "neutral")
   
    if not df_bd_filt.empty:
        top_asset_loss = df_bd_filt.groupby('Machine')['BD Duration Eff'].sum().max()
        bad_actor_impact = (top_asset_loss / total_bd_mins * 100) if total_bd_mins > 0 else 0
    else:
        bad_actor_impact = 0
    card(k10, "Bad Actor Impact", f"{bad_actor_impact:.1f}%", "Single Asset Risk", "bad" if bad_actor_impact > 20 else "good")
 
    st.markdown("---")
    st.markdown("### 📊 Strategic Maintenance Analytics")
 
    # 10 Strategic Charts
    c1, c2 = st.columns(2)
    with c1:
        daily_dt = df_bd_filt.groupby('Date')['BD Duration Eff'].sum().reset_index()
        fig1 = px.area(daily_dt, x='Date', y='BD Duration Eff', title="1. Downtime Trend (Minutes)", color_discrete_sequence=['#ef4444'])
        apply_chart_style(fig1)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        weekly_stats = df_bd_filt.groupby('Week_Start').agg({'BD Duration Eff':'sum', 'Date':'count'}).rename(columns={'Date':'Count', 'BD Duration Eff':'Total_BD'})
        weekly_stats['MTBF_Min'] = ((7*1440*len(selected_machines)) - weekly_stats['Total_BD']) / weekly_stats['Count']
        weekly_stats['MTBF_Hr'] = weekly_stats['MTBF_Min'] / 60
        fig2 = px.line(weekly_stats, x=weekly_stats.index, y='MTBF_Hr', markers=True, title="2. Reliability Growth (MTBF - Weekly)", color_discrete_sequence=['#34d399'])
        apply_chart_style(fig2)
        st.plotly_chart(fig2, use_container_width=True)
 
    c3, c4 = st.columns(2)
    with c3:
        asset_pareto = df_bd_filt.groupby('Machine')['BD Duration Eff'].sum().sort_values(ascending=False).head(10).reset_index()
        fig3 = px.bar(asset_pareto, x='BD Duration Eff', y='Machine', orientation='h', title="3. Top 10 Bad Actor Machines (Pareto)", color='BD Duration Eff', color_continuous_scale='Reds')
        fig3.update_layout(yaxis=dict(autorange="reversed"))
        apply_chart_style(fig3)
        st.plotly_chart(fig3, use_container_width=True)
       
    with c4:
        # 4. Downtime by Line (Replaced Category Chart)
        line_data = df_bd_filt.groupby('Line')['BD Duration Eff'].sum().reset_index()
        fig4 = px.pie(line_data, values='BD Duration Eff', names='Line', title="4. Total Downtime by Line", hole=0.4, color_discrete_sequence=px.colors.sequential.Plasma_r)
        apply_chart_style(fig4)
        st.plotly_chart(fig4, use_container_width=True)
 
    c5, c6 = st.columns(2)
    with c5:
        weekly_stats['MTTR'] = weekly_stats['Total_BD'] / weekly_stats['Count']
        fig5 = px.line(weekly_stats, x=weekly_stats.index, y='MTTR', markers=True, title="5. Repair Efficiency Trend (MTTR)", color_discrete_sequence=['#facc15'])
        apply_chart_style(fig5)
        st.plotly_chart(fig5, use_container_width=True)
    with c6:
        bd_monthly = df_bd_filt.groupby('Month_Name')['BD Duration Eff'].sum().reset_index()
        bd_monthly['Type'] = 'Unplanned'
        if not df_plan_filt.empty:
            df_plan_filt['Month_Name'] = df_plan_filt['Date'].dt.month_name()
            pl_monthly = df_plan_filt.groupby('Month_Name')['Duration'].sum().reset_index()
            pl_monthly.rename(columns={'Duration':'BD Duration Eff'}, inplace=True)
            pl_monthly['Type'] = 'Planned'
            combined_maint = pd.concat([bd_monthly, pl_monthly])
        else:
            combined_maint = bd_monthly
        fig6 = px.bar(combined_maint, x='Month_Name', y='BD Duration Eff', color='Type', title="6. Planned vs. Reactive Workload", barmode='group', color_discrete_map={'Unplanned':'#ef4444', 'Planned':'#3b82f6'})
        apply_chart_style(fig6)
        st.plotly_chart(fig6, use_container_width=True)
 
    c7, c8 = st.columns(2)
    with c7:
        heatmap_data = df_bd_filt.groupby(['Line', 'Month_Name'])['BD Duration Eff'].sum().reset_index()
        fig7 = px.density_heatmap(heatmap_data, x='Month_Name', y='Line', z='BD Duration Eff', title="7. Downtime Intensity Heatmap", color_continuous_scale='Viridis')
        apply_chart_style(fig7)
        st.plotly_chart(fig7, use_container_width=True)
    with c8:
        fig8 = px.histogram(df_bd_filt, x='BD Duration Eff', nbins=30, title="8. Repair Time Distribution (Log Scale)", log_y=True, color_discrete_sequence=['#a855f7'])
        apply_chart_style(fig8)
        st.plotly_chart(fig8, use_container_width=True)
 
    c9, c10 = st.columns(2)
    with c9:
        line_comp = df_bd_filt.groupby('Line').agg({'BD Duration Eff':'sum', 'Date':'count'}).reset_index()
        line_comp.columns = ['Line', 'Total_Duration', 'Count']
        fig9 = px.scatter(line_comp, x='Total_Duration', y='Count', size='Total_Duration', color='Line', title="9. Line Risk Quadrant (Freq vs Severity)", text='Line')
        apply_chart_style(fig9)
        st.plotly_chart(fig9, use_container_width=True)
    with c10:
        if 'Tech' in df_bd_filt.columns:
            tech_perf = df_bd_filt.groupby('Tech').agg({'BD Duration Eff':'mean', 'Date':'count'}).reset_index()
            tech_perf = tech_perf[tech_perf['Date'] > 2] # Filter noise
            fig10 = px.bar(tech_perf, x='Tech', y='BD Duration Eff', title="10. Avg MTTR by Technician", color='Date', color_continuous_scale='Bluyl')
            apply_chart_style(fig10)
            st.plotly_chart(fig10, use_container_width=True)
        else:
            st.info("Technician data not available for Chart 10")
 
# =================================================
# TAB 2: BREAKDOWN ANALYSIS (DEEP DIVE MODIFIED)
# =================================================
with tab2:
    st.markdown("### 🔍 Deep Dive: Root Cause & Frequency Analysis")
   
    # --- DATA PREP FOR NEW ANALYSIS ---
    df_bd_filt['DayOfWeek'] = df_bd_filt['Date'].dt.day_name()
    # Ensure correct order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_bd_filt['DayOfWeek'] = pd.Categorical(df_bd_filt['DayOfWeek'], categories=days_order, ordered=True)
   
    # Count occurrences of each day in the selected date range to calculate true averages
    date_counts = pd.DataFrame({'Date': pd.date_range(start=date_range[0], end=date_range[1])})
    date_counts['DayOfWeek'] = date_counts['Date'].dt.day_name()
    day_counts = date_counts['DayOfWeek'].value_counts()
   
    # Calculate Frequency per Day
    bd_per_day = df_bd_filt['DayOfWeek'].value_counts()
    avg_bd_freq = (bd_per_day / day_counts).reindex(days_order).fillna(0)
   
    # Weekend Calculation (Fri + Sat)
    weekend_days = ['Friday', 'Saturday']
    total_weekend_bds = bd_per_day[weekend_days].sum()
    total_weekend_occurrences = day_counts[weekend_days].sum()
    avg_weekend_bds = total_weekend_bds / total_weekend_occurrences if total_weekend_occurrences > 0 else 0
   
    weekday_days = [d for d in days_order if d not in weekend_days]
    total_weekday_bds = bd_per_day[weekday_days].sum()
    total_weekday_occurrences = day_counts[weekday_days].sum()
    avg_weekday_bds = total_weekday_bds / total_weekday_occurrences if total_weekday_occurrences > 0 else 0
   
    # Other Metrics
    max_downtime = df_bd_filt['BD Duration Eff'].max()
    short_stops = len(df_bd_filt[df_bd_filt['BD Duration Eff'] < 30])
    short_stop_pct = (short_stops / len(df_bd_filt) * 100) if len(df_bd_filt) > 0 else 0
   
    # --- ROW 1: KEY FREQUENCY KPIs ---
    b1, b2, b3, b4 = st.columns(4)
    card(b1, "Avg Weekend BDs", f"{avg_weekend_bds:.1f} / Day", "Fri & Sat", "bad" if avg_weekend_bds > avg_weekday_bds else "good")
    card(b2, "Avg Weekday BDs", f"{avg_weekday_bds:.1f} / Day", "Sun - Thu", "neutral")
    card(b3, "Max Single Event", f"{max_downtime:.0f} Min", "Longest Stop", "bad" if max_downtime > 120 else "neutral")
    card(b4, "Short Stops (<30m)", f"{short_stop_pct:.1f}%", "Minor Stoppages", "warning" if short_stop_pct > 50 else "good")
 
    # --- ROW 2: ADDITIONAL KPIs ---
    b5, b6, b7, b8 = st.columns(4)
    top_bad_actor = df_bd_filt['Machine'].value_counts().idxmax() if not df_bd_filt.empty else "N/A"
    repeat_failures = df_bd_filt['Machine'].value_counts()
    repeat_pct = (len(repeat_failures[repeat_failures > 1]) / len(repeat_failures) * 100) if len(repeat_failures) > 0 else 0
   
    card(b5, "Freq. Bad Actor", f"{top_bad_actor}", "Most Frequent", "bad")
    card(b6, "Recurring Faults", f"{repeat_pct:.1f}%", "Assets w/ >1 BD", "bad" if repeat_pct > 30 else "good")
    card(b7, "Total BD Count", f"{len(df_bd_filt)}", "Selected Period", "neutral")
    card(b8, "MTTR (Current)", f"{mttr:.0f} Min", "Repair Speed", "neutral")
 
    st.markdown("---")
   
    # --- ROW 3: FREQUENCY CHARTS ---
    bf1, bf2 = st.columns(2)
    with bf1:
        # Chart 1: Avg BD Frequency by Day of Week (Requested)
        fig_freq = px.bar(x=avg_bd_freq.index, y=avg_bd_freq.values, title="1. Avg Breakdown Frequency by Day", labels={'x':'Day', 'y':'Avg Count'}, color=avg_bd_freq.values, color_continuous_scale='Reds')
        apply_chart_style(fig_freq)
        st.plotly_chart(fig_freq, use_container_width=True)
       
    with bf2:
        # Chart 2: Breakdown by Hour of Day
        if 'Hour' in df_bd_filt.columns:
            hourly_counts = df_bd_filt.groupby('Hour')['BD Duration Eff'].count().reset_index()
            # Ensure all hours 0-23 exist
            full_hours = pd.DataFrame({'Hour': range(24)})
            hourly_counts = full_hours.merge(hourly_counts, on='Hour', how='left').fillna(0)
            fig_hour = px.bar(hourly_counts, x='Hour', y='BD Duration Eff', title="2. Shift Analysis: Failures by Hour", labels={'BD Duration Eff':'Count'}, color='BD Duration Eff', color_continuous_scale='Viridis')
            apply_chart_style(fig_hour)
            st.plotly_chart(fig_hour, use_container_width=True)
        else:
            st.info("Time data not available for Shift Analysis.")
 
    # --- ROW 4: SEVERITY & TREND ---
    bf3, bf4 = st.columns(2)
    with bf3:
        # Chart 3: Top 10 Bad Actors by Duration
        bad_actors_dur = df_bd_filt.groupby('Machine')['BD Duration Eff'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_bad = px.bar(bad_actors_dur, x='BD Duration Eff', y='Machine', orientation='h', title="3. High Severity Assets (Total Duration)", color='BD Duration Eff', color_continuous_scale='Magma')
        fig_bad.update_layout(yaxis=dict(autorange="reversed"))
        apply_chart_style(fig_bad)
        st.plotly_chart(fig_bad, use_container_width=True)
       
    with bf4:
        # Chart 4: Cumulative Downtime S-Curve
        df_sorted = df_bd_filt.sort_values('Date')
        df_sorted['Cum_Downtime'] = df_sorted['BD Duration Eff'].cumsum()
        fig_cum = px.line(df_sorted, x='Date', y='Cum_Downtime', title="4. Cumulative Downtime (Impact Over Time)", line_shape='hv')
        fig_cum.update_traces(fill='tozeroy', line=dict(color='#3b82f6', width=2))
        apply_chart_style(fig_cum)
        st.plotly_chart(fig_cum, use_container_width=True)
 
    # --- ROW 5: CAUSE & DISTRIBUTION ---
    bf5, bf6 = st.columns(2)
    with bf5:
        # Chart 5: Description Keywords (Top 10)
        desc_counts = df_bd_filt['Description'].value_counts().head(10).reset_index()
        desc_counts.columns = ['Issue', 'Count']
        fig_desc = px.bar(desc_counts, x='Count', y='Issue', orientation='h', title="5. Common Failure Descriptions", color='Count', color_continuous_scale='Teal')
        fig_desc.update_layout(yaxis=dict(autorange="reversed"))
        apply_chart_style(fig_desc)
        st.plotly_chart(fig_desc, use_container_width=True)
       
    with bf6:
        # Chart 6: Duration Distribution Histogram
        fig_hist_bd = px.histogram(df_bd_filt, x='BD Duration Eff', nbins=40, title="6. Failure Severity Distribution", color_discrete_sequence=['#f59e0b'])
        apply_chart_style(fig_hist_bd)
        st.plotly_chart(fig_hist_bd, use_container_width=True)
 
# =================================================
# TAB 3: TECH & ASSET PERFORMANCE (ENRICHED WITH 6 NEW KPIs)
# =================================================
with tab3:
    st.markdown("### 👨‍🔧 Technician Performance & Asset Intelligence")
 
    if 'Tech' in df_bd_filt.columns:
        # --- 1. DATA AGGREGATION ---
        # Base Calculation
        tech_df = df_bd_filt.groupby('Tech').agg(
            Interventions=('Date', 'count'),
            Total_Downtime=('BD Duration Eff', 'sum'),
            Avg_MTTR=('BD Duration Eff', 'mean'),
            Unique_Machines=('Machine', 'nunique'),          # New KPI: Versatility
            Max_Job=('BD Duration Eff', 'max'),                   # New KPI: Hardest Fix
            Min_Job=('BD Duration Eff', 'min'),                   # New KPI: Speed
        ).reset_index()
       
        # Derived KPIs
        total_plant_jobs = len(df_bd_filt)
        tech_df['Workload_Share'] = (tech_df['Interventions'] / total_plant_jobs) * 100 # New KPI: Contribution
        tech_df['Total_Hours'] = tech_df['Total_Downtime'] / 60                         # New KPI: Volume
       
        # Calculate Quick Fix % (<30 mins)
        quick_fixes = df_bd_filt[df_bd_filt['BD Duration Eff'] < 30].groupby('Tech')['Date'].count()
        tech_df = tech_df.merge(quick_fixes.rename('Quick_Fixes'), on='Tech', how='left').fillna(0)
        tech_df['Quick_Fix_Pct'] = (tech_df['Quick_Fixes'] / tech_df['Interventions']) * 100 # New KPI: Efficiency
 
        # Team Overview Stats
        total_techs = tech_df['Tech'].nunique()
        team_avg_mttr = tech_df['Avg_MTTR'].mean()
        total_interventions = tech_df['Interventions'].sum()
 
        # Display Team-Level Summary Cards
        t1, t2, t3 = st.columns(3)
        card(t1, "Active Technicians", f"{total_techs}", "Staff on Duty", "neutral")
        card(t2, "Team Avg MTTR", f"{team_avg_mttr:.0f} Min", "Avg Response Speed", "good" if team_avg_mttr < 60 else "warning")
        card(t3, "Total Interventions", f"{total_interventions}", "Work Orders Closed", "neutral")
 
        st.markdown("---")
        st.subheader("🏆 The Top Performer Leaderboard")
 
        # Top 3 Calculation (Ranking by Interventions - "Most Active")
        top_techs = tech_df.sort_values(by='Interventions', ascending=False).head(3).reset_index(drop=True)
 
        # Podium Display
        col_gold, col_silver, col_bronze = st.columns(3)
       
        def get_tech_kpi(idx):
            if idx < len(top_techs):
                row = top_techs.iloc[idx]
                return row['Tech'], row['Interventions'], row['Avg_MTTR']
            return "N/A", 0, 0
 
        # Gold Medal
        g_name, g_jobs, g_mttr = get_tech_kpi(0)
        with col_gold:
            st.markdown(f"""
            <div style="background-color: rgba(255, 215, 0, 0.2); padding: 20px; border-radius: 10px; text-align: center; border: 1px solid gold;">
                <h1 style="margin:0;">🥇</h1>
                <h3 style="color: gold; margin:0;">{g_name}</h3>
                <p style="color: white; font-size: 1.2rem; font-weight: bold;">{g_jobs} Fixes</p>
                <p style="color: #cbd5e1;">MTTR: {g_mttr:.0f}m</p>
            </div>
            """, unsafe_allow_html=True)
 
        # Silver Medal
        s_name, s_jobs, s_mttr = get_tech_kpi(1)
        with col_silver:
             st.markdown(f"""
            <div style="background-color: rgba(192, 192, 192, 0.2); padding: 20px; border-radius: 10px; text-align: center; border: 1px solid silver;">
                <h1 style="margin:0;">🥈</h1>
                <h3 style="color: silver; margin:0;">{s_name}</h3>
                <p style="color: white; font-size: 1.2rem; font-weight: bold;">{s_jobs} Fixes</p>
                <p style="color: #cbd5e1;">MTTR: {s_mttr:.0f}m</p>
            </div>
            """, unsafe_allow_html=True)
 
        # Bronze Medal
        b_name, b_jobs, b_mttr = get_tech_kpi(2)
        with col_bronze:
             st.markdown(f"""
            <div style="background-color: rgba(205, 127, 50, 0.2); padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #cd7f32;">
                <h1 style="margin:0;">🥉</h1>
                <h3 style="color: #cd7f32; margin:0;">{b_name}</h3>
                <p style="color: white; font-size: 1.2rem; font-weight: bold;">{b_jobs} Fixes</p>
                <p style="color: #cbd5e1;">MTTR: {b_mttr:.0f}m</p>
            </div>
            """, unsafe_allow_html=True)
 
        st.markdown("---")
       
        # --- NEW SECTION: BEST IN CLASS (6 CARDS) ---
        st.subheader("🏅 Best in Class: Performance Highlights")
       
        # Calculate the winners for the 6 new categories
        if not tech_df.empty:
            most_hours_tech = tech_df.loc[tech_df['Total_Hours'].idxmax()]
            fastest_tech = tech_df[tech_df['Interventions'] > 2].sort_values('Avg_MTTR').iloc[0] if len(tech_df[tech_df['Interventions'] > 2]) > 0 else tech_df.iloc[0]
            most_versatile = tech_df.loc[tech_df['Unique_Machines'].idxmax()]
            longest_job_tech = tech_df.loc[tech_df['Max_Job'].idxmax()]
            highest_share = tech_df.loc[tech_df['Workload_Share'].idxmax()]
            quick_fix_king = tech_df.loc[tech_df['Quick_Fix_Pct'].idxmax()]
 
            bc1, bc2, bc3 = st.columns(3)
            card(bc1, "Iron Man (Most Hours)", f"{most_hours_tech['Tech']}", f"{most_hours_tech['Total_Hours']:.1f} Hours Logged", "neutral")
            card(bc2, "The Sprinter (Fastest Avg)", f"{fastest_tech['Tech']}", f"{fastest_tech['Avg_MTTR']:.1f} Mins/Job", "good")
            card(bc3, "The Generalist (Versatility)", f"{most_versatile['Tech']}", f"{most_versatile['Unique_Machines']} Unique Assets", "neutral")
           
            bc4, bc5, bc6 = st.columns(3)
            card(bc4, "Heavy Lifter (Hardest Job)", f"{longest_job_tech['Tech']}", f"{longest_job_tech['Max_Job']:.0f} Min Single Fix", "warning")
            card(bc5, "The Anchor (Workload Share)", f"{highest_share['Tech']}", f"{highest_share['Workload_Share']:.1f}% of Plant Issues", "neutral")
            card(bc6, "Quick Fix Specialist", f"{quick_fix_king['Tech']}", f"{quick_fix_king['Quick_Fix_Pct']:.1f}% Jobs < 30m", "good")
        else:
            st.info("Insufficient data for Best in Class analysis.")
 
        st.markdown("---")
 
        # --- DETAILED TABLE & ASSETS ---
        col_list, col_chart = st.columns([3, 2])
 
        with col_list:
            st.subheader("📋 Detailed Workforce Analytics")
            # Format dataframe for display
            display_df = tech_df.sort_values(by='Interventions', ascending=False)
            st.dataframe(
                display_df,
                column_config={
                    "Tech": "Technician",
                    "Interventions": st.column_config.NumberColumn("Jobs 🔧", format="%d"),
                    "Total_Hours": st.column_config.ProgressColumn("Total Hours ⏳", format="%.1f", min_value=0, max_value=int(display_df['Total_Hours'].max())),
                    "Avg_MTTR": st.column_config.NumberColumn("Avg Speed ⚡", format="%.1f"),
                    "Unique_Machines": st.column_config.NumberColumn("Versatility 🏗️", format="%d"),
                    "Workload_Share": st.column_config.NumberColumn("Share %", format="%.1f"),
                    "Quick_Fix_Pct": st.column_config.NumberColumn("Quick Fix %", format="%.1f"),
                    "Max_Job": st.column_config.NumberColumn("Max Job", format="%d"),
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
 
        with col_chart:
            st.subheader("🚧 Bad Actor Assets")
            # Existing Asset Code
            asset_kpi = df_bd_filt.groupby(['Line', 'Machine']).agg(
                Failures=('Date', 'count'),
                Total_Downtime=('BD Duration Eff', 'sum')
            ).reset_index().sort_values('Total_Downtime', ascending=False).head(10)
           
            fig_asset = px.bar(asset_kpi, x='Total_Downtime', y='Machine', color='Failures', title="Top 10 High-Loss Machines", orientation='h', color_continuous_scale='Viridis')
            fig_asset.update_layout(yaxis=dict(autorange="reversed"))
            apply_chart_style(fig_asset)
            st.plotly_chart(fig_asset, use_container_width=True)
 
    else:
        st.warning("⚠️ Column 'Tech' not found in the uploaded data. Cannot generate Team Analysis.")
 
# =================================================
# TAB 4: AI INSIGHTS (PRESERVED)
# =================================================
with tab4:
    st.markdown("### 🔮 AI Maintenance Forecasting")
   
    daily_dt = df_bd_filt.groupby('Date')['BD Duration Eff'].sum().reset_index()
 
    if len(daily_dt) > 5:
        # Prepare Data for Regression
        daily_dt['Day_Ordinal'] = pd.to_datetime(daily_dt['Date']).map(datetime.datetime.toordinal)
        X = daily_dt[['Day_Ordinal']]
        y = daily_dt['BD Duration Eff']
       
        model = LinearRegression()
        model.fit(X, y)
       
        # Forecast next 30 days
        future_dates = [daily_dt['Date'].max() + datetime.timedelta(days=i) for i in range(1, 31)]
        future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        future_pred = model.predict(future_X)
       
        # Trend Slope
        slope = model.coef_[0]
        trend_text = "Increasing" if slope > 0 else "Decreasing"
        trend_color = "bad" if slope > 0 else "good"
       
        ai1, ai2, ai3 = st.columns(3)
        card(ai1, "Breakdown Trend", trend_text, f"Rate: {slope:.2f} mins/day", trend_color)
        card(ai2, "Predicted Next Month", f"{sum(future_pred)/60:,.0f} hrs", "Total Forecasted Loss", "neutral")
        card(ai3, "Data Points", f"{len(daily_dt)}", "Days Analyzed", "neutral")
       
        # Forecast Plot
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=daily_dt['Date'], y=y, mode='markers', name='Actual History', marker=dict(color='#94a3b8')))
        fig_ai.add_trace(go.Scatter(x=daily_dt['Date'], y=model.predict(X), mode='lines', name='Trend Line', line=dict(color='#f59e0b', width=2)))
        fig_ai.add_trace(go.Scatter(x=future_dates, y=future_pred, mode='lines', name='AI Prediction (30d)', line=dict(color='#3b82f6', dash='dash')))
       
        fig_ai.update_layout(title="Future Downtime Prediction", yaxis_title="Downtime Minutes")
        apply_chart_style(fig_ai)
        st.plotly_chart(fig_ai, use_container_width=True)
       
        # Strategic Advice
        st.info(f"💡 **AI Insight:** The breakdown trend is currently **{trend_text.lower()}**. " +
                ("Consider reviewing PM schedules for aging assets." if slope > 0 else "Maintenance strategies appear effective."))
       
    else:
        st.warning("Not enough data points for AI prediction (Need at least 5 days of data).")
 
    # --- REPORT GENERATION ---
    st.markdown("---")
    st.subheader("📄 Export Report")
   
    if st.button("Generate PDF Report"):
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'Maintenance Engineering Report 2026', 0, 1, 'C')
                self.ln(5)
       
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
       
        pdf.cell(0, 10, f"Report Date: {datetime.date.today()}", 0, 1)
        pdf.cell(0, 10, f"Overall Availability: {availability:.2f}%", 0, 1)
        pdf.cell(0, 10, f"Total Downtime: {total_bd_mins/60:.1f} Hours", 0, 1)
        pdf.cell(0, 10, f"MTTR: {mttr:.1f} Minutes", 0, 1)
        pdf.cell(0, 10, f"MTBF: {mtbf/60:.1f} Hours", 0, 1)
        pdf.ln(10)
       
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Top 5 Bad Actor Machines:", 0, 1)
        pdf.set_font("Arial", size=11)
       
        # Add top 5 machines to PDF
        asset_kpi = df_bd_filt.groupby(['Line', 'Machine']).agg(Failures=('Date', 'count'), Total_Downtime=('BD Duration Eff', 'sum')).reset_index().sort_values('Total_Downtime', ascending=False)
        top_machines = asset_kpi.head(5)
        for index, row in top_machines.iterrows():
            pdf.cell(0, 10, f"- {row['Machine']}: {row['Total_Downtime']} mins", 0, 1)
           
        pdf_output = pdf.output(dest='S').encode('latin-1', 'replace')
        st.download_button("Download PDF", pdf_output, "Maintenance_Report.pdf", "application/pdf")
 
# =================================================
# TAB 5: FISHBONE ANALYSIS (NEW)
# =================================================
with tab5:
    st.markdown("### 🐟 Root Cause Analysis (Ishikawa Diagram)")
    st.markdown("Enter your root causes below to generate the diagram manually.")
   
    # --- Input Section (Left) ---
    col_input, col_viz = st.columns([1, 2.5])
   
    with col_input:
        st.markdown("#### 1. Problem Statement")
        problem = st.text_input("Head of the Fish (Problem)", "High Downtime")
       
        st.markdown("#### 2. The 6Ms (Root Causes)")
        categories = ["Measurement", "Material", "Method", "Environment", "Man Power", "Machine"]
        causes = {}
       
        for cat in categories:
            with st.expander(f"➕ {cat}", expanded=False):
                val = st.text_area(f"Causes for {cat} (one per line)", height=80, key=f"cat_{cat}")
                # Clean inputs: split by new line and remove empty strings
                causes[cat] = [line.strip() for line in val.split('\n') if line.strip()]
 
    # --- Visualization Section (Right) ---
    with col_viz:
        st.markdown("#### 3. Visual Diagram")
       
        if st.button("Generate Fishbone Diagram", type="primary"):
           
            # Initialize Figure
            fig = go.Figure()
           
            # --- 1. Draw Spine ---
            fig.add_trace(go.Scatter(x=[0, 10], y=[0, 0], mode='lines', line=dict(color='white', width=4), hoverinfo='skip'))
           
            # --- 2. Draw Head (Problem) ---
            fig.add_annotation(
                x=10, y=0, text=problem, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, ax=10.1, ay=0,
                bgcolor="#f43f5e", bordercolor="white", borderwidth=2, font=dict(size=14, color="white", weight="bold")
            )
           
            # --- 3. Define Rib Coordinates ---
            # Top ribs (Measurement, Material, Method) -> y positive
            # Bottom ribs (Environment, Man Power, Machine) -> y negative
            # X positions along spine: 2, 5, 8
           
            # Layout Mapping: (Category, x_pos, y_direction)
            layout_map = [
                ("Measurement", 2, 1),
                ("Material", 5, 1),
                ("Method", 8, 1),
                ("Environment", 2, -1),
                ("Man Power", 5, -1),
                ("Machine", 8, -1)
            ]
           
            for cat, x_pos, y_dir in layout_map:
                # End point of the rib (angled)
                y_end = 3.5 * y_dir
                x_end = x_pos - 1
               
                # Draw Rib Line
                fig.add_trace(go.Scatter(
                    x=[x_pos, x_end], y=[0, y_end],
                    mode='lines', line=dict(color='#94a3b8', width=2), hoverinfo='skip'
                ))
               
                # Add Category Label Box
                fig.add_annotation(
                    x=x_end, y=y_end, text=cat, showarrow=False,
                    bgcolor="#3b82f6", font=dict(color="white", weight="bold"), bordercolor="white", borderwidth=1, xanchor="center" if y_dir==1 else "center", yanchor="bottom" if y_dir==1 else "top"
                )
               
                # Draw Sub-causes (Twigs)
                cat_causes = causes.get(cat, [])
                if cat_causes:
                    # Distribute causes along the rib
                    num_causes = len(cat_causes)
                    # Create steps along the rib line for placement
                    x_steps = np.linspace(x_pos, x_end, num_causes + 2)[1:-1]
                    y_steps = np.linspace(0, y_end, num_causes + 2)[1:-1]
                   
                    for i, cause_text in enumerate(cat_causes):
                        cx = x_steps[i]
                        cy = y_steps[i]
                       
                        # Determine twig direction (horizontal)
                        # Top ribs: Text goes left or right? Let's make text float in space
                        # To keep it clean: alternating left/right is hard on a diagonal.
                        # Let's simple draw a horizontal line OUT from the rib.
                       
                        twig_len = 1.5
                        # Alternating sides for visibility if many causes?
                        # Simplification: All twigs point RIGHT for clarity on left-leaning ribs?
                        # Actually ribs lean LEFT (x_pos to x_pos-1).
                        # Let's point twigs RIGHT (+x).
                       
                        tx_end = cx + 0.5
                       
                        # Draw Twig Line
                        fig.add_trace(go.Scatter(
                            x=[cx, tx_end], y=[cy, cy],
                            mode='lines', line=dict(color='#cbd5e1', width=1), hoverinfo='skip'
                        ))
                       
                        # Add Cause Text
                        fig.add_annotation(
                            x=tx_end, y=cy, text=cause_text,
                            showarrow=False, xanchor="left", font=dict(color="#e2e8f0", size=10)
                        )
 
            # --- 4. Chart Styling ---
            fig.update_layout(
                title=f"Ishikawa Diagram: {problem}",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1, 12]),
                yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-5, 5]),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=600
            )
           
            st.plotly_chart(fig, use_container_width=True)
       
        else:
            st.info("👈 Enter problem details and click 'Generate Fishbone Diagram'")
