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
                st.success(f"""### Status: Healthy
The '{selected_part_name}' is operating within safe lifecycle parameters.""")
            elif prob < 70:
                st.warning(f"""### Status: Monitor Closely
The '{selected_part_name}' is approaching its average failure threshold. Add to inspection route.""")
            else:
                st.error(f"""### Status: Critical
High probability of failure. Schedule preventive replacement for the '{selected_part_name}' during next downtime.""")

# ==========================================
# TAB 5: Prescriptive Analytics
# ==========================================
with tab5:
    st.header("Prescriptive Analytics (Inventory Actions)")
    st.markdown("Automated rule engine prescribing reorder actions based on safety stock levels and lead times.")

    df_parts['Status'] = np.where(df_parts['Stock'] <= df_parts['Min_Stock'], 'Critical Low', 'Healthy')
    df_parts['Prescribed_Reorder_Qty'] = np.where(df_parts['Status'] == 'Critical Low', (df_parts['Min_Stock'] * 2) - df_parts['Stock'], 0)
    df_parts['Action_Priority'] = np.where(
        (df_parts['Status'] == 'Critical Low') & (df_parts['Lead_Time_Wks'] > 4), 'URGENT - Air Freight', 
        np.where(df_parts['Status'] == 'Critical Low', 'Standard Reorder', 'None')
    )

    actionable = df_parts[df_parts['Status'] == 'Critical Low'][['Part_ID', 'Name', 'Stock', 'Min_Stock', 'Lead_Time_Wks', 'Prescribed_Reorder_Qty', 'Action_Priority']]

    if actionable.empty:
        st.success("All inventory levels are healthy. No actions required.")
    else:
        st.error(f"Action Required: {len(actionable)} parts are below minimum stock limits.")
        st.dataframe(actionable.style.map(lambda x: "background-color: #ffcccc" if 'URGENT' in str(x) else ""), use_container_width=True)

# ==========================================
# TAB 6: AI Forecasting
# ==========================================
with tab6:
    st.header("AI Forecasting: 30-Day Spare Parts Cost Projection")
    st.markdown("Uses an **XGBoost Time-Series Model** with lag features and rolling averages to forecast daily parts consumption.")

    @st.cache_data
    def generate_forecast(df_u):
        daily_cost = df_u.groupby('Date')['Total_Cost'].sum().reset_index()
        daily_cost['Day'] = daily_cost['Date'].dt.day
        daily_cost['DayOfWeek'] = daily_cost['Date'].dt.dayofweek
        daily_cost['Month'] = daily_cost['Date'].dt.month

        for i in range(1, 8):
            daily_cost[f'Lag_{i}'] = daily_cost['Total_Cost'].shift(i)
        daily_cost['Rolling_7_Mean'] = daily_cost['Total_Cost'].rolling(window=7).mean()

        df_model = daily_cost.dropna()
        features = ['Day', 'DayOfWeek', 'Month', 'Rolling_7_Mean'] + [f'Lag_{i}' for i in range(1, 8)]

        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
        model.fit(df_model[features], df_model['Total_Cost'])

        last_data = df_model.iloc[-1].copy()
        future_dates = pd.date_range(start=daily_cost['Date'].max() + pd.Timedelta(days=1), periods=30)

        predictions = []
        c_lags = [last_data[f'Lag_{i}'] for i in range(1, 8)]
        c_rolling = last_data['Rolling_7_Mean']

        for date in future_dates:
            row = [date.day, date.dayofweek, date.month, c_rolling] + c_lags
            pred = model.predict(pd.DataFrame([row], columns=features))[0]
            pred = max(0, pred)
            predictions.append({'Date': date, 'Type': 'Forecast', 'Cost': pred})

            c_lags = [pred] + c_lags[:-1]
            c_rolling = np.mean(c_lags)

        historical = daily_cost[['Date', 'Total_Cost']].tail(60).copy()
        historical.rename(columns={'Total_Cost': 'Cost'}, inplace=True)
        historical['Type'] = 'Historical'

        return pd.concat([historical, pd.DataFrame(predictions)])

    with st.spinner("Training XGBoost forecasting model..."):
        forecast_df = generate_forecast(df_usage)

    fig4 = px.line(forecast_df, x='Date', y='Cost', color='Type', 
                   color_discrete_map={"Historical": "#1f77b4", "Forecast": "#d62728"})
    fig4.add_vline(x=forecast_df[forecast_df['Type'] == 'Forecast']['Date'].min(), line_dash="dash", line_color="gray")

    st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# TAB 7: 3D Warehouse Simulation (Static Paths)
# ==========================================
with tab7:
    st.header("🤖 Static 3D Routing: Human vs R2G Paths")
    st.markdown("Displays the complete computed routes for both a Human worker (Aisle-by-Aisle) and R2G Robot (Optimized TSP).")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        num_picks_static = st.slider("Number of Parts in Work Order", 5, 50, 20, key='static_slider')
    with sim_col2:
        st.write("")
        if st.button("🔄 Generate Static Routes", use_container_width=True):
            pass # Triggers re-run to update random points

    max_x, max_y, max_z = 20, 20, 5
    np.random.seed(42) # For static tab consistency across re-runs unless slider changes
    picks_df_static = pd.DataFrame({
        'Pick_ID': range(1, num_picks_static + 1),
        'X': np.random.randint(1, max_x, num_picks_static),
        'Y': np.random.randint(1, max_y, num_picks_static),
        'Z': np.random.randint(0, max_z, num_picks_static)
    })
    start_point_static = pd.DataFrame([{'Pick_ID': 0, 'X': 0, 'Y': 0, 'Z': 0}])

    # Routing Logic
    human_picks_static = picks_df_static.copy().sort_values(by=['Y', 'X']) 
    human_path_static = pd.concat([start_point_static, human_picks_static, start_point_static]).reset_index(drop=True)

    robot_picks_list = picks_df_static[['X', 'Y', 'Z']].values.tolist()
    curr = [0, 0, 0]
    robot_path_coords_static = [curr]
    
    while robot_picks_list:
        next_pick = min(robot_picks_list, key=lambda p: abs(p[0]-curr[0]) + abs(p[1]-curr[1]) + abs(p[2]-curr[2]))
        robot_path_coords_static.append(next_pick)
        robot_picks_list.remove(next_pick)
        curr = next_pick
        
    robot_path_coords_static.append([0, 0, 0])
    robot_path_static = pd.DataFrame(robot_path_coords_static, columns=['X', 'Y', 'Z'])

    fig_3d = go.Figure()

    # Racks Grid
    x_grid, y_grid = np.meshgrid(range(0, max_x, 4), range(0, max_y, 4))
    fig_3d.add_trace(go.Scatter3d(
        x=x_grid.flatten(), y=y_grid.flatten(), z=np.zeros_like(x_grid).flatten(),
        mode='markers', marker=dict(size=2, color='lightgray', opacity=0.3), name='Racks'
    ))

    # Targets
    fig_3d.add_trace(go.Scatter3d(
        x=picks_df_static['X'], y=picks_df_static['Y'], z=picks_df_static['Z'],
        mode='markers', marker=dict(size=6, color='gold', symbol='diamond', line=dict(width=1, color='black')), name='Targets'
    ))

    # Paths
    fig_3d.add_trace(go.Scatter3d(x=human_path_static['X'], y=human_path_static['Y'], z=human_path_static['Z'], mode='lines', line=dict(color='blue', width=4, dash='dot'), name='🚶‍♂️ Human Route'))
    fig_3d.add_trace(go.Scatter3d(x=robot_path_static['X'], y=robot_path_static['Y'], z=robot_path_static['Z'], mode='lines', line=dict(color='red', width=5), name='🤖 R2G Route'))

    fig_3d.update_layout(scene=dict(xaxis=dict(range=[0, max_x]), yaxis=dict(range=[0, max_y]), zaxis=dict(range=[0, max_z])), margin=dict(l=0, r=0, b=0, t=0), height=500)
    st.plotly_chart(fig_3d, use_container_width=True)

# ==========================================
# TAB 8: Live Video Simulator (Animated)
# ==========================================
with tab8:
    st.header("▶️ Live Video Simulator: Real-Time Performance")
    st.markdown("Watch the Human and Robot retrieve parts dynamically. KPIs update in real-time as the simulation runs.")

    col_set1, col_set2 = st.columns(2)
    with col_set1:
        num_picks_live = st.slider("Number of Parts to Pick", 5, 40, 15, key='live_picks')
    with col_set2:
        sim_speed = st.slider("Simulation Speed", 1, 10, 5, key='sim_speed')

    if st.button("🚀 Start Live Video Simulation", use_container_width=True):
        
        # 1. Generate Live Data Points
        max_xl, max_yl = 20, 20
        np.random.seed(int(time.time())) 
        picks_df_live = pd.DataFrame({
            'Pick_ID': range(1, num_picks_live + 1),
            'X': np.random.randint(1, max_xl, num_picks_live),
            'Y': np.random.randint(1, max_yl, num_picks_live)
        })
        start_point_live = pd.DataFrame([{'Pick_ID': 0, 'X': 0, 'Y': 0}])

        # Human Path Logic (Sort by Y, X)
        human_picks_live = picks_df_live.copy().sort_values(by=['Y', 'X']) 
        human_path_live = pd.concat([start_point_live, human_picks_live, start_point_live]).reset_index(drop=True)

        # Robot Path Logic (Nearest Neighbor)
        robot_picks_list_live = picks_df_live[['X', 'Y']].values.tolist()
        curr_l = [0, 0]
        robot_path_coords_live = [curr_l]
        while robot_picks_list_live:
            next_pick = min(robot_picks_list_live, key=lambda p: abs(p[0]-curr_l[0]) + abs(p[1]-curr_l[1]))
            robot_path_coords_live.append(next_pick)
            robot_picks_list_live.remove(next_pick)
            curr_l = next_pick
        robot_path_coords_live.append([0, 0])
        robot_path_live = pd.DataFrame(robot_path_coords_live, columns=['X', 'Y'])

        # 2. Timeline Generation Function
        def build_timeline(path_df, speed_mps, pick_time_s):
            timeline = []
            curr_time, curr_dist, picks = 0, 0, 0
            
            timeline.append({'time': curr_time, 'x': path_df.iloc[0]['X'], 'y': path_df.iloc[0]['Y'], 'dist': curr_dist, 'picks': picks})
            
            for i in range(1, len(path_df)):
                prev = path_df.iloc[i-1]
                curr = path_df.iloc[i]
                
                dist = (abs(curr['X'] - prev['X']) + abs(curr['Y'] - prev['Y'])) * 2.5
                travel_time = dist / speed_mps
                
                curr_time += travel_time
                curr_dist += dist
                timeline.append({'time': curr_time, 'x': curr['X'], 'y': curr['Y'], 'dist': curr_dist, 'picks': picks})
                
                if i < len(path_df) - 1:
                    curr_time += pick_time_s
                    picks += 1
                    timeline.append({'time': curr_time, 'x': curr['X'], 'y': curr['Y'], 'dist': curr_dist, 'picks': picks})
                    
            return pd.DataFrame(timeline)

        # Human = 1.2m/s travel, 15s scan. Robot = 2.0m/s travel, 4s scan.
        h_timeline = build_timeline(human_path_live, 1.2, 15)
        r_timeline = build_timeline(robot_path_live, 2.0, 4)

        def get_state(t, t_df):
            if t >= t_df['time'].max(): return t_df.iloc[-1].to_dict()
            past = t_df[t_df['time'] <= t]
            future = t_df[t_df['time'] > t]
            p0 = past.iloc[-1]
            p1 = future.iloc[0]
            if p1['time'] == p0['time']: return p0.to_dict()
            frac = (t - p0['time']) / (p1['time'] - p0['time'])
            return {
                'x': p0['x'] + frac * (p1['x'] - p0['x']),
                'y': p0['y'] + frac * (p1['y'] - p0['y']),
                'dist': p0['dist'] + frac * (p1['dist'] - p0['dist']),
                'picks': p0['picks']
            }

        # 3. Create UI Placeholders
        st.markdown("---")
        st.markdown("### 📈 Live KPI Feed")
        col_h1, col_h2, col_h3 = st.columns(3)
        mh1 = col_h1.empty()
        mh2 = col_h2.empty()
        mh3 = col_h3.empty()
        
        st.write("")
        col_r1, col_r2, col_r3 = st.columns(3)
        mr1 = col_r1.empty()
        mr2 = col_r2.empty()
        mr3 = col_r3.empty()

        st.markdown("---")
        st.markdown("### 🗺️ Live Top-Down Warehouse Map")
        map_placeholder = st.empty()

        # 4. Run Animation Loop
        max_sim_time = max(h_timeline['time'].max(), r_timeline['time'].max())
        total_frames = 50 # Smoothness 
        
        x_grid_live, y_grid_live = np.meshgrid(range(0, max_xl, 4), range(0, max_yl, 4))
        
        for frame in range(total_frames + 1):
            current_t = frame * (max_sim_time / total_frames)
            
            h_state = get_state(current_t, h_timeline)
            r_state = get_state(current_t, r_timeline)

            # Update Metrics (Dynamic KPIs)
            mh1.metric("🚶‍♂️ Human Travel Distance (m)", f"{h_state['dist']:,.1f} m")
            mh2.metric("🚶‍♂️ Human Time Elapsed (mins)", f"{(current_t/60):,.1f} min")
            mh3.metric("🚶‍♂️ Human Parts Picked", f"{h_state['picks']:.0f} / {num_picks_live}")

            mr1.metric("🤖 R2G Travel Distance (m)", f"{r_state['dist']:,.1f} m")
            mr2.metric("🤖 R2G Time Elapsed (mins)", f"{(current_t/60):,.1f} min")
            mr3.metric("🤖 R2G Parts Picked", f"{r_state['picks']:.0f} / {num_picks_live}")

            # Update Plotly Chart
            fig_live = go.Figure()
            
            # Background Racks
            fig_live.add_trace(go.Scatter(x=x_grid_live.flatten(), y=y_grid_live.flatten(), mode='markers', marker=dict(color='lightgray', size=6, symbol='square'), name='Racks'))
            
            # Target Pick Locations
            fig_live.add_trace(go.Scatter(x=picks_df_live['X'], y=picks_df_live['Y'], mode='markers', marker=dict(color='gold', size=14, symbol='star', line=dict(color='black', width=1)), name='Target Parts'))
            
            # Base Station
            fig_live.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', marker=dict(color='green', size=16, symbol='square'), text=['Base'], textposition="bottom center", name='Base'))

            # Active Human Pos
            fig_live.add_trace(go.Scatter(x=[h_state['x']], y=[h_state['y']], mode='markers+text', marker=dict(color='blue', size=24), text=['🚶‍♂️'], textposition="top center", name='Human'))
            
            # Active Robot Pos
            fig_live.add_trace(go.Scatter(x=[r_state['x']], y=[r_state['y']], mode='markers+text', marker=dict(color='red', size=24), text=['🤖'], textposition="bottom center", name='Robot'))

            fig_live.update_layout(
                height=550, 
                xaxis=dict(range=[-2, max_xl+2], title='X-Axis (Aisles)', showgrid=False), 
                yaxis=dict(range=[-2, max_yl+2], title='Y-Axis (Bays)', showgrid=False),
                showlegend=True,
                margin=dict(l=0, r=0, t=0, b=0)
            )

            map_placeholder.plotly_chart(fig_live, use_container_width=True)
            
            # Control simulation speed
            time.sleep(1.0 / sim_speed)

        st.success("✅ Live Simulation Complete! Notice how the R2G Robot finished significantly faster due to route optimization and faster picking/travel times.")


 
 
