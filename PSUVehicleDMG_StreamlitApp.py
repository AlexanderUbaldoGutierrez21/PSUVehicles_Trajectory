import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# PAGE SETUP
st.set_page_config(page_title="Vehicles Trajectory Viewer", layout="wide")

st.image("PSU_Logo1.png", width=400)
st.title("PSU Vehicle Trajectories Viewer")
st.markdown("CE 525: Transportation Operations")

# DATA LOADING
default_file = "Trajectories_Data.txt"
uploaded_file = st.sidebar.file_uploader(
    "Upload Trajectories Data", type=["txt", "csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
elif os.path.exists(default_file):
    df = pd.read_csv(default_file, sep=r"\s+", header=None)
else:
    st.error("❌ No data file found. Please upload one.")
    st.stop()

df.columns = ["time", "vehicle_id", "vehicle_type", "location"]

# SIDEBAR FILTER
st.sidebar.image("PSU_Logo2.png", width=125)
st.sidebar.header("⚪ Display Options")

# ADD LOCATION AND TIME RANGE INPUTS
st.sidebar.subheader("Segment Filters")
loc_min = st.sidebar.number_input("Min Location (ft)", value=100.0, step=10.0)
loc_max = st.sidebar.number_input("Max Location (ft)", value=500.0, step=10.0)
time_min = st.sidebar.number_input("Min Time (seconds)", value=60.0, step=10.0)
time_max = st.sidebar.number_input("Max Time (seconds)", value=180.0, step=10.0)

all_ids = sorted(df["vehicle_id"].unique())
select_all = st.sidebar.checkbox("Select All", value=True)

selected_ids = st.sidebar.multiselect(
    "Select Vehicle IDs to Display",
    options=all_ids,
    default=all_ids if select_all else []
)

# APPLY SEGMENT FILTERS FIRST, THEN VEHICLE SELECTION 
segment_filtered_df = df[
    (df["location"] >= loc_min) &
    (df["location"] <= loc_max) &
    (df["time"] >= time_min) &
    (df["time"] <= time_max)
]
filtered_df = segment_filtered_df[segment_filtered_df["vehicle_id"].isin(selected_ids)]

# FUNCTION TO COMPUTE TRAFFIC METRICS
def compute_traffic_metrics(df_segment, loc_min, loc_max, time_min, time_max):
    if df_segment.empty:
        return {"N": 0, "Density": 0.0, "Flow": 0.0, "Avg_Speed": 0.0}

    # VEHICLE COUNT (N)
    N = len(df_segment["vehicle_id"].unique())

    # SEGMENT LENGTH IN MILES (1 MILE = 5280 ft)
    segment_length_mi = (loc_max - loc_min) / 5280.0

    # TIME PERIOD IN HOURS
    time_period_hr = (time_max - time_min) / 3600.0

    # COMPUTE TOTAL DISTANCE TRAVELED (TDT) AND TOTAL TIME SPENT (TTS) IN SEGMENT
    total_distance_traveled = 0.0  # in ft
    total_time_spent = 0.0  # in seconds

    for vid, group in df_segment.groupby("vehicle_id"):
        group = group.sort_values("time")
        if len(group) > 1:
            # FIND ENTRY AND EXIT POINTS FOR EACH VEHICLE
            entry_idx = group["location"].idxmin()  # First point in segment
            exit_idx = group["location"].idxmax()   # Last point in segment

            entry_time = group.loc[entry_idx, "time"]
            exit_time = group.loc[exit_idx, "time"]
            entry_loc = group.loc[entry_idx, "location"]
            exit_loc = group.loc[exit_idx, "location"]

            # TIME SPENT IN SEGMENT
            time_spent = exit_time - entry_time
            if time_spent > 0:
                total_time_spent += time_spent

            # DISTANCE TRAVELED IN SEGMENT
            distance_traveled = exit_loc - entry_loc
            if distance_traveled > 0:
                total_distance_traveled += distance_traveled

    # DENSITY (VEH/MI) = TOTAL TIME SPENT (HR) / SEGMENT LENGTH (MI)
    total_time_spent_hr = total_time_spent / 3600.0
    density = total_time_spent_hr / segment_length_mi if segment_length_mi > 0 else 0

    # AVERAGE SPEED (MI/HR) = TOTAL DISTANCE TRAVELED (MI) / TOTAL TIME SPENT (HR)
    total_distance_traveled_mi = total_distance_traveled / 5280.0
    avg_speed = total_distance_traveled_mi / total_time_spent_hr if total_time_spent_hr > 0 else 0.0

    # FLOW (VEH/HR) = DENSITY * AVG SPEED
    flow = density * avg_speed

    return {
        "N": N,
        "Density": round(density, 2),
        "Flow": round(flow, 2),
        "Avg_Speed": round(avg_speed, 2)
    }

# COMPUTE METRICS FOR THE SEGMENT 
metrics = compute_traffic_metrics(segment_filtered_df, loc_min, loc_max, time_min, time_max)

# DISPLAY TRAFFIC METRICS
st.header("Traffic Flow Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Vehicle Count (N)", metrics["N"])
with col2:
    st.metric("Density (veh/mi)", f"{metrics['Density']:.2f}")
with col3:
    st.metric("Flow (veh/hr)", f"{metrics['Flow']:.2f}")
with col4:
    st.metric("Avg Speed (mi/hr)", f"{metrics['Avg_Speed']:.2f}")

# PLOT
base_colors = [
    "#0D1B2A", "#1B263B", "#415A77", "#778DA9", "#E0E1DD",
    "#0F4C81", "#2A6F97", "#4FB0C6", "#7AD0D9", "#BEE3DB"
]
colors = (base_colors * ((len(all_ids) // len(base_colors)) + 1))[:len(all_ids)]

fig = px.line(
    filtered_df,
    x="time",
    y="location",
    color="vehicle_id",
    color_discrete_sequence=colors,
    title="💻 Time-Space Diagram",
    labels={"time": "Time (seconds)", "location": "Location (feet)", "vehicle_id": "Vehicle ID"}
)

fig.update_layout(
    legend=dict(title="Vehicle ID"),
    hovermode="x unified"
)

# DISPLAY
st.plotly_chart(fig, use_container_width=True)