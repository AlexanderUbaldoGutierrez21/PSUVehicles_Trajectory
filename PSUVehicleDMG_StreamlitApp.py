import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

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

# COMPUTE FULL DATA RANGE
full_loc_min = df["location"].min()
full_loc_max = df["location"].max()
full_time_min = df["time"].min()
full_time_max = df["time"].max()

# ADD CLEAR SEGMENTS CHECKBOX
clear_segments = st.sidebar.checkbox("Clear Segments", value=False)

# SET DEFAULT VALUES BASED ON CHECKBOX
if clear_segments:
    loc_min_val = full_loc_min
    loc_max_val = full_loc_max
    time_min_val = full_time_min
    time_max_val = full_time_max
else:
    loc_min_val = 0.0    
    loc_max_val = full_loc_max 
    time_min_val = 0.0   
    time_max_val = full_time_max 

# ADD LOCATION AND TIME RANGE INPUTS
st.sidebar.subheader("Segment Filters")
loc_min = st.sidebar.number_input("Min Location (ft)", value=loc_min_val, step=10.0, disabled=clear_segments)
loc_max = st.sidebar.number_input("Max Location (ft)", value=loc_max_val, step=10.0, disabled=clear_segments)
time_min = st.sidebar.number_input("Min Time (seconds)", value=time_min_val, step=10.0, disabled=clear_segments)
time_max = st.sidebar.number_input("Max Time (seconds)", value=time_max_val, step=10.0, disabled=clear_segments)

# ADD FREE-FLOW TRAVEL TIME INPUT (set to 0 to avoid filtering vehicles by default)
free_flow_tt = st.sidebar.number_input("Free-Flow Travel Time (seconds)", value=0.0, step=0.1)

all_ids = sorted(df["vehicle_id"].unique())
select_all = st.sidebar.checkbox("Select All", value=True)

selected_ids = st.sidebar.multiselect(
    "Select Vehicle IDs to Display",
    options=all_ids,
    default=all_ids if select_all else []
)

# FUNDAMENTAL DIAGRAM DATA SOURCE OPTION
fd_from_full = st.sidebar.checkbox(
    "Use full dataset for Fundamental Diagram",
    value=True,
    help="Compute u–k regression using the entire dataset instead of the current segment selection."
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
def compute_traffic_metrics(df_segment, loc_min, loc_max, time_min, time_max, free_flow_tt):
    if df_segment.empty:
        return {"N": 0, "Density": 0.0, "Flow": 0.0, "Avg_Speed": 0.0, "Max_Accumulation": 0, "Max_Travel_Time": 0.0, "Avg_Delay": 0.0}

    # VEHICLE COUNT (N)
    N = len(df_segment["vehicle_id"].unique())

    # SEGMENT LENGTH IN MILES (1 MILE = 5280 FT)
    segment_length_mi = (loc_max - loc_min) / 5280.0

    # TIME PERIOD IN HOURS
    time_period_hr = (time_max - time_min) / 3600.0

    # COMPUTE TOTAL DISTANCE TRAVELED (TDT) AND TOTAL TIME SPENT (TTS) IN SEGMENT
    total_distance_traveled = 0.0  # in ft
    total_time_spent = 0.0  # in seconds
    travel_times = []
    delays = []

    for vid, group in df_segment.groupby("vehicle_id"):
        group = group.sort_values("time")
        if len(group) > 1:
            # TIME SPENT IN SEGMENT: from first to last point in the filtered data
            entry_time = group.iloc[0]["time"]
            exit_time = group.iloc[-1]["time"]
            time_spent = exit_time - entry_time
            if time_spent > 0:
                total_time_spent += time_spent
                travel_times.append(time_spent)
                delay = time_spent - free_flow_tt
                if delay > 0:
                    delays.append(delay)

            # DISTANCE TRAVELED IN SEGMENT: sum of absolute distances between consecutive points
            loc_diff = np.diff(group["location"])
            veh_distances = np.abs(loc_diff)
            total_distance_traveled += np.sum(veh_distances)

    # AVERAGE SPEED (MI/HR) = TOTAL DISTANCE TRAVELED (MI) / TOTAL TIME SPENT (HR)
    total_time_spent_hr = total_time_spent / 3600.0
    total_distance_traveled_mi = total_distance_traveled / 5280.0
    avg_speed = total_distance_traveled_mi / total_time_spent_hr if total_time_spent_hr > 0 else 0.0

    # AVERAGE DENSITY (VEH/MI) = (TOTAL TIME SPENT / TIME PERIOD) PER MILE
    # K = (TTS / T) / L  WHERE TTS IN HOURS, T IN HOURS, L IN MILES 
    density = ((total_time_spent_hr / time_period_hr) / segment_length_mi) if (time_period_hr > 0 and segment_length_mi > 0) else 0.0

    # GENERALIZED FLOW (VEH/HR) = K * U
    flow = density * avg_speed

    # MAXIMUM ACCUMULATION: MAXIMUN NUMBER OF VEHICLES IN SEGMENT AT ANY TIME 
    # FOR QUEUING DIAGRAM, ACCUMULATION IS THE DIFFERENCE BETWEEN INPUT AND OUTPUT 
    # BUT HERE, MAX ACCUMULATION IS THE MAX NUMBER OF VEHICLES PRESENT SIMULTANEOUSLY 
    time_bins = np.arange(time_min, time_max + 1, 1)  
    max_accumulation = 0
    for t in time_bins:
        vehicles_at_t = df_segment[(df_segment["time"] >= t) & (df_segment["time"] < t + 1)]["vehicle_id"].nunique()
        max_accumulation = max(max_accumulation, vehicles_at_t)

    # MAXIMUM TRAVEL TIME
    max_travel_time = max(travel_times) if travel_times else 0.0

    # AVERAGE DELAY
    avg_delay = np.mean(delays) if delays else 0.0

    return {
        "N": N,
        "Density": round(density, 2),
        "Flow": round(flow, 2),
        "Avg_Speed": round(avg_speed, 2),
        "Max_Accumulation": max_accumulation,
        "Max_Travel_Time": round(max_travel_time, 2),
        "Avg_Delay": round(avg_delay, 2)
    }

# FUNCTION TO COMPUTE FUNDAMENTAL DIAGRAM PARAMETERS
def compute_fundamental_diagram(df_segment, loc_min, loc_max, time_min, time_max):
    if df_segment.empty:
        return {"Jam_Density": 0.0, "Free_Flow_Speed": 0.0, "Capacity": 0.0, "fitted_curve": None}

    # FIX: Use user-specified segment length for density (do not infer from filtered data range)
    segment_length_ft = (loc_max - loc_min)
    segment_length_mi = segment_length_ft / 5280.0
    print(f"DEBUG: segment_length_mi = {segment_length_mi}")

    # Guard against zero or negative segment length
    if not np.isfinite(segment_length_mi) or segment_length_mi <= 0:
        return {"Jam_Density": 0.0, "Free_Flow_Speed": 0.0, "Capacity": 0.0, "fitted_curve": None}

    # COMPUTE MICROSCOPIC u–k PAIRS PER OBSERVATION (no time binning)
    density_flow_pairs = []
    density_speed_pairs = []

    # Precompute instantaneous speeds (ft/s) per vehicle using forward differences
    df_speed = df_segment.sort_values(["vehicle_id", "time"]).copy()
    df_speed["speed_fps"] = np.nan
    for vid, g in df_speed.groupby("vehicle_id"):
        g = g.sort_values("time")
        dx = g["location"].shift(-1) - g["location"]
        dt = g["time"].shift(-1) - g["time"]
        speed = dx / dt
        # Guard against zero/negative dt
        speed = speed.where(dt > 0, np.nan)
        # assign back; last row per vehicle remains NaN
        df_speed.loc[g.index, "speed_fps"] = speed

    # Precompute number of vehicles present at each timestamp (for density)
    counts_by_time = df_speed.groupby("time")["vehicle_id"].nunique().to_dict()

    # Filter valid microscopic observations (finite, >0 ft/s, <300 ft/s)
    valid_mask = (
        df_speed["speed_fps"].replace([np.inf, -np.inf], np.nan).notna()
        & (df_speed["speed_fps"] > 0)
        & (df_speed["speed_fps"] < 300)
    )
    valid_obs = df_speed[valid_mask]

    # Build (k, u) and (k, q) pairs per observation
    for row in valid_obs.itertuples(index=False):
        num_veh = counts_by_time.get(row.time, 0)
        if segment_length_mi <= 0 or num_veh <= 0:
            continue
        density = num_veh / segment_length_mi  # veh/mi
        u_mph = float(row.speed_fps) * 3600.0 / 5280.0  # mph
        q_veh_hr = density * u_mph  # veh/hr via q = k * u

        if density > 0 and u_mph > 0 and np.isfinite(u_mph):
            density_speed_pairs.append((density, u_mph))
            density_flow_pairs.append((density, q_veh_hr))

    print(f"DEBUG: density_flow_pairs count = {len(density_flow_pairs)}")
    print(f"DEBUG: density_speed_pairs count = {len(density_speed_pairs)}")

    if len(density_flow_pairs) < 3:
        return {"Jam_Density": 0.0, "Free_Flow_Speed": 0.0, "Capacity": 0.0, "fitted_curve": None}

    # CONVERT TO ARRAYS
    densities = np.array([d for d, f in density_flow_pairs])
    flows = np.array([f for d, f in density_flow_pairs])

    # GREENSHIELDS MODEL
    def greenshields_model(k, u_f, k_j):
        return u_f * k * (1 - k / k_j)

    try:
        # Fit Greenshields via linear regression on u-k: u = u_f - (u_f/k_j) * k
        if len(density_speed_pairs) < 3:
            return {"Jam_Density": 0.0, "Free_Flow_Speed": 0.0, "Capacity": 0.0, "fitted_curve": None}

        ks = np.array([d for d, u in density_speed_pairs])
        us = np.array([u for d, u in density_speed_pairs])

        # Optional basic filtering of extreme values
        valid = np.isfinite(ks) & np.isfinite(us) & (ks > 0) & (us > 0) & (us < 1200)  # speeds in mph safeguarded
        ks = ks[valid]
        us = us[valid]

        print(f"DEBUG: regression sample size = {len(ks)}")

        res = linregress(ks, us)
        m, b, r2 = res.slope, res.intercept, res.rvalue ** 2
        print(f"DEBUG: linregress slope={m}, intercept={b}, R2={r2}")

        u_f = max(0.0, float(b))
        if not np.isfinite(m) or m >= 0 or not np.isfinite(u_f) or u_f <= 0:
            raise ValueError("Invalid regression (non-negative slope or non-finite u_f)")

        k_j = -u_f / m  # since m = -u_f / k_j
        if not np.isfinite(k_j) or k_j <= 0:
            raise ValueError("Invalid k_j from regression")

        # Capacity at k = k_j / 2
        capacity = u_f * k_j / 4.0
        print(f"DEBUG: fitted u_f={u_f}, k_j={k_j}")
        print(f"DEBUG: capacity={capacity}")

        # CREATE FITTED CURVE DATA
        k_fit = np.linspace(0, k_j * 1.1, 100)
        q_fit = greenshields_model(k_fit, u_f, k_j)
        fitted_curve = pd.DataFrame({"density": k_fit, "flow": q_fit})

        return {
            "Jam_Density": round(k_j, 2),
            "Free_Flow_Speed": round(u_f, 2),
            "Capacity": round(capacity, 2),
            "fitted_curve": fitted_curve
        }
    except Exception as e:
        print(f"DEBUG: Exception in fitting (u-k regression fallback to q-k observed): {e}")
        # FALLBACK: Use simple estimation based on observed q-k data
        if len(density_flow_pairs) > 0:
            max_density = max(d for d, f in density_flow_pairs)
            max_flow = max(f for d, f in density_flow_pairs)
            avg_speed = max_flow / max_density if max_density > 0 else 0

            # Estimate jam density as 2x max observed density
            jam_density = max_density * 2
            # Estimate free flow speed as max observed speed
            free_flow_speed = avg_speed
            # Estimate capacity using Greenshields capacity formula
            capacity = free_flow_speed * jam_density / 4.0

            # CREATE SIMPLE CURVE WITH ESTIMATES
            k_fit = np.linspace(0, jam_density, 100)
            q_fit = greenshields_model(k_fit, free_flow_speed, jam_density)
            fitted_curve = pd.DataFrame({"density": k_fit, "flow": q_fit})

            return {
                "Jam_Density": round(jam_density, 2),
                "Free_Flow_Speed": round(free_flow_speed, 2),
                "Capacity": round(capacity, 2),
                "fitted_curve": fitted_curve
            }
        else:
            return {"Jam_Density": 0.0, "Free_Flow_Speed": 0.0, "Capacity": 0.0, "fitted_curve": None}

# FUNCTION TO COMPUTE CUMULATIVE INPUT, OUTPUT, AND VIRTUAL ARRIVAL
def compute_cumulative_curves(df_segment, loc_min, loc_max, time_min, time_max, free_flow_tt):
    if df_segment.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # IDENTIFY ENTRY AND EXITS POINTS 
    entry_df = df_segment.groupby("vehicle_id").first().reset_index()
    exit_df = df_segment.groupby("vehicle_id").last().reset_index()

    # CUMULATIVE INPUT: VEHICLES THAT HAVE ENTERED BY TIME T
    input_times = entry_df["time"].sort_values()
    input_cum = []
    for t in np.arange(time_min, time_max + 1, 1):
        cum_input = (input_times <= t).sum()
        input_cum.append({"time": t, "cumulative": cum_input})

    # CUMULATIVE OUTPUT: VEHICLES THAT EXITED BY THE TIME T 
    output_times = exit_df["time"].sort_values()
    output_cum = []
    for t in np.arange(time_min, time_max + 1, 1):
        cum_output = (output_times <= t).sum()
        output_cum.append({"time": t, "cumulative": cum_output})

    # VIRTUAL ARRIVAL CURVE: CUMULATIVE INPUT SHIFTED BY FREE-FLOW TRAVEL TIME 
    virtual_arrival_cum = []
    for t in np.arange(time_min, time_max + 1, 1):
        virtual_t = t - free_flow_tt
        if virtual_t >= time_min:
            cum_virtual = (input_times <= virtual_t).sum()
        else:
            cum_virtual = 0
        virtual_arrival_cum.append({"time": t, "cumulative": cum_virtual})

    return pd.DataFrame(input_cum), pd.DataFrame(output_cum), pd.DataFrame(virtual_arrival_cum)

# COMPUTE METRICS FOR THE SEGMENT
metrics = compute_traffic_metrics(segment_filtered_df, loc_min, loc_max, time_min, time_max, free_flow_tt)

# COMPUTE FUNDAMENTAL DIAGRAM PARAMETERS
df_for_fd = df if fd_from_full else segment_filtered_df
fd_metrics = compute_fundamental_diagram(df_for_fd, loc_min, loc_max, time_min, time_max)

# DISPLAY TRAFFIC METRICS
st.header("Traffic Flow Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Vehicle Count (N)", metrics["N"])
with col2:
    st.metric("Density (veh/mi)", f"{metrics['Density']:.2f}")
with col3:
    st.metric("Generalized Flow (veh/hr)", f"{metrics['Flow']:.2f}")
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
    labels={"time": "t (seconds)", "location": "x (feet)", "vehicle_id": "Vehicle ID"}
)

fig.update_layout(
    legend=dict(title="Vehicle ID"),
    hovermode="x unified"
)

# DISPLAY
st.plotly_chart(fig, use_container_width=True)

# COMPUTE CUMULATIVE CURVES
input_cum_df, output_cum_df, virtual_arrival_cum_df = compute_cumulative_curves(segment_filtered_df, loc_min, loc_max, time_min, time_max, free_flow_tt)

# PLOT INPUT-OUTPUT AND QUEUING SCENARIO
if not input_cum_df.empty and not output_cum_df.empty and not virtual_arrival_cum_df.empty:
    st.header("Input-Output and Queuing Scenario")

    # DISPLAY QUEUING METRICS BELOW THE HEADER (similar to other sections)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Accumulation (veh)", metrics["Max_Accumulation"])
    with col2:
        st.metric("Max Travel Time (s)", f"{metrics['Max_Travel_Time']:.2f}")
    with col3:
        st.metric("Avg Delay (s)", f"{metrics['Avg_Delay']:.2f}")
    with col4:
        st.empty()

    # CREATE THE PLOT
    fig2 = px.line(
        input_cum_df,
        x="time",
        y="cumulative",
        title="💻 Input-Output and Queuing Diagram",
        labels={"time": "t (seconds)", "cumulative": "N (veh)"}
    )

    # ADD OUTPUT CURVE
    fig2.add_trace(
        px.line(output_cum_df, x="time", y="cumulative").data[0]
    )

    # ADD VIRTUAL ARRIVAL CURVE
    fig2.add_trace(
        px.line(virtual_arrival_cum_df, x="time", y="cumulative").data[0]
    )

    # UPDATE TRACES FOR CLARITY WITH CUSTOM COLORS
    fig2.data[0].name = "Cumulative Input"
    fig2.data[0].line.color = "#0D1B2A"
    fig2.data[1].name = "Cumulative Output"
    fig2.data[1].line.color = "#2A6F97"
    fig2.data[2].name = "Virtual Arrival (500 FT)"
    fig2.data[2].line.color = "#7AD0D9"

    fig2.update_layout(
        legend=dict(title="Curves"),
        hovermode="x unified"
    )

    # ADD ON-CHART TEXT LABELS AT THE TOP OF THE CHART
    y_max = max(input_cum_df["cumulative"].max(), output_cum_df["cumulative"].max(), virtual_arrival_cum_df["cumulative"].max())
    x_range = time_max - time_min

    fig2.add_annotation(
        x=time_min + x_range * 0.05,
        y=y_max * 1.1,
        text="<span style='color:#0D1B2A'>●</span> Cumulative Input (veh)",
        showarrow=False,
        xanchor="left",
    )
    fig2.add_annotation(
        x=time_min + x_range * 0.4,
        y=y_max * 1.1,
        text="<span style='color:#415A77'>●</span> Cumulative Output (veh)",
        showarrow=False,
        xanchor="left",
    )
    fig2.add_annotation(
        x=time_min + x_range * 0.75,
        y=y_max * 1.1,
        text="<span style='color:#778DA9'>●</span> Virtual Arrival (500ft)",
        showarrow=False,
        xanchor="left",
    )
    fig2.update_layout(
        showlegend=False,
        margin=dict(t=100) 
    )

    st.plotly_chart(fig2, use_container_width=True)

# FUNDAMENTAL DIAGRAM SECTION
if fd_metrics["fitted_curve"] is not None:
    st.header("Fundamental Diagram")

    # DISPLAY FUNDAMENTAL DIAGRAM METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jam Density (veh/mi)", f"{fd_metrics['Jam_Density']:.2f}")
    with col2:
        st.metric("Free Flow Speed (mi/hr)", f"{fd_metrics['Free_Flow_Speed']:.2f}")
    with col3:
        st.metric("Capacity (veh/hr)", f"{fd_metrics['Capacity']:.2f}")
    with col4:
        st.empty()

    # CREATE FUNDAMENTAL DIAGRAM PLOT
    fig3 = px.line(
        fd_metrics["fitted_curve"],
        x="density",
        y="flow",
        title="💻 Fundamental Diagram (Density vs Flow)",
        labels={"density": "k(Density)", "flow": "q(Flow)"}
    )

    # ADD REFERENCE POINTS FOR KEY TRAFFIC FLOW PARAMETERS
    if fd_metrics["Jam_Density"] > 0:
        # FREE FLOW SPEED POINT (AT DENSITY = 0)
        fig3.add_trace(
            px.scatter(x=[0], y=[fd_metrics["Free_Flow_Speed"]]).data[0]
        )
        fig3.data[-1].name = "Free Flow Speed"
        fig3.data[-1].marker.color = "#778DA9"
        fig3.data[-1].marker.size = 10
        fig3.data[-1].mode = "markers+text"
        fig3.data[-1].text = [f"Free Flow<br>{fd_metrics['Free_Flow_Speed']:.1f} mi/hr"]
        fig3.data[-1].textposition = "top right"

        # CAPACITY POINT (AT OPTIMAL DENSITY K_J/2)
        optimal_density = fd_metrics["Jam_Density"] / 2
        fig3.add_trace(
            px.scatter(x=[optimal_density], y=[fd_metrics["Capacity"]]).data[0]
        )
        fig3.data[-1].name = "Capacity"
        fig3.data[-1].marker.color = "#4FB0C6"
        fig3.data[-1].marker.size = 10
        fig3.data[-1].mode = "markers+text"
        fig3.data[-1].text = [f"Capacity<br>{fd_metrics['Capacity']:.1f} veh/hr"]
        fig3.data[-1].textposition = "top center"

        # JAM DENSITY POINT (AT FLOW = 0)
        fig3.add_trace(
            px.scatter(x=[fd_metrics["Jam_Density"]], y=[0]).data[0]
        )
        fig3.data[-1].name = "Jam Density"
        fig3.data[-1].marker.color = "#E0E1DD"
        fig3.data[-1].marker.size = 10
        fig3.data[-1].mode = "markers+text"
        fig3.data[-1].text = [f"Jam Density<br>{fd_metrics['Jam_Density']:.1f} veh/mi"]
        fig3.data[-1].textposition = "bottom center"

    fig3.update_traces(line=dict(color="#0D1B2A", width=3))
    fig3.update_layout(
        showlegend=False,
        hovermode="x unified"
    )

    st.plotly_chart(fig3, use_container_width=True)