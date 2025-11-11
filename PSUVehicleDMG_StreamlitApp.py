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

# ADD FREE-FLOW TRAVEL TIME INPUT 
free_flow_tt = st.sidebar.number_input("Free-Flow Travel Time (seconds)", value=0.0, step=0.1)

all_ids = sorted(df["vehicle_id"].unique())
select_all = st.sidebar.checkbox("Select All", value=True)

selected_ids = st.sidebar.multiselect(
    "Select Vehicle IDs to Display",
    options=all_ids,
    default=all_ids if select_all else []
)


# 3-DETECTOR METHODOLOGY SECTION
st.sidebar.subheader("3-Detector Methodology")
detector_1_loc = st.sidebar.number_input("Detector 1 Location (ft)", value=50.0, step=10.0, help="Upstream detector location")
detector_2_loc = st.sidebar.number_input("Detector 2 Location (ft)", value=300.0, step=10.0, help="Middle detector location (to estimate)")
detector_3_loc = st.sidebar.number_input("Detector 3 Location (ft)", value=450.0, step=10.0, help="Downstream detector location")

# TRIANGULAR FUNDAMENTAL DIAGRAM PARAMETERS
st.sidebar.subheader("Triangular FD Parameters")
backward_wave_speed = st.sidebar.number_input("Backward Wave Speed (mph)", value=8.5, step=0.1, help="w_b")
free_flow_speed_fd = st.sidebar.number_input("Free Flow Speed (mph)", value=34.1, step=0.1, help="u_f")
jam_density_fd = st.sidebar.number_input("Jam Density (veh/mi)", value=314.0, step=1.0, help="k_j")

# APPLY SEGMENT FILTERS FIRST, THEN VEHICLE SELECTION
segment_filtered_df = df[
    (df["location"] >= loc_min) &
    (df["location"] <= loc_max) &
    (df["time"] >= time_min) &
    (df["time"] <= time_max)
]
filtered_df = segment_filtered_df[segment_filtered_df["vehicle_id"].isin(selected_ids)]

# FUNCTION TO COMPUTE TRAFFIC METRICS (UNMODIFIED)
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
            # TIME SPENT IN SEGMENT
            entry_time = group.iloc[0]["time"]
            exit_time = group.iloc[-1]["time"]
            time_spent = exit_time - entry_time
            if time_spent > 0:
                total_time_spent += time_spent
                travel_times.append(time_spent)
                delay = time_spent - free_flow_tt
                if delay > 0:
                    delays.append(delay)

            # DISTANCE TRAVELED IN SEGMENT
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

# FUNCTION TO COMPUTE FUNDAMENTAL DIAGRAM PARAMETERS (CORRECTED)
def compute_fundamental_diagram(df_segment, loc_min, loc_max, time_min, time_max):
    if df_segment.empty:
        return {"Jam_Density": 0.0, "Free_Flow_Speed": 0.0, "Capacity": 0.0, "fitted_curve": None}

    # USER-SPECIFIED SEGMENT LENGHT FOR DEBUG/CONTEXT ONLY 
    segment_length_ft = (loc_max - loc_min)
    segment_length_mi = segment_length_ft / 5280.0
    
    # GUARD AGAINST ZERO OR NEGATIVE SEGMENT LENGHT
    if not np.isfinite(segment_length_mi) or segment_length_mi <= 0:
        return {"Jam_Density": 0.0, "Free_Flow_Speed": 0.0, "Capacity": 0.0, "fitted_curve": None}

    # COMPUTE MICROSCOPIC U–K PAIRS PER VEHICLE PER TIME STEP
    density_flow_pairs = []
    density_speed_pairs = []

    # PRECOMPUTE INSTANTANEOUS SPEED (FT/S) PER VEHICLE USING FORWARD DIFFERENCES 
    df_speed = df_segment.sort_values(["vehicle_id", "time"]).copy()
    df_speed["speed_fps"] = np.nan
    for vid, g in df_speed.groupby("vehicle_id"):
        g = g.sort_values("time")
        dx = g["location"].shift(-1) - g["location"]
        dt = g["time"].shift(-1) - g["time"]
        speed = dx / dt
        # GUARD AGAINST ZERO/NEGATIVE DT
        speed = speed.where(dt > 0, np.nan)
        # ASSIGN BACK / LAST ROW PER VEHICLES REMAINS NAN
        df_speed.loc[g.index, "speed_fps"] = speed

    # FILTER VALID MICROSCOPIC OBSERVATIONS (FINITE >0 FT/S, <300 FT/S)
    valid_mask = (
        df_speed["speed_fps"].replace([np.inf, -np.inf], np.nan).notna()
        & (df_speed["speed_fps"] > 0)
        & (df_speed["speed_fps"] < 300)
    )
    valid_obs = df_speed[valid_mask]

    # COMPUTE MICROSCOPIC DENSITY PER VEHICLE PER TIME STEP USING SPACE HEADWAY
    for t, group_t in valid_obs.groupby("time"):
        # SORT BY LOCATION 
        group_t = group_t.sort_values("location")
        locations = group_t["location"].values
        speeds = group_t["speed_fps"].values

        # COMPUTE SPACE HEADWAYS (DISTANCE TO THE NEXT VEHICLE AHEAD)
        headways_ft = np.diff(locations)  
        # FOR EACH VEHICLE I (EXCEPT LAST), DENSITY K_I = 1 / H_I (VEH/FT)
        for i in range(len(headways_ft)):
            h_ft = headways_ft[i]
            if h_ft > 0:
                density_veh_per_ft = 1.0 / h_ft
                density_veh_per_mi = density_veh_per_ft * 5280.0  
                u_mph = float(speeds[i]) * 3600.0 / 5280.0  
                q_veh_hr = density_veh_per_mi * u_mph 

                if density_veh_per_mi > 0 and u_mph > 0 and np.isfinite(u_mph):
                    density_speed_pairs.append((density_veh_per_mi, u_mph))
                    density_flow_pairs.append((density_veh_per_mi, q_veh_hr))

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
        # FIT GREENSHIELDS VIA LINEAR REGRESSION ON U-K / U = U_F - (U_F/K_J) * K
        if len(density_speed_pairs) < 3:
            return {"Jam_Density": 0.0, "Free_Flow_Speed": 0.0, "Capacity": 0.0, "fitted_curve": None}

        ks = np.array([d for d, u in density_speed_pairs])
        us = np.array([u for d, u in density_speed_pairs])

        # OPTIONAL BASIC FILTERING OF EXTREME VALUES 
        valid = np.isfinite(ks) & np.isfinite(us) & (ks > 0) & (us > 0) & (us < 1200) 
        ks = ks[valid]
        us = us[valid]

        print(f"DEBUG: regression sample size = {len(ks)}")

        res = linregress(ks, us)
        m, b, r2 = res.slope, res.intercept, res.rvalue ** 2
        print(f"DEBUG: linregress slope={m}, intercept={b}, R2={r2}")

        u_f = max(0.0, float(b))
        if not np.isfinite(m) or m >= 0 or not np.isfinite(u_f) or u_f <= 0:
            raise ValueError("Invalid regression (non-negative slope or non-finite u_f)")

        k_j = -u_f / m  
        if not np.isfinite(k_j) or k_j <= 0:
            raise ValueError("Invalid k_j from regression")

        # CAPACITY AT K = K_J / 2
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
        # FALLBACK
        if len(density_flow_pairs) > 0:
            max_density = max(d for d, f in density_flow_pairs)
            max_flow = max(f for d, f in density_flow_pairs)
            avg_speed = max_flow / max_density if max_density > 0 else 0

            # ESTIMATE JAM DENSITY AS 2X MAX OBSERVED DENSITY 
            jam_density = max_density * 2
            # ESTIMATE FREE FLOW SPEED AS MAX OBSERVED SPEED 
            free_flow_speed = avg_speed
            # ESTIMATE CAPACITY USING GREENSHIELDS CAPACITY FORMULA 
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


# FUNCTION TO COMPUTE CUMULATIVE INPUT, OUTPUT, AND VIRTUAL ARRIVAL (UNMODIFIED)
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

# FUNCTION TO COMPUTE CUMULATIVE VEHICLES PASSED A SPECIFIC LOCATION
def compute_cumulative_at_location(df_full, location, time_min, time_max):
    """
    Compute cumulative vehicles that have passed a specific location by time t.
    A vehicle has "passed" the location if its maximum location >= location.
    """
    if df_full.empty:
        return pd.DataFrame()

    # For each vehicle, find the time when it first reaches or exceeds the location
    passage_times = []
    for vid, group in df_full.groupby("vehicle_id"):
        group = group.sort_values("time")
        # Find first time where location >= target_location
        passed = group[group["location"] >= location]
        if not passed.empty:
            passage_time = passed.iloc[0]["time"]
            passage_times.append(passage_time)

    passage_times = sorted(passage_times)

    # Compute cumulative counts
    cum_data = []
    for t in np.arange(time_min, time_max + 1, 1):
        cum_count = sum(1 for pt in passage_times if pt <= t)
        cum_data.append({"time": t, "cumulative": cum_count})

    return pd.DataFrame(cum_data)

# FUNCTION TO COMPUTE TRIANGULAR FUNDAMENTAL DIAGRAM
def triangular_fundamental_diagram(u_f, w_b, k_j):
    """
    Create triangular fundamental diagram data.
    u_f: free flow speed (mph)
    w_b: backward wave speed (mph)
    k_j: jam density (veh/mi)
    """
    # Critical density k_c = u_f / (u_f + w_b) * k_j
    k_c = (u_f / (u_f + w_b)) * k_j
    # Capacity q_c = u_f * k_c
    q_c = u_f * k_c

    # Create density array
    k_vals = np.linspace(0, k_j, 100)

    # Flow values
    q_vals = []
    for k in k_vals:
        if k <= k_c:
            q = u_f * k
        else:
            q = q_c - w_b * (k - k_c)
        q_vals.append(q)

    return pd.DataFrame({"density": k_vals, "flow": q_vals}), k_c, q_c

# FUNCTION FOR 3-DETECTOR ESTIMATION
def estimate_cumulative_3_detector(cum_1_df, cum_3_df, loc_1, loc_3, loc_2, u_f, w_b, k_j, time_min, time_max):
    """
    Estimate cumulative at location 2 using 3-detector methodology with triangular FD.
    cum_1_df, cum_3_df: DataFrames with 'time' and 'cumulative' columns
    loc_1, loc_3, loc_2: detector locations (ft)
    u_f, w_b, k_j: triangular FD parameters
    """
    if cum_1_df.empty or cum_3_df.empty:
        return pd.DataFrame()

    # CONVERT TO MPH AND VEH/MI UNITS
    u_f_fps = u_f * 5280 / 3600  # ft/s
    w_b_fps = w_b * 5280 / 3600  # ft/s

    # DISTANCE BETWEEN DETECTORS
    d_13 = abs(loc_3 - loc_1)  # ft
    d_12 = abs(loc_2 - loc_1)  # ft

    estimated_cum = []

    for t in np.arange(time_min, time_max + 1, 1):
        # Get cumulative at detectors 1 and 3 at time t
        cum_1_t = cum_1_df[cum_1_df["time"] == t]["cumulative"].values
        cum_3_t = cum_3_df[cum_3_df["time"] == t]["cumulative"].values

        if len(cum_1_t) == 0 or len(cum_3_t) == 0:
            estimated_cum.append({"time": t, "cumulative": 0})
            continue

        cum_1_t = cum_1_t[0]
        cum_3_t = cum_3_t[0]

        # TIME FOR FREE FLOW TRAVEL BETWEEN DETECTORS
        t_ff_13 = d_13 / u_f_fps  # seconds
        t_ff_12 = d_12 / u_f_fps  # seconds

        # Estimate density and flow using triangular FD
        # This is a simplified approach - in practice, more sophisticated methods exist
        # For triangular FD, we can estimate based on the difference in cumulative counts

        # Simple estimation: assume linear propagation for free flow
        # More accurate would involve solving the kinematic wave equations
        if cum_3_t > cum_1_t:
            # Traffic is moving, estimate based on time shift
            time_shift = t_ff_12
            t_est = t - time_shift
            if t_est >= time_min:
                cum_est = cum_1_df[cum_1_df["time"] <= t_est]["cumulative"].max()
                if np.isnan(cum_est):
                    cum_est = 0
            else:
                cum_est = 0
        else:
            cum_est = cum_1_t  # If no vehicles passed detector 3, assume same as detector 1

        estimated_cum.append({"time": t, "cumulative": int(cum_est)})

    return pd.DataFrame(estimated_cum)

# COMPUTE METRICS FOR THE SEGMENT
metrics = compute_traffic_metrics(segment_filtered_df, loc_min, loc_max, time_min, time_max, free_flow_tt)

# COMPUTE FUNDAMENTAL DIAGRAM PARAMETERS
fd_metrics = compute_fundamental_diagram(segment_filtered_df, loc_min, loc_max, time_min, time_max)

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

# COMPUTE CUMULATIVE CURVES FOR DETECTOR LOCATIONS
cum_detector_1_df = compute_cumulative_at_location(df, detector_1_loc, time_min, time_max)
cum_detector_2_df = compute_cumulative_at_location(df, detector_2_loc, time_min, time_max)
cum_detector_3_df = compute_cumulative_at_location(df, detector_3_loc, time_min, time_max)

# COMPUTE TRIANGULAR FUNDAMENTAL DIAGRAM
triangular_fd_df, k_c, q_c = triangular_fundamental_diagram(free_flow_speed_fd, backward_wave_speed, jam_density_fd)

# ESTIMATE CUMULATIVE AT DETECTOR 2 USING 3-DETECTOR METHODOLOGY
estimated_cum_detector_2_df = estimate_cumulative_3_detector(
    cum_detector_1_df, cum_detector_3_df,
    detector_1_loc, detector_3_loc, detector_2_loc,
    free_flow_speed_fd, backward_wave_speed, jam_density_fd,
    time_min, time_max
)

# PLOT INPUT-OUTPUT AND QUEUING SCENARIO
if not input_cum_df.empty and not output_cum_df.empty and not virtual_arrival_cum_df.empty:
    st.header("Input-Output / Queuing Scenario")

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
        title="💻 Cumulative Count Diagram",
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
    fig2.data[0].name = "Arrival Curve"
    fig2.data[0].line.color = "#0D1B2A"
    fig2.data[1].name = "Departure Curve"
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
        text="<span style='color:#0D1B2A'>●</span> Arrival Curve (veh)",
        showarrow=False,
        xanchor="left",
    )
    fig2.add_annotation(
        x=time_min + x_range * 0.4,
        y=y_max * 1.1,
        text="<span style='color:#415A77'>●</span> Departure Curve (veh)",
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

# TRIANGULAR FUNDAMENTAL DIAGRAM SECTION
st.header("Triangular Fundamental Diagram")

# DISPLAY TRIANGULAR FD METRICS
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Free Flow Speed (mph)", f"{free_flow_speed_fd:.1f}")
with col2:
    st.metric("Backward Wave Speed (mph)", f"{backward_wave_speed:.1f}")
with col3:
    st.metric("Jam Density (veh/mi)", f"{jam_density_fd:.0f}")
with col4:
    st.metric("Capacity (veh/hr)", f"{q_c:.1f}")

# CREATE TRIANGULAR FUNDAMENTAL DIAGRAM PLOT
fig_tri = px.line(
    triangular_fd_df,
    x="density",
    y="flow",
    title="💻 Triangular Fundamental Diagram",
    labels={"density": "k (Density, veh/mi)", "flow": "q (Flow, veh/hr)"}
)

# ADD REFERENCE POINTS
# FREE FLOW SPEED POINT (AT DENSITY = 0)
fig_tri.add_trace(
    px.scatter(x=[0], y=[free_flow_speed_fd]).data[0]
)
fig_tri.data[-1].name = "Free Flow Speed"
fig_tri.data[-1].marker.color = "#778DA9"
fig_tri.data[-1].marker.size = 10
fig_tri.data[-1].mode = "markers+text"
fig_tri.data[-1].text = [f"Free Flow<br>{free_flow_speed_fd:.1f} mph"]
fig_tri.data[-1].textposition = "top right"

# CAPACITY POINT (AT CRITICAL DENSITY)
fig_tri.add_trace(
    px.scatter(x=[k_c], y=[q_c]).data[0]
)
fig_tri.data[-1].name = "Capacity"
fig_tri.data[-1].marker.color = "#4FB0C6"
fig_tri.data[-1].marker.size = 10
fig_tri.data[-1].mode = "markers+text"
fig_tri.data[-1].text = [f"Capacity<br>{q_c:.1f} veh/hr"]
fig_tri.data[-1].textposition = "top center"

# JAM DENSITY POINT (AT FLOW = 0)
fig_tri.add_trace(
    px.scatter(x=[jam_density_fd], y=[0]).data[0]
)
fig_tri.data[-1].name = "Jam Density"
fig_tri.data[-1].marker.color = "#E0E1DD"
fig_tri.data[-1].marker.size = 10
fig_tri.data[-1].mode = "markers+text"
fig_tri.data[-1].text = [f"Jam Density<br>{jam_density_fd:.0f} veh/mi"]
fig_tri.data[-1].textposition = "bottom center"

fig_tri.update_traces(line=dict(color="#0D1B2A", width=3))
fig_tri.update_layout(
    showlegend=False,
    hovermode="x unified"
)

st.plotly_chart(fig_tri, use_container_width=True)

# DETECTOR CUMULATIVE CURVES SECTION
if not cum_detector_1_df.empty and not cum_detector_2_df.empty and not cum_detector_3_df.empty:
    st.header("Detector Cumulative Curves")

    # DISPLAY DETECTOR METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"Detector 1 ({detector_1_loc} ft)", f"{cum_detector_1_df['cumulative'].max()} veh")
    with col2:
        st.metric(f"Detector 2 ({detector_2_loc} ft)", f"{cum_detector_2_df['cumulative'].max()} veh")
    with col3:
        st.metric(f"Detector 3 ({detector_3_loc} ft)", f"{cum_detector_3_df['cumulative'].max()} veh")
    with col4:
        st.empty()

    # CREATE THE PLOT
    fig_det = px.line(
        cum_detector_1_df,
        x="time",
        y="cumulative",
        title="💻 Detector Cumulative Vehicle Counts",
        labels={"time": "t (seconds)", "cumulative": "N (veh)"}
    )

    # ADD DETECTOR 2 CURVE
    fig_det.add_trace(
        px.line(cum_detector_2_df, x="time", y="cumulative").data[0]
    )

    # ADD DETECTOR 3 CURVE
    fig_det.add_trace(
        px.line(cum_detector_3_df, x="time", y="cumulative").data[0]
    )

    # UPDATE TRACES FOR CLARITY
    fig_det.data[0].name = f"Detector 1 ({detector_1_loc} ft)"
    fig_det.data[0].line.color = "#0D1B2A"
    fig_det.data[1].name = f"Detector 2 ({detector_2_loc} ft)"
    fig_det.data[1].line.color = "#2A6F97"
    fig_det.data[2].name = f"Detector 3 ({detector_3_loc} ft)"
    fig_det.data[2].line.color = "#7AD0D9"

    fig_det.update_layout(
        legend=dict(title="Detectors"),
        hovermode="x unified"
    )

    st.plotly_chart(fig_det, use_container_width=True)

# 3-DETECTOR ESTIMATION AND COMPARISON SECTION
if not estimated_cum_detector_2_df.empty and not cum_detector_2_df.empty:
    st.header("3-Detector Estimation vs Actual at Detector 2")

# COMPUTE COMPARISON METRICS
merged_df = pd.merge(
    cum_detector_2_df.rename(columns={"cumulative": "actual"}),
    estimated_cum_detector_2_df.rename(columns={"cumulative": "estimated"}),
    on="time",
    how="inner"
)

if not merged_df.empty:
    # CALCULATE ABSOLUTE DIFFERENCES
    merged_df["abs_diff"] = np.abs(merged_df["actual"] - merged_df["estimated"])
    avg_abs_diff = merged_df["abs_diff"].mean()
    max_abs_diff = merged_df["abs_diff"].max()
    rmse = np.sqrt(np.mean(merged_df["abs_diff"] ** 2))

    # DISPLAY COMPARISON METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Absolute Difference", f"{avg_abs_diff:.2f} veh")
    with col2:
        st.metric("Max Absolute Difference", f"{max_abs_diff:.0f} veh")
    with col3:
        st.metric("RMSE", f"{rmse:.2f} veh")
    with col4:
        st.metric("Sample Size", f"{len(merged_df)} points")

    # CREATE COMPARISON PLOT
    fig_comp = px.line(
        merged_df,
        x="time",
        y=["actual", "estimated"],
        title="💻 3-Detector Estimation vs Actual Cumulative Counts",
        labels={"time": "t (seconds)", "value": "N (veh)", "variable": "Type"}
    )

    # UPDATE COLORS AND LEGEND
    fig_comp.data[0].name = "Actual"
    fig_comp.data[0].line.color = "#0D1B2A"
    fig_comp.data[1].name = "Estimated (3-Detector)"
    fig_comp.data[1].line.color = "#E74C3C"
    fig_comp.data[1].line.dash = "dash"

    fig_comp.update_layout(
        legend=dict(title="Cumulative Counts"),
        hovermode="x unified"
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    # ANALYSIS SECTION
    st.subheader("Estimation Analysis")

    # CALCULATE PERCENTAGE ACCURACY
    non_zero_actual = merged_df[merged_df["actual"] > 0]
    if not non_zero_actual.empty:
        mape = np.mean(np.abs(non_zero_actual["abs_diff"] / non_zero_actual["actual"])) * 100
        st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
