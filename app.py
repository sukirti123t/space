"""
app.py
------
Streamlit dashboard for the AI-Driven Space Debris Detection and
Collision Prediction System.

Two core workflows:
  1. Image-based debris/object detection using YOLOv8 (detect.py)
  2. Orbital conjunction (collision-risk) analysis using Skyfield (orbit.py)

Run:
    streamlit run app.py

"""

import os
import datetime as dt

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from detect import DebrisDetector, simulate_synthetic_frame
from orbit import (
    fetch_tle, analyze_conjunction, get_orbit_track,
    load_satellite_catalog, batch_conjunction_screening, TLEFetchError
)
from alerts import notify_high_risk
from nasa_api import fetch_neo_threat_data, NASAAPIError
from satellite_tracker import (
    fetch_tle_multi_source, compute_live_position,
    compute_ground_track, get_all_satellite_positions,
)

# --------------------------------------------------------------------------
# Page config + theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Space Debris Detection & Collision Prediction",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = os.path.join("assets", "style.css")
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown(
            "<style>.stApp{background:#05060f;color:#e6ecff;}</style>",
            unsafe_allow_html=True,
        )


load_css()

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🛰️ Mission Control")
page = st.sidebar.radio(
    "Navigate",
    [
        "🧠 Debris Image Detection",
        "🌌 Collision Risk Predictor",
        "📋 Batch Risk Screening",
        "🌠 NASA NEO Threat Monitor",
        "📡 Live Satellite Tracker",
    ],
    label_visibility="collapsed",
)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    "<div class='hero-title'>AI-Driven Space Debris Detection & Collision Prediction</div>"
    "<div class='hero-subtitle'>YOLOv8 vision pipeline + orbital-mechanics conjunction "
    "analysis for Indian satellite safety</div><br>",
    unsafe_allow_html=True,
)

# ==========================================================================
# PAGE 1: Debris Image Detection
# ==========================================================================
if page == "🧠 Debris Image Detection":
    st.subheader("🧠 YOLOv8 Object / Debris Detection")
    st.write(
        "Upload a telescope, radar-derived, or simulated space-imagery frame. "
        "The model will localize candidate debris objects with bounding boxes "
        "and confidence scores."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded = st.file_uploader(
            "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )
        use_synthetic = st.button("🎲 Use a simulated star-field frame instead")
        conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

    image_to_process = None
    if uploaded is not None:
        try:
            image_to_process = Image.open(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded image: {e}")
    elif use_synthetic:
        image_to_process = simulate_synthetic_frame(seed=None)

    if image_to_process is not None:
        with col_left:
            st.image(image_to_process, caption="Input frame", use_container_width=True)

        with st.spinner("Running YOLOv8 inference..."):
            try:
                detector = DebrisDetector()
                if not detector.is_ready:
                    st.error(detector.load_error)
                else:
                    if detector.using_fallback:
                        st.warning(
                            "⚠️ No custom-trained debris model found at "
                            "`models/yolov8_debris.pt`. Using a stock "
                            "COCO-pretrained YOLOv8n model for demonstration. "
                            "Detections are labeled 'debris-candidate' and are "
                            "illustrative -- train a domain-specific model for "
                            "production accuracy (see README)."
                        )
                    annotated, detections = detector.detect(
                        image_to_process, conf_threshold=conf_threshold
                    )
                    with col_right:
                        st.image(
                            annotated,
                            caption=f"Detections: {len(detections)} object(s) found",
                            use_container_width=True,
                        )
                    if detections:
                        df = pd.DataFrame(
                            [
                                {
                                    "Label": d.label,
                                    "Confidence": f"{d.confidence*100:.1f}%",
                                    "Box (x1,y1,x2,y2)": d.box_xyxy,
                                }
                                for d in detections
                            ]
                        )
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info(
                            "No objects detected above the confidence threshold. "
                            "Try lowering the slider or using the simulated frame."
                        )
            except Exception as e:
                st.error(f"Detection failed: {e}")
    else:
        with col_right:
            st.info("Upload an image or generate a simulated frame to begin.")

# ==========================================================================
# PAGE 2: Collision Risk Predictor
# ==========================================================================
elif page == "🌌 Collision Risk Predictor":
    st.subheader("🌌 Orbital Conjunction & Collision Risk Analysis")
    st.write(
        "Select a real Indian satellite and a tracked debris/rocket-body object. "
        "The system fetches live TLE data, propagates both orbits with SGP4, "
        "and estimates the closest approach within your chosen time window."
    )

    try:
        satellites = load_satellite_catalog("data/satellites.json")
    except Exception as e:
        st.error(f"Could not load satellite catalog: {e}")
        satellites = []

    try:
        debris_df = pd.read_csv("data/debris_catalog.csv")
    except Exception as e:
        st.error(f"Could not load debris catalog: {e}")
        debris_df = pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    with col1:
        sat_names = [s["name"] for s in satellites]
        sat_choice = st.selectbox("Indian Satellite", sat_names) if sat_names else None
    with col2:
        deb_names = debris_df["name"].tolist() if not debris_df.empty else []
        deb_choice = st.selectbox("Debris / Rocket Body", deb_names) if deb_names else None
    with col3:
        window_hours = st.slider("Analysis window (hours from now)", 1, 48, 6)

    run = st.button("🚀 Run Conjunction Analysis")

    if run and sat_choice and deb_choice:
        sat_info = next(s for s in satellites if s["name"] == sat_choice)
        deb_info = debris_df[debris_df["name"] == deb_choice].iloc[0]

        with st.spinner("Fetching TLEs and propagating orbits..."):
            try:
                name_a, l1_a, l2_a = fetch_tle(sat_info["norad_id"])
                name_b, l1_b, l2_b = fetch_tle(int(deb_info["norad_id"]))

                start = dt.datetime.now(dt.timezone.utc)
                end = start + dt.timedelta(hours=window_hours)

                result = analyze_conjunction(
                    sat_choice, l1_a, l2_a,
                    deb_choice, l1_b, l2_b,
                    start, end,
                    step_seconds=20,
                    size_a_m=2.0,
                    size_b_m=float(deb_info["approx_size_m"]),
                )
            except TLEFetchError as e:
                st.error(f"TLE fetch error: {e}")
                result = None
            except Exception as e:
                st.error(f"Conjunction analysis failed: {e}")
                result = None

        if result:
            st.markdown("### 📊 Conjunction Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(
                f"<div class='metric-card'><b>Miss Distance</b><br>"
                f"<span style='font-size:1.5rem'>{result.miss_distance_km:.2f} km</span></div>",
                unsafe_allow_html=True,
            )
            m2.markdown(
                f"<div class='metric-card'><b>Relative Speed</b><br>"
                f"<span style='font-size:1.5rem'>{result.relative_speed_kms:.2f} km/s</span></div>",
                unsafe_allow_html=True,
            )
            m3.markdown(
                f"<div class='metric-card'><b>Risk Level</b><br>"
                f"<span class='risk-{result.risk_level}' style='font-size:1.5rem'>"
                f"{result.risk_level}</span></div>",
                unsafe_allow_html=True,
            )
            m4.markdown(
                f"<div class='metric-card'><b>Pseudo-Probability</b><br>"
                f"<span style='font-size:1.5rem'>{result.pseudo_probability*100:.3f}%</span></div>",
                unsafe_allow_html=True,
            )

            st.caption(
                f"Time of closest approach (UTC): "
                f"{result.closest_approach_time.strftime('%Y-%m-%d %H:%M:%S')}  |  "
                f"Combined hard-body radius used: {result.combined_hard_body_radius_km*1000:.1f} m"
            )
            st.info(
                "ℹ️ The pseudo-probability is a simplified educational heuristic "
                "based on miss distance, combined object size, and relative speed. "
                "It is **not** the operational Probability of Collision (Pc) used "
                "by real conjunction assessment systems, which require full "
                "position-covariance data."
            )

            if result.risk_level == "HIGH":
                st.error("🚨 HIGH RISK conjunction detected!")
                if st.button("📣 Send Alert (Telegram / Email)"):
                    with st.spinner("Dispatching alerts..."):
                        outcomes = notify_high_risk(result)
                    for o in outcomes:
                        if o.sent:
                            st.success(f"✅ {o.channel.capitalize()}: {o.message}")
                        else:
                            st.warning(f"⚠️ {o.channel.capitalize()}: {o.message}")

            # 3D orbit visualization
            st.markdown("### 🌍 3D Orbit Visualization")
            try:
                _, track_a = get_orbit_track(name_a, l1_a, l2_a, hours=window_hours)
                _, track_b = get_orbit_track(name_b, l1_b, l2_b, hours=window_hours)

                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=track_a[:, 0], y=track_a[:, 1], z=track_a[:, 2],
                    mode="lines", name=sat_choice,
                    line=dict(color="#7cf5ff", width=4),
                ))
                fig.add_trace(go.Scatter3d(
                    x=track_b[:, 0], y=track_b[:, 1], z=track_b[:, 2],
                    mode="lines", name=deb_choice,
                    line=dict(color="#ff5470", width=4),
                ))
                # Earth reference sphere (simplified, radius 6371 km)
                u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
                ex = 6371 * np.cos(u) * np.sin(v)
                ey = 6371 * np.sin(u) * np.sin(v)
                ez = 6371 * np.cos(v)
                fig.add_trace(go.Surface(
                    x=ex, y=ey, z=ez, opacity=0.35, showscale=False,
                    colorscale=[[0, "#1c2748"], [1, "#2a1c48"]],
                    name="Earth",
                ))
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e6ecff"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=520,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render 3D orbit visualization: {e}")
    elif run:
        st.warning("Please select both a satellite and a debris object.")

# ==========================================================================
# PAGE 3: Batch Risk Screening
# ==========================================================================
elif page == "📋 Batch Risk Screening":
    st.subheader("📋 Batch Conjunction Screening")
    st.write(
        "Screen **one Indian satellite against the entire tracked debris "
        "catalog** in a single pass — the same basic concept operational "
        "conjunction-assessment teams use to triage which objects deserve "
        "closer attention."
    )

    try:
        satellites = load_satellite_catalog("data/satellites.json")
    except Exception as e:
        st.error(f"Could not load satellite catalog: {e}")
        satellites = []

    try:
        debris_df = pd.read_csv("data/debris_catalog.csv")
    except Exception as e:
        st.error(f"Could not load debris catalog: {e}")
        debris_df = pd.DataFrame()

    col1, col2 = st.columns(2)
    with col1:
        sat_names = [s["name"] for s in satellites]
        primary_choice = st.selectbox("Primary Satellite to Protect", sat_names) if sat_names else None
    with col2:
        window_hours_batch = st.slider("Screening window (hours from now)", 1, 48, 6, key="batch_hours")

    run_batch = st.button("🛰️ Run Batch Screening")

    if run_batch and primary_choice and not debris_df.empty:
        primary_info = next(s for s in satellites if s["name"] == primary_choice)

        progress_bar = st.progress(0, text="Starting batch screening...")

        def _update_progress(i, total, name):
            progress_bar.progress(i / total, text=f"Screening {name} ({i}/{total})...")

        start = dt.datetime.now(dt.timezone.utc)
        end = start + dt.timedelta(hours=window_hours_batch)

        try:
            results, errors = batch_conjunction_screening(
                primary_choice,
                primary_info["norad_id"],
                debris_df,
                start, end,
                step_seconds=30,
                progress_callback=_update_progress,
            )
            progress_bar.progress(1.0, text="Screening complete.")
        except TLEFetchError as e:
            st.error(f"Batch screening failed: {e}")
            results, errors = [], []

        if results:
            st.markdown(f"### 🏁 Ranked Risk Table — {primary_choice} vs {len(results)} objects")
            table = pd.DataFrame([
                {
                    "Debris Object": r.object_b,
                    "Miss Distance (km)": round(r.miss_distance_km, 2),
                    "Relative Speed (km/s)": round(r.relative_speed_kms, 2),
                    "Risk Level": r.risk_level,
                    "Pseudo-Probability (%)": round(r.pseudo_probability * 100, 4),
                    "Closest Approach (UTC)": r.closest_approach_time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for r in results
            ])

            def _highlight_risk(row):
                color_map = {
                    "HIGH": "background-color: rgba(255,84,112,0.25)",
                    "MEDIUM": "background-color: rgba(255,184,77,0.20)",
                    "LOW": "background-color: rgba(124,245,255,0.10)",
                    "NEGLIGIBLE": "",
                }
                return [color_map.get(row["Risk Level"], "")] * len(row)

            st.dataframe(
                table.style.apply(_highlight_risk, axis=1),
                use_container_width=True,
            )

            high_risk_results = [r for r in results if r.risk_level == "HIGH"]
            if high_risk_results:
                st.error(
                    f"🚨 {len(high_risk_results)} HIGH-risk object(s) found "
                    f"against {primary_choice}!"
                )
                if st.button("📣 Send Alert for Highest-Risk Object"):
                    with st.spinner("Dispatching alerts..."):
                        outcomes = notify_high_risk(high_risk_results[0])
                    for o in outcomes:
                        if o.sent:
                            st.success(f"✅ {o.channel.capitalize()}: {o.message}")
                        else:
                            st.warning(f"⚠️ {o.channel.capitalize()}: {o.message}")
            else:
                st.success("✅ No HIGH-risk conjunctions found in this window.")

            csv_bytes = table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Risk Report (CSV)", csv_bytes,
                file_name=f"{primary_choice.replace(' ', '_')}_risk_report.csv",
                mime="text/csv",
            )

        if errors:
            with st.expander(f"⚠️ {len(errors)} object(s) skipped (TLE/propagation errors)"):
                for name, err in errors:
                    st.write(f"- **{name}**: {err}")
    elif run_batch:
        st.warning("Please select a primary satellite and ensure the debris catalog loaded.")

# ==========================================================================
# PAGE 4: NASA NEO Threat Monitor
# ==========================================================================
elif page == "🌠 NASA NEO Threat Monitor":
    st.subheader("🌠 NASA Near Earth Object Threat Monitor")
    st.write(
        "Live feed of asteroids and Near Earth Objects approaching Earth, "
        "powered by NASA's NeoWs API. Objects are ranked by a multi-factor "
        "risk score based on size, velocity, distance, and hazard designation."
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    with col_ctrl1:
        days_ahead = st.slider("Forecast window (days)", 1, 7, 7, key="neo_days")
    with col_ctrl2:
        force_refresh = st.button("🔄 Force Refresh")
    with col_ctrl3:
        st.markdown(
            "<div class='api-banner'>"
            "<span class='status-dot live'></span>"
            "Data from <b>api.nasa.gov</b></div>",
            unsafe_allow_html=True,
        )

    with st.spinner("Fetching NEO data from NASA..."):
        feed = fetch_neo_threat_data(days_ahead=days_ahead, force_refresh=force_refresh)

    # Status banner
    if feed.api_key_type == "DEMO_DATA":
        st.warning(
            "⚠️ Using demo data — NASA API unreachable. Set `NASA_API_KEY` env var "
            "or add it to `.streamlit/secrets.toml`. Get a free key at https://api.nasa.gov"
        )
    elif feed.api_key_type == "DEMO_KEY":
        st.info("ℹ️ Using NASA DEMO_KEY (rate-limited). Get your own free key at https://api.nasa.gov")
    if feed.from_cache:
        st.caption("📦 Serving cached data (refreshes every hour)")

    # Summary metrics
    neos = feed.objects
    pha_count = sum(1 for n in neos if n.is_potentially_hazardous)
    high_risk = sum(1 for n in neos if n.risk_level in ("HIGH", "CRITICAL"))
    avg_vel = sum(n.relative_velocity_kms for n in neos) / max(len(neos), 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f"<div class='metric-card'><b>Total NEOs</b><br>"
        f"<span style='font-size:1.8rem;color:#7cf5ff'>{feed.element_count}</span></div>",
        unsafe_allow_html=True,
    )
    m2.markdown(
        f"<div class='metric-card'><b>Potentially Hazardous</b><br>"
        f"<span style='font-size:1.8rem;color:#ff5470'>{pha_count}</span></div>",
        unsafe_allow_html=True,
    )
    m3.markdown(
        f"<div class='metric-card'><b>High/Critical Risk</b><br>"
        f"<span style='font-size:1.8rem;color:#ffb84d'>{high_risk}</span></div>",
        unsafe_allow_html=True,
    )
    m4.markdown(
        f"<div class='metric-card'><b>Avg Velocity</b><br>"
        f"<span style='font-size:1.8rem;color:#b388ff'>{avg_vel:.1f} km/s</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if neos:
        # Scatter: distance vs velocity, sized by diameter
        st.markdown("### 📊 NEO Threat Landscape")
        scatter_df = pd.DataFrame([{
            "Name": n.name,
            "Miss Distance (LD)": round(n.miss_distance_lunar, 2),
            "Velocity (km/s)": round(n.relative_velocity_kms, 2),
            "Avg Diameter (m)": round((n.estimated_diameter_min_m + n.estimated_diameter_max_m) / 2, 1),
            "Risk": n.risk_level,
            "PHA": "Yes" if n.is_potentially_hazardous else "No",
            "Score": n.risk_score,
        } for n in neos])

        color_map = {"CRITICAL": "#ff2d55", "HIGH": "#ff5470", "MEDIUM": "#ffb84d", "LOW": "#7cf5ff", "NEGLIGIBLE": "#6dff9e"}
        fig_scatter = px.scatter(
            scatter_df, x="Miss Distance (LD)", y="Velocity (km/s)",
            size="Avg Diameter (m)", color="Risk", hover_name="Name",
            color_discrete_map=color_map, size_max=40,
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,18,36,0.5)",
            font=dict(color="#e6ecff"), height=420,
            xaxis=dict(gridcolor="rgba(124,245,255,0.1)"),
            yaxis=dict(gridcolor="rgba(124,245,255,0.1)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Detailed table
        st.markdown("### 🗂️ NEO Detail Table")
        table_df = pd.DataFrame([{
            "Name": n.name,
            "Approach Date": n.close_approach_date,
            "Diameter (m)": f"{n.estimated_diameter_min_m:.0f}–{n.estimated_diameter_max_m:.0f}",
            "Distance (LD)": f"{n.miss_distance_lunar:.2f}",
            "Distance (km)": f"{n.miss_distance_km:,.0f}",
            "Velocity (km/s)": f"{n.relative_velocity_kms:.2f}",
            "PHA": "⚠️ Yes" if n.is_potentially_hazardous else "No",
            "Risk": n.risk_level,
            "Score": n.risk_score,
        } for n in neos])

        def _neo_highlight(row):
            cmap = {"CRITICAL": "background-color:rgba(255,45,85,0.2)", "HIGH": "background-color:rgba(255,84,112,0.15)",
                    "MEDIUM": "background-color:rgba(255,184,77,0.12)", "LOW": "", "NEGLIGIBLE": ""}
            return [cmap.get(row["Risk"], "")] * len(row)

        st.dataframe(table_df.style.apply(_neo_highlight, axis=1), use_container_width=True, height=400)

        csv = table_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download NEO Report (CSV)", csv, "neo_threat_report.csv", "text/csv")
    else:
        st.info("No NEO data available for the selected window.")

# ==========================================================================
# PAGE 5: Live Satellite Tracker
# ==========================================================================
elif page == "📡 Live Satellite Tracker":
    st.subheader("📡 Real-Time Indian Satellite Tracker")
    st.write(
        "Track live positions of ISRO satellites using TLE data propagated "
        "with Skyfield's SGP4 engine. Select a satellite to see its current "
        "coordinates, ground track, and 3D orbit."
    )

    try:
        satellites = load_satellite_catalog("data/satellites.json")
    except Exception as e:
        st.error(f"Could not load satellite catalog: {e}")
        satellites = []

    if satellites:
        # Satellite selector
        sat_names = [s["name"] for s in satellites]
        selected_sats = st.multiselect(
            "Select satellites to track", sat_names, default=sat_names[:3]
        )

        if st.button("📍 Compute Live Positions") or selected_sats:
            positions = []
            tracks = []
            for sname in selected_sats:
                sinfo = next(s for s in satellites if s["name"] == sname)
                try:
                    name, l1, l2 = fetch_tle_multi_source(sinfo["norad_id"])
                    pos = compute_live_position(
                        name=sinfo["name"], norad_id=sinfo["norad_id"],
                        line1=l1, line2=l2, satellite_type=sinfo.get("type", ""),
                    )
                    positions.append(pos)
                    track = compute_ground_track(
                        name=sinfo["name"], norad_id=sinfo["norad_id"],
                        line1=l1, line2=l2, duration_minutes=90, step_seconds=60,
                    )
                    tracks.append(track)
                except Exception as e:
                    st.warning(f"⚠️ Could not track {sname}: {e}")

            if positions:
                # Live position cards
                st.markdown("### 📊 Current Positions")
                cols = st.columns(min(len(positions), 3))
                for i, pos in enumerate(positions):
                    with cols[i % 3]:
                        st.markdown(
                            f"<div class='tracker-card'>"
                            f"<b style='color:#b388ff;font-size:1.1rem'>{pos.name}</b><br>"
                            f"<span class='neo-stat'>🌐 {pos.latitude_deg:.2f}°, {pos.longitude_deg:.2f}°</span>"
                            f"<span class='neo-stat'>📏 {pos.altitude_km:.1f} km</span>"
                            f"<span class='neo-stat'>🚀 {pos.velocity_kms:.2f} km/s</span><br>"
                            f"<span style='color:#6c7599;font-size:0.8rem'>{pos.timestamp_utc}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # Ground track map
                if tracks:
                    st.markdown("### 🗺️ Ground Track (Next 90 min)")
                    fig_map = go.Figure()
                    colors = ["#7cf5ff", "#b388ff", "#ff8fd6", "#6dff9e", "#ffb84d",
                              "#ff5470", "#64b5f6", "#81c784", "#fff176", "#f48fb1"]
                    for idx, track in enumerate(tracks):
                        color = colors[idx % len(colors)]
                        fig_map.add_trace(go.Scattergeo(
                            lon=track.longitudes, lat=track.latitudes,
                            mode="lines+markers",
                            name=track.name,
                            line=dict(color=color, width=2),
                            marker=dict(size=3),
                        ))
                        # Mark current position
                        if positions[idx]:
                            fig_map.add_trace(go.Scattergeo(
                                lon=[positions[idx].longitude_deg],
                                lat=[positions[idx].latitude_deg],
                                mode="markers+text",
                                name=f"{track.name} (now)",
                                marker=dict(size=10, color=color, symbol="star"),
                                text=[track.name], textposition="top center",
                                textfont=dict(color=color, size=10),
                                showlegend=False,
                            ))
                    fig_map.update_geos(
                        showland=True, landcolor="rgb(15,20,40)",
                        showocean=True, oceancolor="rgb(8,12,28)",
                        showcoastlines=True, coastlinecolor="rgba(124,245,255,0.3)",
                        showframe=False, bgcolor="rgba(0,0,0,0)",
                    )
                    fig_map.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", height=450,
                        font=dict(color="#e6ecff"),
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                        margin=dict(l=0, r=0, t=0, b=0),
                    )
                    st.plotly_chart(fig_map, use_container_width=True)

                # 3D orbit viz
                if tracks:
                    st.markdown("### 🌍 3D Orbital View")
                    fig3d = go.Figure()
                    for idx, sname in enumerate(selected_sats):
                        sinfo = next(s for s in satellites if s["name"] == sname)
                        try:
                            nm, l1, l2 = fetch_tle_multi_source(sinfo["norad_id"])
                            _, trk = get_orbit_track(nm, l1, l2, hours=1.5)
                            color = colors[idx % len(colors)]
                            fig3d.add_trace(go.Scatter3d(
                                x=trk[:, 0], y=trk[:, 1], z=trk[:, 2],
                                mode="lines", name=sname,
                                line=dict(color=color, width=3),
                            ))
                        except Exception:
                            pass
                    # Earth sphere
                    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
                    fig3d.add_trace(go.Surface(
                        x=6371*np.cos(u)*np.sin(v),
                        y=6371*np.sin(u)*np.sin(v),
                        z=6371*np.cos(v),
                        opacity=0.35, showscale=False,
                        colorscale=[[0,"#1c2748"],[1,"#2a1c48"]], name="Earth",
                    ))
                    fig3d.update_layout(
                        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"),
                        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6ecff"),
                        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=0,b=0), height=520,
                    )
                    st.plotly_chart(fig3d, use_container_width=True)

