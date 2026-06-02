"""Single-address analysis UI helpers."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree

from .calculations import calculate_aal, geocode_address, get_gauge_color_hex
from .config import MAP_LEVEL_TO_SUFFIX, SELECTED_RP_FOR_DAMAGE_PLOT


def initialize_session_state():
    """Initialize mutable UI state once."""
    if "selected_building_id" not in st.session_state:
        st.session_state.selected_building_id = None
    if "scenario_type" not in st.session_state:
        st.session_state.scenario_type = "Moderate"
    if "asset_value" not in st.session_state:
        st.session_state.asset_value = None


def resolve_selected_building(address_query, building_centroids, full_buildings_gdf):
    """Update selected building id from address query."""
    if not address_query:
        return

    location = geocode_address(address_query, user_agent="bergen_wind_app_user_query")
    if not location:
        st.error("Could not geocode the address or find a nearby building. Please try a more specific address in Bergen.")
        st.session_state.selected_building_id = None
        return

    query_coords = np.array([[location.longitude, location.latitude]])
    _, indices = cKDTree(np.array([[p.x, p.y] for p in building_centroids])).query(query_coords)

    nearest_building_index = indices[0]
    selected_building_gdf_row = full_buildings_gdf.iloc[nearest_building_index]
    st.session_state.selected_building_id = selected_building_gdf_row.name


def render_risk_odometer(selected_building_data):
    """Render adaptation-level risk gauges for a selected building."""
    st.subheader("Risk Index & Wind Speed by Adaptation Level for Selected Building")
    levels = ["No Adaptation", "Medium Adaptation", "High Adaptation"]
    odometer_cols = st.columns(3)

    for idx, level_name in enumerate(levels):
        with odometer_cols[idx]:
            suffix = MAP_LEVEL_TO_SUFFIX[level_name]
            klawa_index_norm = selected_building_data[f"normalized_klawa_risk_index_{suffix}"]

            st.markdown(f"**{level_name}**")
            if pd.isna(klawa_index_norm):
                st.markdown("<div style='text-align: center; color: grey;'>No Data</div>", unsafe_allow_html=True)
                continue

            gauge_color = get_gauge_color_hex(klawa_index_norm)
            unique_class_suffix = level_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            container_class = f"gauge-container-{unique_class_suffix}"
            fill_class = f"gauge-fill-{unique_class_suffix}"
            value_label_class = f"gauge-value-label-{unique_class_suffix}"

            st.markdown(
                f"""
                <style>
                .{container_class}{{
                    width: 100%;
                    height: 25px;
                    background-color: #eee;
                    border-radius: 12px;
                    overflow: hidden;
                    position: relative;
                    margin-bottom: 5px;
                    box-shadow: inset 0 0 3px rgba(0,0,0,0.1);
                }}
                .{fill_class} {{
                    height: 100%;
                    border-radius: 12px;
                    background-color: {gauge_color};
                    width: {klawa_index_norm:.2f}%;
                    transition: width 0.5s ease-in-out;
                }}
                .{value_label_class} {{
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    font-weight: bold;
                    color: black;
                    text-shadow: 1px 1px 1px rgba(255,255,255,0.7);
                    font-size: 0.9em;
                }}
                </style>
                <div class="{container_class}">
                    <div class="{fill_class}"></div>
                    <div class="{value_label_class}">{klawa_index_norm:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_damage_table(selected_building_data, display_return_periods, all_return_periods):
    """Render scenario buttons, damage table, and optional AAL metric."""
    st.subheader("Damage Ratio")
    st.write("Select Building Resilience:")

    resilience_cols_buttons = st.columns(3)
    with resilience_cols_buttons[0]:
        if st.button("Weak", key="res_weak", help="Assume weak building resilience"):
            st.session_state.scenario_type = "Weak"
    with resilience_cols_buttons[1]:
        if st.button("Moderate", key="res_moderate", help="Assume moderate building resilience"):
            st.session_state.scenario_type = "Moderate"
    with resilience_cols_buttons[2]:
        if st.button("Strong", key="res_strong", help="Assume strong building resilience"):
            st.session_state.scenario_type = "Strong"

    current_scenario_type = st.session_state.scenario_type
    st.write(f"--- Results for **{current_scenario_type}** ---")

    scenario_results = []
    for rp in display_return_periods:
        wind_speed = selected_building_data[f"wind_speed_rp{rp}"]
        damage_ratio_col_name = f"damage_ratio_{current_scenario_type.lower()}_rp{rp}"
        damage_ratio = selected_building_data[damage_ratio_col_name]

        row_data = {
            "Return Period (years)": rp,
            "Wind Speed (m/s)": f"{wind_speed:.2f}",
            "Damage Ratio (%)": f"{(damage_ratio * 100):.2f}%" if pd.notna(damage_ratio) else "N/A",
        }

        if st.session_state.asset_value is not None:
            calculated_damage = damage_ratio * st.session_state.asset_value * 1_000_000 if pd.notna(damage_ratio) else np.nan
            row_data["Calculated Damage (NOK)"] = f"{calculated_damage:,.2f}" if pd.notna(calculated_damage) else "N/A"

        scenario_results.append(row_data)

    st.markdown(pd.DataFrame(scenario_results).to_html(index=False), unsafe_allow_html=True)

    if st.session_state.asset_value is None:
        return

    aal_for_display = calculate_aal(
        selected_building_data,
        current_scenario_type,
        all_return_periods,
        st.session_state.asset_value * 1_000_000,
    )
    st.markdown("---")
    if pd.notna(aal_for_display):
        st.metric(
            label=f"Average Annual Loss (AAL) for {current_scenario_type} Resilience",
            value=f"{aal_for_display:,.2f} NOK",
        )
    else:
        st.info("AAL not calculated. Ensure asset value is provided and damage data is available.")
