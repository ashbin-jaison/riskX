"""Main Streamlit application composition."""

import pandas as pd
import pydeck as pdk
import streamlit as st

from .config import (
    GEOMETRY_SIMPLIFICATION_TOLERANCE,
    KLAWA_DATA_SOURCES_PATHS,
    LOGO_PATH_1,
    MAP_LEVEL_TO_SUFFIX,
    ORANGE_ZONE_THRESHOLD_PERC,
    RED_ZONE_THRESHOLD_PERC,
    VULNERABILITY_PARAMETERS,
    WIND_DATA_PATH,
    YELLOW_ZONE_THRESHOLD_PERC,
)
from .data_pipeline import load_and_process_data
from .ui_batch import render_batch_analysis
from .ui_single import (
    initialize_session_state,
    render_damage_table,
    render_risk_odometer,
    resolve_selected_building,
)


def run_app():
    """Run the WindRisk Streamlit app."""
    st.set_page_config(
        page_title="Bergen Wind Risk Map",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    (
        pydeck_buildings_gdf,
        full_buildings_gdf,
        center_lat,
        center_lon,
        display_return_periods,
        building_centroids,
        all_return_periods,
    ) = load_and_process_data(
        WIND_DATA_PATH,
        KLAWA_DATA_SOURCES_PATHS,
        VULNERABILITY_PARAMETERS,
        GEOMETRY_SIMPLIFICATION_TOLERANCE,
    )

    if pydeck_buildings_gdf is None:
        st.error("Data processing failed or no buildings found. Please check logs for details.")
        st.stop()

    try:
        st.sidebar.image(LOGO_PATH_1, use_container_width=True)
    except FileNotFoundError:
        st.sidebar.error("Logo 1 image file not found. Please check the path in the script.")
    except Exception as exc:
        st.sidebar.error(f"Error loading Logo 1: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.title("Select Adaptation Level for Map Display")
    map_adaptation_level = st.sidebar.radio(" ", ("No Adaptation", "Medium Adaptation", "High Adaptation"), index=0)

    color_column_for_map = f"risk_color_rgba_{MAP_LEVEL_TO_SUFFIX[map_adaptation_level]}"

    st.sidebar.markdown("---")
    st.sidebar.header("Map Risk Zone Definition")
    st.sidebar.write(f"**Red Zone:** Risk Index >= {RED_ZONE_THRESHOLD_PERC}")
    st.sidebar.write(f"**Orange Zone:** Risk Index {ORANGE_ZONE_THRESHOLD_PERC} to < {RED_ZONE_THRESHOLD_PERC}")
    st.sidebar.write(f"**Yellow Zone:** Risk Index {YELLOW_ZONE_THRESHOLD_PERC} to < {ORANGE_ZONE_THRESHOLD_PERC}")
    st.sidebar.write(f"**Green Zone:** Risk Index < {YELLOW_ZONE_THRESHOLD_PERC}")

    initialize_session_state()

    input_col_1, input_col_2 = st.columns([0.5, 0.5])
    with input_col_1:
        st.header("WindRisk: Asset level risk analysis")
        st.subheader("Analyze a Single Address")
        address_query = st.text_input("Enter an address in Bergen:", key="address_input_top_col")

        st.session_state.asset_value = st.number_input(
            "Monetary Asset Value in million Norwegian Krone (NOK, Optional):",
            min_value=0.0,
            value=st.session_state.asset_value,
            format="%.2f",
            key="asset_value_input_top",
            help="Enter the monetary value of the asset (in millions of NOK) to calculate potential damages.",
        )
        if st.session_state.asset_value is not None and st.session_state.asset_value == 0.0:
            st.session_state.asset_value = None

    with input_col_2:
        st.header(" ")
        st.subheader("Upload a File for Batch Analysis")
        uploaded_file = st.file_uploader(
            "Upload an Excel file (columns: 'Address', 'Monetary Asset Value (million NOK)')",
            type=["xlsx"],
        )

    resolve_selected_building(address_query, building_centroids, full_buildings_gdf)

    if st.session_state.selected_building_id is not None and not uploaded_file:
        selected_building_data = full_buildings_gdf.loc[st.session_state.selected_building_id]
        render_risk_odometer(selected_building_data)
    else:
        selected_building_data = None

    render_batch_analysis(uploaded_file, full_buildings_gdf, building_centroids, all_return_periods)

    st.markdown("---")
    map_col, damage_table_col = st.columns([0.6, 0.4])

    with map_col:
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)
        layer = pdk.Layer(
            "GeoJsonLayer",
            pydeck_buildings_gdf,
            pickable=True,
            auto_highlight=True,
            get_fill_color=color_column_for_map,
            get_line_color=[0, 0, 0, 80],
            get_line_width=2,
            line_width_min_pixels=1,
        )

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "{tooltip}", "html": "{tooltip}"},
        )
        st.pydeck_chart(deck, height=450)

        st.subheader(f"Map Risk Zone Legend (Risk Index - {map_adaptation_level})")
        colors = ["#D4EDDA", "#FFD700", "#FB6A4A", "#DE2D26"]
        labels = ["Low Risk", "Medium-Low Risk", "Medium-High Risk", "High Risk"]
        legend_items_html = "".join(
            [
                (
                    f'<div style="background-color: {color}; width: 60px; height: 20px; '
                    'border-radius: 4px; display: inline-block;"></div>'
                    f'<span style="font-size: 0.9em; margin-right: 10px;">{label}</span>'
                )
                for color, label in zip(colors, labels)
            ]
        )
        st.markdown(
            f"""
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;">
                {legend_items_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with damage_table_col:
        if st.session_state.selected_building_id is not None and not uploaded_file:
            render_damage_table(selected_building_data, display_return_periods, all_return_periods)
        elif uploaded_file is None:
            st.info("Enter an address to view damage ratio analysis.")

    st.markdown("---")
    st.caption("Powered by Streamlit, Pydeck, Xarray, OSMnx, Geopy, and Matplotlib.")
