"""Batch analysis UI section."""

import pandas as pd
import streamlit as st
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from scipy.spatial import cKDTree
import numpy as np

from .calculations import calculate_aal
from .config import SELECTED_RP_FOR_DAMAGE_PLOT
from .reporting import generate_docx_report


def render_batch_analysis(uploaded_file, full_buildings_gdf, building_centroids, all_return_periods):
    """Render batch upload processing and report download."""
    if uploaded_file is None:
        return

    st.subheader("Batch Analysis Results")
    try:
        df_uploaded = pd.read_excel(uploaded_file)

        required_cols = ["Address", "Monetary Asset Value (million NOK)"]
        if not all(col in df_uploaded.columns for col in required_cols):
            st.error(f"Uploaded Excel must contain columns: {required_cols}")
            return

        batch_results = []
        total_portfolio_asset_value = 0.0
        geolocator = Nominatim(user_agent="bergen_wind_app_batch_query")

        with st.spinner("Processing addresses from Excel... This may take a while for many addresses."):
            for _, row in df_uploaded.iterrows():
                address = row["Address"]
                asset_value_millions = row["Monetary Asset Value (million NOK)"]
                total_portfolio_asset_value += asset_value_millions * 1_000_000

                location = None
                try:
                    location = geolocator.geocode(address, timeout=10)
                except (GeocoderTimedOut, GeocoderServiceError) as exc:
                    st.warning(f"Geocoding error for '{address}': {exc}. Skipping this address.")

                if location:
                    if building_centroids.empty:
                        st.error("No valid building data loaded for spatial matching. Please check input files and data ranges.")
                        break

                    query_coords = np.array([[location.longitude, location.latitude]])
                    _, indices = cKDTree(np.array([[p.x, p.y] for p in building_centroids])).query(query_coords)
                    nearest_building_index = indices[0]
                    selected_building_data_batch = full_buildings_gdf.iloc[nearest_building_index]

                    damage_ratio_moderate_rp100 = selected_building_data_batch.get(
                        f"damage_ratio_moderate_rp{SELECTED_RP_FOR_DAMAGE_PLOT}", np.nan
                    )

                    calculated_damage_nok = np.nan
                    if pd.notna(damage_ratio_moderate_rp100) and pd.notna(asset_value_millions):
                        calculated_damage_nok = damage_ratio_moderate_rp100 * asset_value_millions * 1_000_000

                    aal_nok = calculate_aal(
                        selected_building_data_batch,
                        "Moderate",
                        all_return_periods,
                        asset_value_millions * 1_000_000,
                    )

                    batch_results.append(
                        {
                            "Address": address,
                            "Latitude": f"{location.latitude:.4f}",
                            "Longitude": f"{location.longitude:.4f}",
                            "Asset Value": f"{asset_value_millions * 1_000_000:,.2f}",
                            "Risk Index (No Adapt)": f"{selected_building_data_batch['normalized_klawa_risk_index_no_adapt']:.2f}",
                            "Risk Index (Medium Adapt)": f"{selected_building_data_batch['normalized_klawa_risk_index_medium_adapt']:.2f}",
                            "Risk Index (High Adapt)": f"{selected_building_data_batch['normalized_klawa_risk_index_high_adapt']:.2f}",
                            f"Damage Ratio (RP {SELECTED_RP_FOR_DAMAGE_PLOT} yr, Moderate)": (
                                f"{(damage_ratio_moderate_rp100 * 100):.2f}%"
                                if pd.notna(damage_ratio_moderate_rp100)
                                else "N/A"
                            ),
                            "Calculated Damage (NOK)": (
                                f"{calculated_damage_nok:,.2f}" if pd.notna(calculated_damage_nok) else "N/A"
                            ),
                            "Average Annual Loss (NOK)": f"{aal_nok:,.2f}" if pd.notna(aal_nok) else "N/A",
                            "Map Risk Zone (No Adapt)": selected_building_data_batch["risk_zone_no_adapt"],
                        }
                    )
                else:
                    batch_results.append(
                        {
                            "Address": address,
                            "Latitude": "N/A",
                            "Longitude": "N/A",
                            "Asset Value": "N/A",
                            "Risk Index (No Adapt)": "N/A",
                            "Risk Index (Medium Adapt)": "N/A",
                            "Risk Index (High Adapt)": "N/A",
                            f"Damage Ratio (RP {SELECTED_RP_FOR_DAMAGE_PLOT} yr, Moderate)": "N/A",
                            "Calculated Damage (NOK)": "N/A",
                            "Average Annual Loss (NOK)": "N/A",
                            "Map Risk Zone (No Adapt)": "N/A",
                            "Notes": "Could not geocode address or find building",
                        }
                    )

        df_batch_results = pd.DataFrame(batch_results)
        st.dataframe(df_batch_results)

        st.markdown("---")
        st.subheader("Generate Portfolio Report")

        docx_bytes = generate_docx_report(df_batch_results, len(df_uploaded), total_portfolio_asset_value)
        st.download_button(
            label="Download DOCX Report",
            data=docx_bytes,
            file_name="Portfolio_Wind_Risk_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            help="Generates a DOCX report of the batch analysis results.",
        )

    except Exception as exc:
        st.error(f"Error reading or processing Excel file: {exc}")
