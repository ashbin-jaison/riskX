"""Data loading and preprocessing pipeline."""

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from scipy.spatial import cKDTree

import osmnx as ox
from shapely.geometry import box

from .calculations import get_risk_color_rgba, vulnerability_curve
from .config import (
    ORANGE_ZONE_THRESHOLD_PERC,
    RED_ZONE_THRESHOLD_PERC,
    SELECTED_RP_FOR_DAMAGE_PLOT,
    YELLOW_ZONE_THRESHOLD_PERC,
)


@st.cache_data
def load_and_process_data(wind_data_path, klawa_data_sources_paths, vuln_params, simplify_tolerance):
    """Load, spatially join, and derive wind/damage/risk fields for buildings."""
    try:
        wind_data = xr.open_dataset(wind_data_path)
    except FileNotFoundError:
        st.error(f"Error: Wind data file not found at {wind_data_path}. Please check the path.")
        st.stop()
    except Exception as exc:
        st.error(f"Error loading wind data: {exc}")
        st.stop()

    wind_lons = wind_data.x.values
    wind_lats = wind_data.y.values

    min_lon, max_lon = wind_lons.min(), wind_lons.max()
    min_lat, max_lat = wind_lats.min(), wind_lats.max()

    north, south, east, west = max_lat, min_lat, max_lon, min_lon

    klawa_and_wind_grids_raw_xarray = {}
    klawa_lons = None
    klawa_lats = None

    for key, info in klawa_data_sources_paths.items():
        try:
            temp_klawa_ds = xr.open_dataset(info["path"])
            klawa_and_wind_grids_raw_xarray[key] = {
                "klawa_risk_index": temp_klawa_ds["klawa_risk_index"].squeeze(),
                "wind_percentile_speed": temp_klawa_ds[info["wind_var_name"]].squeeze(),
            }
            if klawa_lons is None:
                klawa_lons = temp_klawa_ds.x.values
                klawa_lats = temp_klawa_ds.y.values
        except FileNotFoundError:
            st.error(
                f"Error: Klawa risk index or wind percentile file not found at {info['path']}. Please check the path."
            )
            st.stop()
        except KeyError as exc:
            st.error(f"Error: Variable '{exc}' not found in {info['path']}. Please confirm variable names.")
            st.stop()
        except Exception as exc:
            st.error(f"Error loading Klawa/Wind data for '{key}': {exc}")
            st.stop()

    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
    tags = {
        "building": [
            "residential",
            "house",
            "apartments",
            "flats",
            "detached",
            "semidetached_house",
            "terrace",
            "bungalow",
            "farm",
            "shed",
            "cabin",
        ]
    }
    try:
        buildings = ox.features_from_polygon(bbox_polygon, tags)
        buildings = buildings[buildings.geometry.type == "Polygon"].copy()
        buildings = buildings.to_crs(epsg=4326)
    except Exception as exc:
        st.error(f"Error loading OSM data: {exc}. Please ensure OSMnx is updated and working correctly.")
        st.stop()

    if buildings.empty:
        st.warning("No residential buildings found. Check OSM tags or the derived bounding box.")
        return None, None, None, None, None, None, None

    buildings["geometry"] = buildings.geometry.simplify(simplify_tolerance, preserve_topology=True)
    buildings = buildings[~buildings.geometry.is_empty].copy()

    if buildings.empty:
        st.warning(
            "All residential buildings were filtered out or simplified to empty geometries. Cannot proceed with analysis."
        )
        return None, None, None, None, None, None, None

    building_centroids = buildings.geometry.centroid
    valid_centroids_mask = building_centroids.apply(
        lambda p: p is not None and p.x is not None and p.y is not None and np.isfinite(p.x) and np.isfinite(p.y)
    )
    buildings = buildings[valid_centroids_mask].copy()
    building_centroids = building_centroids[valid_centroids_mask]

    if buildings.empty:
        st.warning("No valid building centroids found after filtering. Cannot proceed with spatial analysis.")
        return None, None, None, None, None, None, None

    building_coords = np.array([[p.x, p.y] for p in building_centroids])

    all_return_periods = wind_data.return_period.values
    display_return_periods = [rp for rp in all_return_periods if rp != 1]

    wind_lon_mesh, wind_lat_mesh = np.meshgrid(wind_lons, wind_lats)
    wind_points = np.column_stack([wind_lon_mesh.flatten(), wind_lat_mesh.flatten()])
    wind_tree = cKDTree(wind_points)

    _, indices = wind_tree.query(building_coords)
    wind_grid_shape = (len(wind_lats), len(wind_lons))
    wind_grid_rows, wind_grid_cols = np.unravel_index(indices, wind_grid_shape)

    for rp in all_return_periods:
        buildings[f"wind_speed_rp{rp}"] = np.nan
        current_wind_speeds_data = wind_data["ReturnPeriodWindSpeed"].sel(return_period=rp).squeeze()

        buildings.loc[buildings.index, f"wind_speed_rp{rp}"] = current_wind_speeds_data.isel(
            y=xr.DataArray(wind_grid_rows, dims="temp_dim"),
            x=xr.DataArray(wind_grid_cols, dims="temp_dim"),
        ).values

    for b_type, params in vuln_params.items():
        k_val = params["k"]
        v0_val = params["v0"]
        for rp in all_return_periods:
            buildings[f"damage_ratio_{b_type.lower()}_rp{rp}"] = buildings[f"wind_speed_rp{rp}"].apply(
                lambda x: vulnerability_curve(x, k_val, v0_val)
            )

    klawa_lon_mesh, klawa_lat_mesh = np.meshgrid(klawa_lons, klawa_lats)
    klawa_points = np.column_stack([klawa_lon_mesh.flatten(), klawa_lat_mesh.flatten()])
    klawa_tree = cKDTree(klawa_points)

    _, indices_klawa = klawa_tree.query(building_coords)
    klawa_grid_shape = (len(klawa_lats), len(klawa_lons))
    klawa_grid_rows, klawa_grid_cols = np.unravel_index(indices_klawa, klawa_grid_shape)

    all_raw_klawa_values_for_global_min_max = []

    for key in klawa_data_sources_paths.keys():
        klawa_grid_da = klawa_and_wind_grids_raw_xarray[key]["klawa_risk_index"]
        wind_percentile_da = klawa_and_wind_grids_raw_xarray[key]["wind_percentile_speed"]

        raw_klawa_col_name = f"klawa_risk_index_raw_{key}"
        raw_wind_percentile_col_name = f"wind_speed_percentile_{key}"

        buildings.loc[buildings.index, raw_klawa_col_name] = klawa_grid_da.isel(
            y=xr.DataArray(klawa_grid_rows, dims="temp_dim"),
            x=xr.DataArray(klawa_grid_cols, dims="temp_dim"),
        ).values
        buildings.loc[buildings.index, raw_wind_percentile_col_name] = wind_percentile_da.isel(
            y=xr.DataArray(klawa_grid_rows, dims="temp_dim"),
            x=xr.DataArray(klawa_grid_cols, dims="temp_dim"),
        ).values

        all_raw_klawa_values_for_global_min_max.extend(buildings[raw_klawa_col_name].dropna().tolist())

    if not all_raw_klawa_values_for_global_min_max:
        global_min_klawa = 0.0
        global_max_klawa = 1.0
        st.warning("No valid Klawa risk index values found to calculate global min/max. Using default range (0-1).")
    else:
        global_min_klawa = np.nanmin(all_raw_klawa_values_for_global_min_max)
        global_max_klawa = np.nanmax(all_raw_klawa_values_for_global_min_max)

    if global_max_klawa == global_min_klawa:
        global_max_klawa += 1e-6

    for key in klawa_data_sources_paths.keys():
        raw_klawa_col_name = f"klawa_risk_index_raw_{key}"
        norm_klawa_col_name = f"normalized_klawa_risk_index_{key}"
        risk_zone_col_name = f"risk_zone_{key}"
        risk_color_col_name = f"risk_color_rgba_{key}"

        valid_raw_klawa_values = buildings[raw_klawa_col_name].dropna()
        normalized_klawa_values = ((valid_raw_klawa_values - global_min_klawa) / (global_max_klawa - global_min_klawa)) * 100
        normalized_klawa_values = np.clip(normalized_klawa_values, 0, 100)

        buildings[norm_klawa_col_name] = np.nan
        buildings.loc[valid_raw_klawa_values.index, norm_klawa_col_name] = normalized_klawa_values

        buildings[risk_zone_col_name] = "Green"
        valid_normalized_indices = buildings[norm_klawa_col_name].dropna().index

        buildings.loc[
            (buildings.index.isin(valid_normalized_indices))
            & (buildings[norm_klawa_col_name] >= YELLOW_ZONE_THRESHOLD_PERC)
            & (buildings[norm_klawa_col_name] < ORANGE_ZONE_THRESHOLD_PERC),
            risk_zone_col_name,
        ] = "Yellow"
        buildings.loc[
            (buildings.index.isin(valid_normalized_indices))
            & (buildings[norm_klawa_col_name] >= ORANGE_ZONE_THRESHOLD_PERC)
            & (buildings[norm_klawa_col_name] < RED_ZONE_THRESHOLD_PERC),
            risk_zone_col_name,
        ] = "Orange"
        buildings.loc[
            (buildings.index.isin(valid_normalized_indices)) & (buildings[norm_klawa_col_name] >= RED_ZONE_THRESHOLD_PERC),
            risk_zone_col_name,
        ] = "Red"

        buildings[risk_color_col_name] = buildings[norm_klawa_col_name].apply(
            lambda x: get_risk_color_rgba(x) if pd.notna(x) else [128, 128, 128, 0]
        )

    if not buildings.empty:
        minx, miny, maxx, maxy = buildings.total_bounds
        center_lat = (miny + maxy) / 2
        center_lon = (minx + maxx) / 2
    else:
        center_lat = (north + south) / 2
        center_lon = (east + west) / 2

    buildings["tooltip"] = buildings.apply(
        lambda row: f"<b>Building ID:</b> {row.name}<br>"
        f"<b>Damage Ratio (RP {SELECTED_RP_FOR_DAMAGE_PLOT} yr, Moderate):</b> {(row[f'damage_ratio_moderate_rp{SELECTED_RP_FOR_DAMAGE_PLOT}'] * 100):.2f}%<br>"
        f"<b>Risk Index (No Adapt):</b> {row['normalized_klawa_risk_index_no_adapt']:.2f}<br>"
        f"<b>Risk Index (Medium Adapt):</b> {row['normalized_klawa_risk_index_medium_adapt']:.2f}<br>"
        f"<b>Risk Index (High Adapt):</b> {row['normalized_klawa_risk_index_high_adapt']:.2f}<br>"
        f"<b>Map Risk Zone (No Adapt):</b> {row['risk_zone_no_adapt']}<br>"
        f"<b>Map Risk Zone (Medium Adapt):</b> {row['risk_zone_medium_adapt']}<br>"
        f"<b>Map Risk Zone (High Adapt):</b> {row['risk_zone_high_adapt']}<br>"
        f"<b>98th Percentile Wind Speed:</b> {row['wind_speed_percentile_no_adapt']:.2f} m/s<br>"
        f"<b>99th Percentile Wind Speed:</b> {row['wind_speed_percentile_medium_adapt']:.2f} m/s<br>"
        f"<b>99.5th Percentile Wind Speed:</b> {row['wind_speed_percentile_high_adapt']:.2f} m/s",
        axis=1,
    )

    pydeck_buildings_gdf = buildings[
        [
            "geometry",
            "risk_color_rgba_no_adapt",
            "risk_color_rgba_medium_adapt",
            "risk_color_rgba_high_adapt",
            "tooltip",
        ]
    ].copy()

    return (
        pydeck_buildings_gdf,
        buildings,
        center_lat,
        center_lon,
        display_return_periods,
        building_centroids,
        all_return_periods,
    )
