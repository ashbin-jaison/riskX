"""Core risk calculations and geocoding helpers."""

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from .config import (
    ORANGE_ZONE_THRESHOLD_PERC,
    RED_ZONE_THRESHOLD_PERC,
    YELLOW_ZONE_THRESHOLD_PERC,
)


@st.cache_data
def vulnerability_curve(wind_speed, k_val, v0_val):
    """Sigmoid vulnerability curve: D(U10) = 1 / (1 + exp(-k * (U10 - v0)))."""
    if not np.isfinite(wind_speed):
        return 0.0
    return 1.0 / (1 + np.exp(-k_val * (wind_speed - v0_val)))


def get_risk_color_rgba(klawa_index_norm):
    """Convert normalized Klawa index to RGBA by threshold."""
    if klawa_index_norm >= RED_ZONE_THRESHOLD_PERC:
        hex_color = "#DE2D26"
    elif klawa_index_norm >= ORANGE_ZONE_THRESHOLD_PERC:
        hex_color = "#FB6A4A"
    elif klawa_index_norm >= YELLOW_ZONE_THRESHOLD_PERC:
        hex_color = "#FFD700"
    else:
        hex_color = "#D4EDDA"
    rgba_color = mcolors.to_rgba(hex_color)
    return [int(c * 255) for c in rgba_color]


def get_gauge_color_hex(klawa_index_norm):
    """Get gauge color by normalized Klawa index."""
    if klawa_index_norm >= RED_ZONE_THRESHOLD_PERC:
        return "#DE2D26"
    if klawa_index_norm >= ORANGE_ZONE_THRESHOLD_PERC:
        return "#FB6A4A"
    if klawa_index_norm >= YELLOW_ZONE_THRESHOLD_PERC:
        return "#FFD700"
    return "#D4EDDA"


@st.cache_data
def geocode_address(address, user_agent="bergen_wind_app"):
    """Geocode an address with Nominatim."""
    geolocator = Nominatim(user_agent=user_agent)
    try:
        location = geolocator.geocode(address, timeout=10)
        return location
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        st.error(f"Geocoding error: {exc}. Please try again or refine the address.")
        return None


def calculate_aal(building_data, resilience_type, all_available_return_periods, asset_value_nok):
    """Calculate Average Annual Loss using trapezoidal integration."""
    if asset_value_nok is None or pd.isna(asset_value_nok):
        return np.nan

    rp_damage_pairs = []
    for rp in all_available_return_periods:
        col_name = f"damage_ratio_{resilience_type.lower()}_rp{rp}"
        damage_ratio = building_data.get(col_name)
        if pd.notna(damage_ratio):
            rp_damage_pairs.append((rp, damage_ratio))

    if not rp_damage_pairs:
        return np.nan

    rp_damage_pairs.sort(key=lambda x: x[0])
    sorted_rps = [item[0] for item in rp_damage_pairs]
    sorted_damage_ratios = [item[1] for item in rp_damage_pairs]

    probabilities = [1.0] + [1.0 / rp for rp in sorted_rps] + [0.0]
    damages = [sorted_damage_ratios[0] if sorted_damage_ratios else 0.0] + sorted_damage_ratios + [0.0]

    aal_value = 0.0
    for idx in range(len(probabilities) - 1):
        p1 = probabilities[idx]
        p2 = probabilities[idx + 1]
        d1 = damages[idx]
        d2 = damages[idx + 1]

        if p1 >= p2:
            aal_value += 0.5 * (d1 + d2) * (p1 - p2)
        else:
            st.warning(
                f"Non-monotonic probability detected in AAL calculation: {p1} -> {p2}. Segment skipped."
            )

    return aal_value * asset_value_nok
