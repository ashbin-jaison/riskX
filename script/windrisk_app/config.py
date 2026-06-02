"""Application configuration and static constants."""

# IMPORTANT: Adjust these paths to your actual NetCDF file locations.
WIND_DATA_PATH = "data/bergen_return_period_winds_1985_2020_debugged.nc"
KLAWA_BASE_DATA_PATH = "data/klawa_risk_index.nc"
KLAWA_V99_DATA_PATH = "data/klawa_risk_index_v99.nc"
KLAWA_V995_DATA_PATH = "data/klawa_risk_index_v995.nc"

# Logo file paths.
LOGO_PATH_1 = "data/cl_logo_tp.png"

# Vulnerability curve parameters for different building types:
# D(U10) = 1 / (1 + exp(-k * (U10 - v0))).
VULNERABILITY_PARAMETERS = {
    "Weak": {"k": 0.4894, "v0": 29.9949},
    "Moderate": {"k": 0.2887, "v0": 41.3318},
    "Strong": {"k": 0.1821, "v0": 53.9558},
}

# Risk zone percentile thresholds applied to normalized Klawa Risk Index (0-100).
RED_ZONE_THRESHOLD_PERC = 90
ORANGE_ZONE_THRESHOLD_PERC = 70
YELLOW_ZONE_THRESHOLD_PERC = 30

# The return period selected for displaying damage ratio in tooltips.
SELECTED_RP_FOR_DAMAGE_PLOT = 100

# Geometry simplification tolerance (in degrees).
GEOMETRY_SIMPLIFICATION_TOLERANCE = 0.00001

# Mapping for adaptation level display name to internal column suffix.
MAP_LEVEL_TO_SUFFIX = {
    "No Adaptation": "no_adapt",
    "Medium Adaptation": "medium_adapt",
    "High Adaptation": "high_adapt",
}

# Mapping for adaptation level display name to corresponding wind percentile variable
# name in NC files.
MAP_LEVEL_TO_WIND_VAR = {
    "No Adaptation": "wind_98th_percentile_speed",
    "Medium Adaptation": "wind_99th_percentile_speed",
    "High Adaptation": "wind_995th_percentile_speed",
}


KLAWA_DATA_SOURCES_PATHS = {
    "no_adapt": {
        "path": KLAWA_BASE_DATA_PATH,
        "wind_var_name": MAP_LEVEL_TO_WIND_VAR["No Adaptation"],
    },
    "medium_adapt": {
        "path": KLAWA_V99_DATA_PATH,
        "wind_var_name": MAP_LEVEL_TO_WIND_VAR["Medium Adaptation"],
    },
    "high_adapt": {
        "path": KLAWA_V995_DATA_PATH,
        "wind_var_name": MAP_LEVEL_TO_WIND_VAR["High Adaptation"],
    },
}
