🌍 Introduction
RiskX is designed to quantify the chronic risks associated with climate change and extreme events at the level of individual assets. The platform enables detailed, asset-level risk assessment by combining high-resolution hazard modelling, building-level exposure data, and vulnerability functions. With limited resources, the prototype is only applied to Bergen. 

For each building, RiskX provides:
Risk scores under various climate change scenarios (e.g., RCP pathways, though not implemented in the prototype) and building-level adaptation options.


Loss ratios and expected damage estimates.


Portfolio-level risk aggregation to manage systemic risk.

🧭 Methodology Overview
Risk in RiskX is conceptualized as the combination of three key components:
Risk=Hazard×Exposure×Vulnerability

Hazard: The probability and intensity of damaging events, e.g., windstorms under current and future climate conditions.


Exposure: The physical assets at risk, described by their location, type, value, and structural characteristics.


Vulnerability: The expected damage given a hazard level, determined by the building characteristics such as materials, and resilience measures.

By integrating these three elements at high spatial resolution, RiskX aims to:
✅ Asset-level risk mapping
 ✅ Scenario-based planning (including climate change and adaptation levels)
 ✅ Portfolio risk evaluation under evolving climate conditions


Hazard: Wind Data and Downscaling Methodology
This section explains how RiskX estimates high-resolution wind hazard layers suitable for asset-level risk analysis. The goal is to provide detailed wind speed maps that account for local terrain effects, using both existing climate model data and statistical methods.

1. Source Data: NORA3 Wind Speeds
The NORA3 dataset provides wind gust data over Northern Europe at ~3 km spatial resolution.


While NORA3 is excellent for regional analyses, its resolution is too coarse for estimating risk at individual building or asset level.

2. The Need for Downscaling
Wind speeds are highly influenced by local terrain features such as hills, valleys, and land cover.


To model these local effects, RiskX downscales the coarse wind fields to a ~90 m resolution grid (this can be further downscaled to 30m resolution).


This enables detailed estimation of wind hazard for individual buildings or infrastructure assets.

3. Statistical Downscaling Approach
RiskX uses statistical learning methods to model the relationship between wind speed and terrain features. The general idea:
Coarse-resolution wind data is paired with terrain variables to learn how wind speeds vary with local features.


This relationship is then applied to a high-resolution Digital Elevation Model (DEM) to predict wind speeds at fine scale (30m-90m resolutions).


3.1 Training Data
The coarse NORA3 wind gust values are used as the target variable for learning.


Terrain variables are the predictor variables, including:


Elevation (from Copernicus DEM)


Slope and aspect


Terrain roughness metrics


Land cover / vegetation type


3.2 Machine Learning Model
A Random Forest regression model is used to learn the relationship between wind speed and terrain features (wind speed ~ f(terrain features)).


This model is chosen for its ability to capture nonlinear relationships and interactions.


3.3 Application to High-Resolution Grid
Once trained, the model is applied to the full ~90 m DEM grid.


This produces fine-scale wind hazard maps that capture local variability, critical for asset-level risk assessments.

4. Data Sources Used
NORA3 wind speed data: Regional climate reanalysis at ~3 km resolution.


Digital Elevation Model (DEM): Copernicus DEM at ~90 m resolution.

5. Planned Enhancements and Calibration
RiskX is designed to be modular and upgradable. Planned improvements include:
Incorporating observational wind station data for calibration and validation of downscaled fields.


Including additional terrain-derived predictors such as:


Aspect and slope at multiple scales


Local roughness length estimates


Using ensemble approaches to estimate uncertainty ranges in downscaled wind speeds.


Hazard layers for future periods (e.g., 2050, 2080) using bias-corrected EURO-CORDEX projections.



6. Limitations
The accuracy of downscaled wind fields depends on the quality of input data (DEM, land cover, NORA3).


Initial implementations used simpler models (e.g., elevation only) while advanced models with more predictors will be phased in.


Local microclimate effects (urban roughness, small-scale channeling) may not be fully captured without further calibration.

7. How It’s Used in RiskX
Downscaled wind speed maps at ~90 m resolution serve as the hazard layer in RiskX’s risk calculations.


Each asset is overlaid on these layers to extract wind speeds at various return periods (e.g., 2-, 5-, 10-, 50-, 100-, 200-year events).


These hazard estimates feed into the damage functions and vulnerability curves to compute expected loss metrics.



 ✅ Summary
 RiskX transforms coarse regional wind data into high-resolution, asset-level hazard maps using statistical downscaling techniques. This process enables insurers, banks and asset owners to quantify wind risk at the level of individual buildings, while also supporting future climate scenario analysis.

🏠 Exposure: Building-Level Data
Accurate exposure data is essential for asset-level climate risk assessment. In the RiskX framework, exposure represents the characteristics of individual buildings that determine their susceptibility to wind damage.

1. Current Data Sources
OpenStreetMap (OSM) provides widespread coverage of building footprints (shapes and locations).


While valuable for mapping, OSM data is typically limited to geometry only, without detailed structural attributes.



2. Importance of Detailed Building Attributes
To assess individual building risk meaningfully, additional characteristics are required, including:
Number of storeys (affecting wind load)


Roof type and geometry (critical for damage modeling)


Building material (e.g., wood, concrete, steel)


Construction year or age (proxy for building codes and resilience standards)


These variables directly influence vulnerability curves and damage functions used in RiskX.

3. Challenges
Such detailed attributes are rarely available in open data globally.


Local building databases (when they exist) are often incomplete, proprietary, or not harmonized across regions.

4. Planned Enhancements with AI and Remote Sensing
The plan is to evolve by augmenting basic building footprints with richer attributes derived from satellite imagery:
Computer vision models applied to high-resolution satellite or street-level imagery can classify:


Roof shapes and materials


Number of storeys


Construction types


Multi-source data fusion can combine OSM shapes, and remote sensing features.


Machine learning approaches will be trained on regions with labeled data and generalized to new geographies.
These methods promise to systematically generate the critical building attributes necessary for asset-level risk modeling, even in data-poor regions.

5. Role in RiskX Calculations
Each building’s attributes influence its vulnerability curve—the mathematical function relating wind speed to expected damage.


Accurate exposure data enables tailored risk estimates rather than relying on generic assumptions.


As RiskX improves exposure data enrichment, the precision of loss forecasts for individual assets will increase substantially.

✅ Summary
While current open data (like OSM) offers only basic building shapes, RiskX will extend these footprints with advanced AI-driven mapping of critical structural details. This enables a truly asset-level risk assessment capable of supporting insurers, banks, and public planners in understanding climate-related wind risk in unprecedented detail.

🧱 Vulnerability: Estimating Wind Damage to Buildings
The vulnerability module in RiskX translates wind hazard into expected damage at the individual building level. This is achieved using vulnerability curves—functions that relate wind speed to the fraction of asset value likely to be damaged.

1. Current Approach
RiskX currently adopts generalized vulnerability curves derived from Koks & Haer (2020), who developed a high-resolution wind damage model for Europe.



Their approach demonstrated that open-source exposure data (e.g., OpenStreetMap footprints) combined with hazard layers and generalized vulnerability curves can produce credible, building-level wind damage estimates across large areas.

2. Why Vulnerability Curves Matter
Vulnerability curves capture how much damage a building is likely to sustain at a given wind speed.


They account for:


Building type and construction quality


Age and code compliance


Structural features (e.g., roof type)


In the risk model, these curves convert hazard metrics (e.g., gust speed for 50-year return period) into expected damage ratios (EDR)—which then feed into loss estimates.

3. Current Implementation Details
For the initial prototype, standardized damage functions are applied across similar building classes.


These curves represent average expected damage for given wind speed thresholds.

4. Planned Enhancements
RiskX is designed to support progressive refinement of vulnerability estimation:
Incorporation of proprietary insurance claims data to locally calibrate vulnerability curves.


Vulnerability curves has to be tailored to:


Local building codes


Construction practices


Material types


Differentiation by asset class (residential, commercial, industrial) to reflect varying structural resilience.


Post-disaster damage surveys and remote sensing analysis to validate and update curves.

5. Role in Risk Calculations
For each asset, the selected vulnerability curve maps wind hazard metrics (e.g., gust speeds at specified return periods) to expected damage ratios.


The final risk calculation multiplies:


Hazard (wind speed probability)


Vulnerability (damage ratio)


Exposure (asset value)


This produces:


Expected Annual Loss (EAL)


Scenario-based loss estimates (e.g., 10-, 50-, 100-year events)



✅ Summary
RiskX's vulnerability module applies proven, peer-reviewed damage functions to estimate wind damage at building scale. While initial implementations use generalized curves (Koks & Haer, 2020), the platform is designed to integrate regionally calibrated, proprietary, or engineering-based models over time, increasing accuracy and value for insurers, planners, and asset owners.

📈 Risk Score Computation
RiskX assigns risk scores to individual buildings to reflect their relative wind hazard exposure under current and future climate conditions, including adaptation scenarios.
The computation follows these steps:
1 Wind Speed Normalization
For each grid point, the maximum wind gust v_max is compared to a defined adaptation threshold v_threshold:

This formula emphasizes extreme exceedances above the threshold while ignoring non-damaging winds.
2 Adaptation Levels
Different adaptation scenarios adjust the threshold:


Low adaptation: v_98​ (98th percentile wind speed)


Medium adaptation: v_99


High adaptation: v_99.5


By raising the threshold, better-adapted buildings show lower risk scores.


3 Pooling Across Scenarios
Scores are calculated for each adaptation level.


These values are pooled to represent the building’s performance across varying adaptation strategies.


4 Normalization
All pooled scores are scaled between 0 and 100:


The highest-risk building receives a score of 100.


The lowest-risk building receives a score of 0.


This ensures relative comparability across all assets in the study area.



✅ Interpretation:
A higher RiskX score indicates a building is significantly more exposed to damaging wind events relative to its local adaptation threshold. This enables users to prioritize adaptation investments, insurance pricing adjustments, or urban planning interventions.

🌐 RiskX Interactive Application
An interactive web application has been developed to demonstrate and explore RiskX’s asset-level wind risk assessment.
✅ Users can access the app here:
 RiskX Demo Application

Purpose
The app allows to:
Explore building-level risk scores.


Examine loss ratios and expected damages for individual buildings.


Analyze portfolio-level risk metrics.



Access and Sharing
The hosted application is publicly accessible to facilitate review, collaboration, and feedback.


It is intended for demonstration purposes and can be used to showcase RiskX’s preliminary version.

✅ Note: The current version is only a prototype.


