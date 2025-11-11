# Multi-Scale Driver Interactions on Crop Phenology via Graph-Based Ensembled Machine Learning
## Project Overview
This project investigates how climate, soil, and topographic factors interact across multiple spatial and temporal scales to influence crop phenology. We leverage satellite-derived vegetation indices (MODIS NDVI/EVI) as proxies for crop greenness, together with climate and terrain data, to model the timing of phenological events (e.g. start/end of growing season). Vegetation phenology is known to be primarily controlled by climatic drivers (precipitation, temperature) with terrain modulating local variability. To handle large geospatial datasets, we use Google Earth Engine-a cloud platform for planetary-scale satellite data analysis. Our novel approach builds a graph-based ensemble ML framework combining a Graph Convolutional Network (GCN) to capture non-linear interactions among drivers and an XGBoost model for robust regression.

## Objectives
Main Aim: Develop a graph-based ensemble ML framework to quantify and predict multi-scale interactions of dynamic climate, static soil, and topographic drivers on crop phenological stages.

Data Integration: Acquire, preprocess, and harmonize multi-source geospatial datasets (climate time series, soil maps, DEM/topography, vegetation indices) and engineer features that capture key cross-scale interactions.

Graph & EDA: Construct graph representations of driver interactions and perform exploratory data analysis (EDA) to identify synergies, correlations, and spatial/temporal patterns influencing phenology.

Modeling: Implement and optimize an ensemble of two models-a graph-oriented model (GCN) and a complementary model (XGBoost)-to predict phenological metrics, using spatial-temporal cross-validation for evaluation.

Interpretation: Apply explainable AI techniques (e.g. SHAP values, Accumulated Local Effects) and sensitivity analysis to interpret model outputs, disentangle driver contributions, and assess responses under varying scenarios.

Validation: Test the framework with independent data, assess transferability across different crops or regions, and discuss implications for sustainable agriculture (e.g. adaptation strategies).

## Data Sources

MODIS (Vegetation Indices): 16-day global NDVI and EVI products at 250–500 m resolution, providing consistent measures of canopy greenness for phenology.

ERA5 Climate Reanalysis: Global hourly weather data (temperature, precipitation, radiation, etc.) on ~31 km grid from 1940–present. We aggregate to seasonal or monthly summaries as needed.

SRTM Topography: Near-global digital elevation model (30 m resolution) from NASA’s Shuttle Radar Topography Mission, supplying elevation, slope, and aspect.

iSDAsoil: High-resolution (30 m) soil property maps for Africa (pH, texture, organic carbon, etc.), created via ML on >100k samples. (We use analogous soil data where available.)

Processing Platform: All datasets are accessed and preprocessed in Google Earth Engine, ensuring consistent spatial alignment and scale harmonization.

## Methodology

### The workflow comprises several stages:

Random Sampling: Generate random sample points (or grid cells) across the study area to gather a representative set of locations.

Feature Extraction: For each sample, extract static features (soil properties from iSDAsoil; elevation, slope, aspect from SRTM) and dynamic features (climate variables from ERA5; NDVI/EVI from MODIS). Temporal aggregation (e.g. seasonal/monthly means, growing-season totals) is applied to dynamic variables.

Phenology Proxies: Derive phenological metrics from vegetation index time series. For example, smooth the NDVI/EVI curve and identify key dates for Start of Season (SOS) and End of Season (EOS) by thresholding or rate-of-change methods. These metrics serve as target variables.

Feature Engineering: Create interaction features (e.g. temperature×soil moisture, elevation×precipitation) to capture synergies. Normalize features (e.g. z-scores) and handle missing data or collinearities (e.g. via imputation or Variance Inflation Factor checks).

Multi-Scale Integration: Harmonize data across scales - e.g., ensure climate time steps align with phenology observations, and resample coarse climate grids to match local spatial scales. This yields a final dataset where each sample has a feature vector spanning local (soil/topo) and broader (climate) contexts.

Exploratory Data Analysis (EDA): Visualize spatial/temporal trends and compute correlations. For instance, plot NDVI curves over years, map SOS across the region, and examine scatterplots or correlation heatmaps among drivers. Preliminary EDA often shows that precipitation and temperature are the strongest predictors of SOS/EOS, as found in the literature.

Graph Construction: Construct interaction graphs (using NetworkX, etc.) where nodes represent features or variables (e.g. “temperature”, “soil moisture”) and edges encode relationships (e.g. weighted by Pearson correlation or mutual information). This graph encodes the multi-way connections among drivers and can include spatial adjacency if modeling locations as nodes.

Ensemble Modeling: Train a Graph Convolutional Network (GCN) on the constructed graph to learn feature embeddings and propagate influence through the network. Concurrently, train an XGBoost regressor on the tabular feature set for robust prediction. Finally, use a stacked ensemble (e.g. scikit-learn’s StackingRegressor) to combine GCN and XGBoost outputs into a final prediction. Hyperparameters are tuned (e.g. via GridSearchCV or Optuna), and models are evaluated with metrics like RMSE and R² using spatial and temporal cross-validation.

## Results Highlights

#### From the exploratory and preliminary modeling results, we note:

Dominant Drivers: Climate variables (especially cumulative precipitation and mean temperature) consistently rank as the top drivers of phenological timing. For example, areas with higher spring rainfall tend to exhibit earlier SOS, echoing prior findings that precipitation largely determines the onset of greenness.

Feature Importances: In initial XGBoost models, the highest importance scores go to seasonal climate features. Static factors (soil nutrients, elevation) have secondary but significant effects, often modulating how climate translates to plant response.

Correlation Patterns: Heatmaps reveal strong positive correlations among related features (e.g., different temperature metrics) and negative correlations where expected (e.g., rainfall vs. aridity indices). Interaction plots suggest, for instance, that elevation can dampen the effect of temperature on phenology.

Spatial/Temporal Trends: Multi-year plots of SOS/EOS show clear seasonality and some interannual variability. Spatial maps (saved in /data_output/) illustrate gradients – for instance, cooler high-elevation zones show delayed greening compared to lowlands. The project outputs numerous visualizations (feature histograms, trend lines, choropleth maps, network graphs) to summarize these patterns.

## Running the Project

### To reproduce the analysis:

Environment Setup: Install Python 3.x and the required libraries. Key libraries include earthengine-api, geopandas, rasterio, numpy, pandas, scikit-learn, xgboost, torch, torch-geometric, etc.

##### Earth Engine Authentication: Authenticate access to Google Earth Engine.

For example:
`pip install earthengine-api`
`earthengine authenticate`


Then initialize in Python (e.g. import ee; ee.Initialize()).

Run the Notebook: Open the Jupyter notebook (e.g. MultiScalePhenology.ipynb) provided with the code. Execute the cells in order. The notebook contains data loading (via GEE), preprocessing, EDA, and the ML modeling steps.

Dependencies: A requirements.txt file is provided to install all needed packages. An Earth Engine account (free for research) is required. A GPU is optional but can speed up training the GCN on large graphs.

## Visual Summaries

The project generates several key figures (saved in the data_output/ directory):

Feature Distributions: Histograms and boxplots of each driver and derived feature (e.g. soil pH, seasonal rainfall) to assess their ranges and variances.

Phenology Time Series: Line charts of annual NDVI/EVI curves and extracted SOS/EOS over multiple years, showing seasonal cycles and trends.

Spatial Maps: Geospatial maps of SOS, EOS, and selected drivers (e.g. precipitation, elevation), highlighting regional variability in phenology and environmental conditions.

Interaction Graphs: Network diagrams where nodes are drivers and edges reflect interaction strengths (e.g. based on correlation). These help visualize complex dependencies.

These plots help interpret the data: for example, correlation heatmaps highlight multicollinearity issues, and feature interaction graphs suggest which driver pairs might jointly influence phenology.

## Future Work

Model Ensemble: Complete the integrated GCN + XGBoost ensemble by final training on the full dataset, refining how GCN embeddings feed into the XGBoost.

Sensitivity Analysis: Perform systematic perturbations (e.g. Monte Carlo or scenario analysis using climate model projections) to quantify how phenology predictions respond to changes in drivers.

Generalization: Apply the framework to other crops or regions by obtaining analogous data (e.g. crop-specific calendars or different biomes) and evaluating transferability.

Causal Inference: Explore causal modeling techniques to strengthen claims about driver impacts (e.g. using causal discovery libraries).

Scalability: Optimize computation (e.g. parallel GEE exports, Dask for data handling) to handle larger spatial extents or finer resolutions.

## Acknowledgments

We acknowledge Google Earth Engine and its public data archive, and thank the providers of the open-source datasets: NASA (MODIS, SRTM), ECMWF (ERA5 reanalysis), and ISRIC/iSDA (soil maps). We also appreciate the open-source Python ecosystems (NumPy, SciPy, PyTorch, etc.) used in this work. Special thanks to research advisors <a href="https://iggres.dkut.ac.ke/dr-bartholomew-thiongo-kuria/" target="_blank">Kuria, T. B.</a>, <a href="https://iggres.dkut.ac.ke/dr-arthur-w-sichangi/" target="_blank">Sichangi W. A.</a>, <a href="https://iggres.dkut.ac.ke/dr-duncan-maina-kimwatu/" target="_blank">Kimwatu M. D.</a>, and supporters for guidance on the design and domain expertise.
