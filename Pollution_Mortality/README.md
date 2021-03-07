#### Project 5: Predicted Pollution Mortality
#### Corey J Sinnott
# Data Import and Cleaning

## Executive Summary

This report was commissioned to explore mortality influenced by pollution. Data was obtained from several sources listed below. The problem statement was defined as, can we predict pollution mortality? After in-depth analysis, conclusions and recommendations will be presented.

Data was obtained from the following source:
- Our World in Data: 
 - https://ourworldindata.org/outdoor-air-pollution
- Environmental Protection Agency (EPA): 
 - https://www.EPA.gov/
 
The data was initially obtained using the Our World in Data's direct links. Further datasets were obtained from querying EPA.gov's user interface. Together, over 30 datasets were obtained and inspected for usability. Features were extracted from several of these sets, and combined into a final working dataframe. Raw data was used for some visualizations and further analysis outside of modeling.

Prior to analysis, dataframes were filtered against a list of recognized nations, null values were removed, and it was decided the earliest data used would be from 1990, due to the inconsistent recording prior. The data was then explored visually for trends and correlations. The resulting graphs can be found in 02_p5_visualizations.ipynb.

Finally, variables were assigned and several regressions were fit. A lasso with least-angles regression (LassoLARS) with standard scaling performed highest, and was chosen for further tuning. 

Ultimately, using annual CO2 emissions, national health spending, life expectancy, ozone depleting emissions, mean daily ozone, and population, we were able to explain pollution mortality with an r-squared of 0.99, and an RMSE of 7400, compared to a null hypothesis result with RMSE equal to 104k deaths. Without population, an r-squared of 0.821 and RMSE of 44k was acheived.

These results are difficult to interpret due to several factors. The first, is the lag between pollution-influenced disease state and mortality. Only 1.4% of pollution deaths are accute,  and researchers have yet to agree on a standard mathematical representation of this cause and effect relationships. To further complicate the relationship between pollution and mortality, experts disagree on estimates of how many deaths are impacted by pollution, with estimates ranging from 1 in 8, up to 40%. It is generally agreed upon that pollution causes inflammation, respiratory distress, various cancers, and ultimately decreases lifespan, but the specific mechanisms are still being researched, and the considerations are endlessly intertwined. Finally, the biggest challenge to accurate predictions, is the drastic increase in health spending, quality of care, and infrastructure that has occured concurrently with the decrease in air quality we have seen over the past 30 years. A standardized crude mortality rate actually shows a slight decrease in mortality over the last 30 years. In order to account for this, variables such as national health spending were included.

In conclusion, pollution metrics alone are not sufficient in predicting mortality. Pollution mortality is a complex mechanism influenced by dozens of outside factors, and is too heavily influenced by population. It is our recommendation that more research is applied toward determining specific disease mechanisms, and risk factors are created from these relationships, rather than determining pollution mortality as one entity.

**Data dictionary** can be found in README.


## Sources
 - Word Health Organization: https://www.who.int/health-topics/air-pollution#tab=tab_1
 - The Distributed Lag between Air Pollution and Deaths. https://www.researchgate.net/publication/12533027_The_Distributed_Lag_between_Air_Pollution_and_Daily_Deaths
 - The Mechanism of Air Pollution and Particulate Matter in Cardiovascular Diseases. https://pubmed.ncbi.nlm.nih.gov/28303426/
 - Pollution Causes 40 Percent of Deaths Worldwide: https://www.sciencedaily.com/releases/2007/08/070813162438.htm
 - National Particle Component Toxicity (NPACT) Initiative: integrated epidemiologic and toxicologic studies of health effects or particulate matter components: https://pubmed.ncbi.nlm.nih.gov/24377209/

## Dictionary  
|Feature|Type|Dataset|Description|
|---|---|---|---|  
|**Year**|*integer*|Found in all dataframes.|Represents the year of data collection (1990 - 2017 for most metrics).|  
|---|---|---|---|  
|**annual_co2_emissions**|*float*|Found in all dataframes.|Reported in parts-per-millions (ppm).|  
|---|---|---|---|  
|**health_spend_per_capita**|*float*|Found in all dataframes.|Dollars spent per capita by government for healthcare.|   
|---|---|---|---|  
|**life_expectancy**|*float*|Found in all dataframes.|Value represents the average life expectancy, per year, per country.|  
|---|---|---|---|  
|**ozone_depleting_emissions**|*float*|Found in all dataframes.|Represents a summed amount of all noxious emissions.|   
|---|---|---|---| 
|**min_daily_ozone**|*float*|Found in all dataframes.|Value represents the minimum amount of ozone recorded for the day.|  
|---|---|---|---| 
|**mean_daily_ozone**|*float*|Found in all dataframes.|Value represents the mean amount of ozone recorded for the day.| 
|---|---|---|---|  
|**population**|*float*|Found in most dataframes.|The total population, per country, per year.|  
|---|---|---|---|  
|**pollution_deaths**|*float*|**Target variable**, Found in all dataframes.|Deaths reported as being in some part due to pollution; both acute and chronic exposure.|   
|---|---|---|---|  
|**PM2.5_ug_m3**|*float*|Found in all dataframes.|Total volume in micrograms per cubic meter of particulate matter smaller than 2.5 microns.|  
|---|---|---|---|  
|**country**|*object*|Found in all dataframes.|Countries, filtered against the European Union's list of recognized nations.|  
|---|---|---|---| 
