#### Windfall Children's Center Propensity to Give
#### Corey J Sinnott
   
## Executive Summary
   
This report was commissioned to develop a model used to predict major donors. A major donor will be considered as any individual likely to give at least 20,000 USD. An initial dataset of ~130,000 donors can be found in the data folder, as well as a dataset containing potential feature values.

The initial data set of donations was first converted to timeseries. Next, a rolling average with a period of 1 year (to capture new donors), and a window of 5 years (to best represent the most recent donation-years) was engineered. The dataset was then filtered for only the last value, which we will assume is the best estimate for a yearly donation. Finally, binary values for were added, with those averaging >= 4000 USD/year donations classed as 1, and all others, 0. The target was highly imbalanced, at 98% 0, 2% 1.

The processed donor dataset was then merged with the feature set on the candidate ID, and all donors without features (null values) were dropped. With this final dataset, exploratory data analysis and visualization was performed using Pandas Profiling and SweetViz. The resulting HTML docs can be found in the images folder.

Next, several models were tested. Initially, the imbalanced data was classified using a Logistic Regression, to establish a baseline of performance. This model did not outperform a null model. Next, several strategies were used to approach the imbalanced target. The highest performing was an over-sampling technique from the imblearn library, ADASYN. Per the imblearn learning docs, the ADASYN algorithm "focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier."

After discovering the best over-sampling approach, several classification models were implented. The highest performing was a random forest classifier. This model was then tuned using a GridSearch, and the final metrics were:
- Accuracy  = 0.992
- Precision = 0.994
- f1-score  = 0.991
- Recall    = 0.988
- ROC AUC   = 0.991
- Tendency toward false positives (confusion matrix available in notebook 04).  

Scoring more false positives is considered ideal as the likelihood of missing a large donor is lessened.

Next, feature importances were extracted and graphed (notebook 04). The most important feature was 'isClassDDonor' followed by 'NetWorth'. After extracting features, probabilities were calculated for candidate. This value is of great importance, and can be considered when determining additional donor close to being classified as elite donors.

Finally, all of the steps above were scripted and deployed using Streamlit. The application script can be found in this repository, and the app can be used here: https://share.streamlit.io/csinnott0219/projects/main/Works_in_progress/donor_predictor/script_files/propensity_to_give.py. Futhermore, an analyst could download the script and run it on their local machine via their terminal ($ streamlit run propensity_to_give.py).

The findings from this analysis can be used by clients to target elite donors. Furthermore, using the extracted features and propabilities, a client could identify future elite donors as well, or use the predictive model to classify new donors as data becomes available.

