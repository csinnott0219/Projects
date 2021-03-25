import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score,\
         precision_score, recall_score, roc_auc_score,\
         plot_confusion_matrix, classification_report, plot_roc_curve, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import ADASYN
import os
import io
import streamlit as st
import streamlit.components.v1 as stc 
from pandas_profiling import profile_report, ProfileReport
import sweetviz as sv
from streamlit_pandas_profiling import st_profile_report

st.markdown(
    """
<style>
.reportview-container .markdown-text-container {
    
    font-family: IBM Plex Sans;
}
.sidebar .sidebar-content {
    background-image: url("https://www.kovifabrics.com/img/thumbs/797c835879623bd16ac825e62d203c22.JPG");
    color: white;
}
.Widget>label {
    color: white;
    font-family: monospace;
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.st-bb {
    background-color: #d19da6;
}
.st-at {
    background-color: #3d393a;
}
footer {
    font-family: monospace;

}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #013d29;
}
header .decoration {
    background-image: url("https://img2.mahoneswallpapershop.com/prodimage/ProductImage/560/aecd25f8-4822-42ca-85f7-62d63cd41fc3.jpg");
}

</style>
""",
    unsafe_allow_html=True,
)

st.title('Propensity to Give')
st.text('by Corey J Sinnott')
st.subheader('This app prepares dataframes, models and classifies, and\
              outputs feature importances and probabilities.')

# ---------------------- functions ------------------------------------ #
# select files
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

# create target
def define_target(df_donations):
    """
    Takes in dataframe of yearly donations.
    
    Args: donations dataframe
    
    Returns: dataframe of donors with binary classification
    """
    # progress output
    st.spinner()
    with st.spinner(text='Target preparation in-progress ...'):
        # convert to timeseries
        df_donations['trans_date'] = pd.to_datetime(df_donations['trans_date'])
        # filter for year to select for active donors
        df_donations = df_donations[df_donations.trans_date >= '2000-01-01 00:00:00']
        # reset index to date
        df_donations = df_donations.set_index(keys = 'trans_date')
        # sum yearly total donations
        yearly_donations = pd.DataFrame(df_donations.reset_index().groupby(['cand_id', 
                                    pd.Grouper(key='trans_date', freq='A-DEC')])['amount'].sum())
        # filter out zeros
        yearly_donations_classed = yearly_donations[yearly_donations['amount'] != 0]
        # create rolling average columns
        yearly_donations_classed['rolling_avg'] = \
            yearly_donations_classed.groupby('cand_id')['amount'].transform(lambda x: x.rolling(12, 1).mean())
        # reset index to push cand_id back out
        yearly_donations_classed = yearly_donations_classed.reset_index()
        # selecting the final average per year amount
        # value represents the most recent yearly donation average
        donor_final_df = \
            yearly_donations_classed.loc[yearly_donations_classed.groupby('cand_id').rolling_avg.idxmax()]
        # trim dataframe
        donor_final_df.drop(columns = ['trans_date','amount'], inplace = True)
        # create binary target
        # yearly donations averaging >= $4k, 1, < $4k, 0
        donor_final_df['20k_donor'] = np.where(donor_final_df['rolling_avg'] >= 4000, 1, 0)
        final_counts = donor_final_df['20k_donor']
        # inspect classes
        print(f'Target value counts: {final_counts.value_counts(normalize = True)}')
        
        st.success('Target prep complete')

    return donor_final_df

# combine dataframes
def df_combiner(df_features, df_donations):
    """
    Combines dataframe of features and dataframe
    of target values.
    
    Args: features df, donations df
    
    Returns: trimmed and prepared dataframe for classification
    """
    st.spinner()
    with st.spinner(text='Prepping features and merging ...'):
        # rename columns
        df_features.rename(columns = {'candidate_id' : 'cand_id'}, inplace = True)
        # merge dataframes on cand_id
        df = pd.merge(left = df_features, right = df_donations, on = 'cand_id', how = 'right')
        # remove nulls
        df_trim = df[df['NetWorth'].notna()]
        # exporting the final df as csv
        # df_trim.to_csv('final_df.csv') # uncomment to export
        st.success('Features and merge complete')

    return df_trim

# export EDA and visuals at HTML docs
def EDA_visualizations(df):
    """
    Outputs an HTML doc for EDA, and another
    of visualizations.

    Args: dataframe

    Returns: (2) HTML docs
    """
    st.spinner()
    with st.spinner(text='Generating EDA and visualization HTML docs ...'):
        #df.profile_report() # uncomment to view profile in notebook
        profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
        # uncomment to export Pandas profile report as HTML
        # profile.to_file("df_EDA_report.html")
        # create sweetviz HTML doc
        # report = sv.analyze(df)
        # show_viz = st.checkbox('Show visualizations')
        #report.show_notebook()

        st.success('EDA complete')
        show_eda = st.checkbox('Show EDA report (this may take up to 3 minutes)')
        if show_eda:
            st_profile_report(profile)

    return profile

def donor_predict_classification (df, model, imblearn_tool):
    """
    Takes in dataframe of features and target,
    creates x and y variables, scales features, 
    balances target, fits a classification model,
    and return metrics.
    
    Args: dataframe, model, imblearn over-sampling tool
    
    Returns: trained model, X_train, X_test, y_train, y_test, y_pred
    """
    st.spinner()
    with st.spinner(text='Random Forest classification in-progress ...'):

        # define variables
        X = df.drop(columns = ['cand_id', 'rolling_avg', '20k_donor'])
        y = df['20k_donor']
        # scale features 
        X_scaled = StandardScaler().fit_transform(X)
        # oversampling technique
        sm = imblearn_tool
        X, y = sm.fit_resample(X, y)
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    stratify = y, random_state = 22)
        # instantiate and fit, then predict
        model = model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # uncomment to print metrics to terminal
        # print(f'accuracy  = {np.round(accuracy_score(y_test, y_pred), 3)}')
        # print(f'precision = {np.round(precision_score(y_test, y_pred), 3)}')
        # print(f'recall    = {np.round(recall_score(y_test, y_pred), 3)}')
        # print(f'f1-score  = {np.round(f1_score(y_test, y_pred), 3)}')
        # print(f'roc auc   = {np.round(roc_auc_score(y_test, y_pred), 3)}')
        # print(f'{confusion_matrix(y_test, y_pred)} -> confusion matrix')
        # print(f'{round(max(y_test.mean(), 1 - y_test.mean()), 2)} -> null accuracy')
        # streamlit version metrics
        st.write(f'Accuracy  = {np.round(accuracy_score(y_test, y_pred), 3)}')
        st.write(f'Precision = {np.round(precision_score(y_test, y_pred), 3)}')
        st.write(f'Recall    = {np.round(recall_score(y_test, y_pred), 3)}')
        st.write(f'F1-score  = {np.round(f1_score(y_test, y_pred), 3)}')
        st.write(f'ROC AUC   = {np.round(roc_auc_score(y_test, y_pred), 3)}')
        # st.write(f'{confusion_matrix(y_test, y_pred)} -> confusion matrix')
        st.write(f'{(round(max(y_test.mean(), 1 - y_test.mean()), 2))*100}% null accuracy')
        # graph confusion matrix
        # matrix = plot_confusion_matrix(model, X_test, y_test);
        # st.pyplot(f'Confusion matrix: {matrix}')
        # visualize ROC AUC
        plot_roc_curve(model, X_test, y_test)
        plt.title('ROC AUC score and curve for donor classification');
        plt.plot([0,1], [0,1], 'k--')
        plt.show();
        # alternative to metrics above
        # st.text('Model Report:\n ' + classification_report(y_test, y_pred))
            
        # return model and variables for feature extraction
    return model, X_train, X_test, y_train, y_test, y_pred, X_scaled

def probabilities(df, model, X):
    """
    Calculates probabilites for each candidate.
    
    Args: original dataframe, model, X (scaled)

    Return: updated dataframe
    """
    df['pred_probability'] = [i for i in model.predict_proba(X_scaled)]

    return df

def feature_importance(model, X):
    """
    Determines feature importances for each feature

    Args: model

    Return: dataframe of feature importances, and graph
    """
    st.spinner()
    with st.spinner(text='Generating feature importances and probabilites ...'):
        # create dataframe
        feature_import_df = pd.DataFrame(model.feature_importances_, 
                                    index =X_train.columns,  
                                    columns=['importance']).sort_values('importance', 
                                                                        ascending=False)
        # prep data for graphing
        graph_feat_importance = feature_import_df.reset_index(col_fill = 'feature')
        # graph features
        plt.figure(figsize=(15, 10))
        x = graph_feat_importance['importance'].head(10)
        y = graph_feat_importance['index'].head(10)
        sns.barplot(x = x, y = y).set_title('Top ranking features by importance');

        st.success('Analysis complete')

    return feature_import_df

# ------------------------------------------------------------- #

# sidebar user interface
# with st.sidebar.header('Drop in donation and feature .csvs to get started.'):

    
# st.write('Donations csv:')
# df_donations_in = st.file_uploader('Enter donations csv file path:', type = ['csv'])
# df_donations = pd.read_csv(df_donations_in)

# st.write('Features csv:')
# df_features_in = st.file_uploader('Enter features csv file path:', type = ['csv'])
# df_features = pd.read_csv(df_features_in)


# button = st.button('Submit')

# primary user interface
show_upload = st.checkbox('Ready to begin')
if show_upload:
    #upload pic code here 
    st.text('Upload your donations csv file below:')
    uploaded_file_1 = st.file_uploader("Drop donations csv here", type = 'csv')
    st.text('Upload your features csv file below:')
    uploaded_file_2 = st.file_uploader("Drop features csv here", type = 'csv')

    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        df_donations = pd.read_csv(uploaded_file_1)
        df_features = pd.read_csv(uploaded_file_2)
        
# initiating script
# if button:
    # pre-processing
        df_donations_classed = define_target(df_donations)
        df_final = df_combiner(df_features, df_donations_classed)
        # EDA and visuals
        EDA_visualizations(df_final)
        # running model
        model, X_train, X_test, y_train, y_test, y_pred, X_scaled = \
                                        donor_predict_classification (df_final, 
                                        RandomForestClassifier(), 
                                        ADASYN(sampling_strategy = 0.8))
        # calculating probabilites, and adding to df
        df_to_export = probabilities(df_final, model, X_scaled)
        # extracting feature importances
        feature_importances_df = feature_importance(model, X_train)
        # display feature importances
        st.subheader('Feature importances:')
        st.dataframe(feature_importances_df)
        # display probabilities
        final_df = df_to_export[['cand_id', 'rolling_avg', '20k_donor', 'pred_probability']]
        st.subheader('Prediction probabilites:')
        st.dataframe(final_df)

        # export final dataframe as csv
        # final_df.to_csv('final_df.csv') #uncomment to save