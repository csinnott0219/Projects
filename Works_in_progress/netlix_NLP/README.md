#### Project 4: Netflix NLP
#### Corey J Sinnott
# Modeling and Classification

## Executive Summary

This report was commissioned to perform natural language processing (NLP) and analysis on Netflix movies and television shows. Data includes a description of 7778 titles available on the platform during a time period in 2019. The problem statement was defined as, can we determine if a show or movie has adult content based on the description? For the purpose of this analysis, "adult content" will include any media that has a rating of R, TV-MA, or NC-17. After in-depth analysis, conclusions and recommendations will be presented.

Data was obtained from the following source:
- https://www.kaggle.com/shivamb/netflix-shows
 
Prior to analysis, additional features were created. Beautiful Soup's TextBlob sentiment analysis tools were used to create feature columns for description subjectivity and polarity. The descriptions and titles were also merged, and several forms of count vectorization were implemented, with various tuning, the best being Sci-Kit Learn's CountVectorizer, with stop words set to "english," max features equal to 2500, and n-grams limited to 1, 1.
    
The data was then explored visually for trends and correlations. The resulting graphs can be found in 02_EDA_data_viz_p4_csinnott.ipynb.

Finally, the data was separated into training and test sets, and run through several models, including Naive Bayes, Logistic Regression, AdaBoost, Random Forest, and a TensorFlow Neural Net. Random Forest classificaiton had the highest baseline accuracy, at 72% (null model = 54%), and was chosen for the final classification tasks. 

Based on the findings, we can determine if a Netflix program will have adult content by ~20% greater than a null model. The key limitation in obtaining greater accuracy was a lack of defining key words, and perhaps some overlap between TV-14 and TV-MA ratings, as well as PG-13 ratings. Furthermore, with more time, the model can be further refined and more word vectorization techniques can be explored.
 
In conclusion, it is recommended that more time is put toward refining and tuning the model using GridSearchCV parameters, and that other word vectorization techniques are explored, such as increading max features, and including bi and tri-grams.


## Dictionary  
|Feature|Type|Dataset|Description|
|---|---|---|---|  
|**description**|*object*|Found in all dataframes.|Represents the description of a television show or movie.|  
|---|---|---|---|  
|**description_length**|*integer*|Found in all dataframes.|Represents the total number of characters of the description.|   
|---|---|---|---| 
|**description_word_count**|*integer*|Found in all dataframes, training and test data sets.|Value represents the total number of words in a description.|  
|---|---|---|---|
|**description_polarity**|*integer*|Found in most dataframes, training and test data sets.|Represents the result of Beautiful Soup's TextBlob sentiment analyer's polarity metric on the description. Measures how polarizing a text statement is on a scale of -1 to 1.|   
|---|---|---|---|  
|**description_subjectivity**|*integer*|Found in most dataframes, training and test data sets.|Represents the result of Beautiful Soup's TextBlob sentiment analyer's polarity metric on the description. Measures how subjective a text statement is on a scale of 0 to |---|---|---|---|   
* the remaining terms can be found at https://www.kaggle.com/shivamb/netflix-shows *  

## Sources
 - https://www.kaggle.com/shivamb/netflix-shows