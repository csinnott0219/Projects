#### Project 3: Reddit NLP
#### Corey J Sinnott
   
## Executive Summary
   
This report was commissioned to perform natural language processing (NLP) and analysis on two subreddits of Reddit.com. Data includes 8000 posts, 4000 belonging to r/AskALiberal, and 4000 belonging to r/AskAConservative. The problem statement was defined as, can we classify to which subreddit a post belongs? After in-depth analysis, conclusions and recommendations will be presented.
   
Data was obtained from the following source:
- r/AskALiberal of Reddit.com: 
 - https://www.reddit.com/r/askaliberal/
- r/AskAConservative of Reddit.com: 
 - https://www.reddit.com/r/askaconservative/
 
The data was initially obtained using the Pushshift API, and a novel function created to best utilize the API. The API and function together allowed the extraction of 8000 post titles and post bodies, which were then exported to CSV for analysis.
  
Prior to analysis, additional features were created. The Python Language Tool library was used to count grammar errors in post bodies, and a column was engineered for grammar error rate. Next, Beautiful Soup's TextBlob sentiment analysis tools were used to create feature columns for post body subjectivity and polarity, as well as post title subjectivity and polarity. Finally, the posts and titles were merged, and several forms of count vectorization were implemented, with various tuning, the best being Sci-Kit Learn's CountVectorizer, with stop words set to "english," max features equal to 2500, and n-grams limited to 1, 1.
      
The data was then explored visually for trends and correlations. The resulting graphs can be found in data_visuals_p3_csinnott.ipynb.
   
Finally, the data was separated into training and test sets, and run through several models, including Naive Bayes, Logistic Regression, K-Nearest, MLP, AdaBoost, Bagging, Random Forest, and a TensorFlow Neural Net. Random Forest classificaiton performed with the highest degree of accuracy, at 74%. Precision was measured at 75%, recall 62%, and an f-1 score of 68%.
    
Based on the findings, and to answer our problem statement, we can determine the subreddit of origin of r/askaliberal and r/askaconservative posts by ~25% greater than a null model. The key limitation in obtaining greater accuracy was the similarity of posts in the two subreddits. This is due to more than the political nature of the subreddits. Instead, both subreddits are used for antagonistic cross-posting, resulting in questions and posts that are very similar between the two. Furthermore, one subreddit has much more moderation than the other, leading both to resemble one subreddit when analyzed. Of the 8000 posts obtained, ~800 were "removed by moderator." All ~800 removed posts belonged to r/askaconservative, equating to ~20% removal rate. To increase accuracy, it would be necessary to obtain posts as they are posted, prior to being moderated. This, in addition to obtaining several thousand more posts, will increase our future ability to predict subreddits.

## Dictionary  
|Feature|Type|Dataset|Description|
|---|---|---|---|  
|**title**|*object*|Found in all dataframes.|Represents the title of a subreddit post.|  
|---|---|---|---|  
|**selftext**|*object*|Found in all dataframes.|Represents the body of the subreddit post.|  
|---|---|---|---|  
|**subreddit**|*integer*|**Target variable**, Found in all dataframes, training and test data sets.|The origin of the post; converted to binary for classification.|   
|---|---|---|---|  
|**created_utc**|*integer*|Found in all dataframes.|Value represents the epoch time of post.|  
|---|---|---|---|  
|**post_length**|*integer*|Found in all dataframes.|Represents the total number of characters of the post.|   
|---|---|---|---| 
|**post_word_count**|*integer*|Found in all dataframes, training and test data sets.|Value represents the total number of words in a post.|  
|---|---|---|---| 
|**title_length**|*integer*|Found in all dataframes, training and test data sets.|Represents the total number of characters in a post's title.|   
|---|---|---|---|  
|**title_word_count**|*integer*|Found in most dataframes.|The total number of words in the post's title.|  
|---|---|---|---|  
|**num_of_gram_erros**|*integer*|Found in most dataframes.|Calculated using Python Language Tool; a count of grammar errors and miss-spellings.|   
|---|---|---|---|  
|**gram_err_rate**|*integer*|Found in most dataframes, training and test data sets.|The number of grammatical errors divided by the post-length.|  
|---|---|---|---|  
|**selftext_polarity**|*integer*|Found in most dataframes, training and test data sets.|Represents the result of Beautiful Soup's TextBlob sentiment analyer's polarity metric on the post body. Measures how polarizing a text statement is on a scale of -1 to 1.|   
|---|---|---|---|  
|**selftext_subjectivity**|*integer*|Found in most dataframes, training and test data sets.|Represents the result of Beautiful Soup's TextBlob sentiment analyer's polarity metric on the post body. Measures how subjective a text statement is on a scale of 0 to 1.|  
|---|---|---|---|  
|**title_polarity**|*integer*|Found in most dataframes, training and test data sets.|Represents the result of Beautiful Soup's TextBlob sentiment analyer's polarity metric on the post title. Measures how polarizing a text statement is on a scale of -1 to 1.|  
|---|---|---|---|  
|**title_subjectivity**|*integer*|Found in most dataframes, training and test data sets.|Represents the result of Beautiful Soup's TextBlob sentiment analyer's polarity metric on the post title. Measures how subjective a text statement is on a scale of 0 to 1.|  
|---|---|---|---|  
|**all_text**|*object*|Found in final dataframes, training and test data sets.|Represents a combined text composed of the post body and post title; used for CountVectorization.|  
  

## Sources
 - www.reddit.com
 - https://www.reddit.com/r/askaliberal/
 - https://www.reddit.com/r/askaconservative/
 - https://en.wikipedia.org/wiki/Reddit
 - http://scholar.google.com/scholar_url?url=https://www.mdpi.com/1099-4300/19/7/330/pdf&hl=en&sa=X&ei=29cJYKqGHL3EywSi74rAAQ&scisig=AAGBfm1Wq9xZiTx05ui5vdTMsAtGb7CUvw&nossl=1&oi=scholarr
 - https://www.researchgate.net/publication/318252186_Overfitting_Reduction_of_Text_Classification_Based_on_AdaBELM