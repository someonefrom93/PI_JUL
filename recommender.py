import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


data = pd.read_parquet("parquet_data/data_for_recommender.parquet")
data_sample = data.sample(2000, random_state=10)
data_sample.reset_index(drop=True, inplace=True)
indeces = pd.Series(data_sample.index, index=data_sample["title"]).drop_duplicates()

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(data_sample["content"].fillna(""))

cosine_similarities_ = linear_kernel(tfidf_matrix, tfidf_matrix)



def get_recommendations(value):
    """Return a dataframe of content recommendations based on TF-IDF cosine similarity.
    
    Args:
        df (object): Pandas dataframe containing the text data. 
        column (string): Name of column used, i.e. 'title'. 
        value (string): Name of title to get recommendations for, i.e. 1982 Ferrari 308 GTSi For Sale by Auction
        cosine_similarities (array): Cosine similarities matrix from linear_kernel
        limit (int, optional): Optional limit on number of recommendations to return. 
        
    Returns: 
        Pandas dataframe. 
    """
    data = pd.read_parquet("parquet_data/data_for_recommender.parquet")
    data_sample = data.sample(2000, random_state=10)
    data_sample.reset_index(drop=True, inplace=True)


    # Return indices for the target dataframe column and drop any duplicates
    indices = pd.Series(data_sample.index, index=data_sample["title"]).drop_duplicates()

    # Get the index for the target value
    target_index = indices[value]

    # Get the cosine similarity scores for the target value
    cosine_similarity_scores = list(enumerate(cosine_similarities_[target_index]))

    # Sort the cosine similarities in order of closest similarity
    cosine_similarity_scores = sorted(cosine_similarity_scores, key=lambda x: x[1], reverse=True)

    # Return tuple of the requested closest scores excluding the target item and index
    cosine_similarity_scores = cosine_similarity_scores[1:11]

    # Extract the tuple values
    index = (x[0] for x in cosine_similarity_scores)
    scores = (x[1] for x in cosine_similarity_scores)    

    # Get the indices for the closest items
    recommendation_indices = [i[0] for i in cosine_similarity_scores]

    # Get the actutal recommendations
    recommendations = data_sample["title"].loc[recommendation_indices]

    # Return a dataframe
    data_sample = pd.DataFrame(list(zip(index, recommendations, scores)), 
                      columns=['index','recommendation', 'cosine_similarity_score']) 

    return {"recommendations": data_sample["recommendation"].tolist()}