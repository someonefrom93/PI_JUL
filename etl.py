#!/usr/bin/env python
# coding: utf-8

# # ETL
# 

# In[1]:


import pandas as pd
import numpy as np
import ast
import sys


# # Reducing data sources (csv files to parquet)
# 
# First, let's depurate all fields that wont be needed from our csv files, those csv files will be ignored in the github repo. This code will be commented because this process is run only in my local machine in order to upload only lighwieght parquet files

# In[2]:


# movies = pd.read_csv("data/movies_dataset.csv")
# credits = pd.read_csv("data/credits.csv")
# movies["popularity"] = movies["popularity"].astype(str)


# ### Dropping Fields
# Let's begin with the easiest ones: dropping fields! 
# 
# For the porpuse of meeting all the returns of the end points, these columns are not needed:
# 
#         * video
#         * imdb_id
#         * adult
#         * original_title
#         * poster_path
#         * homepage
#         * status
#         * tagline
#         * vote_average
#         * vote_count
#         * popularity
# 

# In[3]:


# sys.getsizeof(movies) / 1000000


# In[4]:


# movies.drop(["video", "imdb_id", "adult", "original_title", "poster_path", "homepage", "status", "tagline", "vote_average", "vote_count", "popularity", "overview"], axis=1, inplace=True)


# In[5]:


# sys.getsizeof(movies) / 1000000


# ### Reducing credit dataframe size

# In[6]:


# sys.getsizeof(credits) / 1000000


# In[7]:


# def safe_literal_eval(x):
#     try:
#         return ast.literal_eval(x)
#     except (ValueError, SyntaxError):
#         return {}


# In[8]:


# # Dropping heavyweight columns cast
# credits.drop("cast", axis=1, inplace=True)
# # Converting all string data into object
# credits["crew"] = credits["crew"].apply(safe_literal_eval)
# # Getting only job position whose value is "Director only"
# credits["crew"] = credits["crew"].apply(lambda colaborators: [colaborator for colaborator in colaborators if colaborator["job"] == "Director"])
# # Getting only name of the director in order to reduce memory usage
# credits["crew"] = credits["crew"].apply(lambda director: [{"name": attribute["name"]} for attribute in director])
# # Renaming the column to a more suitable one
# credits.rename(columns={"crew": "directors"}, inplace=True)
# # Converting the id field in string in order to merge this data with the movies data frame
# credits["id"] = credits["id"].astype(str)


# In[9]:


# sys.getsizeof(credits) / 1000000


# In[10]:


# movies.to_parquet("parquet_data/movies_parquet.parquet")
# credits.to_parquet("parquet_data/credits_parquet.parquet")


# # Importing new parquet Files

# In[11]:


movies_df = pd.read_parquet("parquet_data/movies_parquet.parquet")
credits_parquet = pd.read_parquet("parquet_data/credits_parquet.parquet")


# ### Movies Data Frame

# In[12]:


movies_df = pd.read_parquet("parquet_data/movies_parquet.parquet")




# ### Data Types Consistency
# Sometimes, there could be some unexpected data in a field, either a string datatype in a numeric field or a numeric dtype in a string field. It is important to check datatype consistency on each field. And the bellow ad-hoc function will help us to do that a bit easier. 
# 
# ##### What it does? 
# 
# Answer: It returns the indexes of intrusive rows in the series, which it'll be used as a mask. That will allow us to: 
#  * Have a visualization of the inconsistencies and 
#  * The rows position that we would like to impute them.

# In[18]:


def dtype_checker(data: pd.DataFrame, column: str, data_type) -> list:
    """
    Returns an array of indexes of rows with a different data type in the specified column.

    Parameters:
        data (pd.DataFrame or pd.Series): The DataFrame or Series to check.
        column (str): The name of the column to check for data type.
        data_type: The expected data type for the values in the column.

    Returns:
        list: An array of indexes of rows where the data type in the specified column is different from the expected data type.
    """

    invalid_dtype_rows = []

    if isinstance(data, pd.DataFrame):
        for row in data[column].items():
            if not isinstance(row[1], data_type):
                invalid_dtype_rows.append(row[0])
    elif isinstance(data, pd.Series):
        for row in data.items():
            if not isinstance(row[1], data_type):
                invalid_dtype_rows.append(row[0])
    else:
        raise ValueError("Invalid input data type. The data must be a pandas DataFrame or Series.")

    return invalid_dtype_rows


# It is usefull to get to know what are the default dtype the dataframe is built of. For this, lets check the data type on a single row in the release_date field



# Once we get to know the dtype the data is readen, we can see the simple output of the function wich is only  the index position of all that rows with different data type from string



# And the above list will be usefull for slicing data and do what may be more comvinience: 
# 
# * impute
# * drop
# * isolate
# * etc..

# ### Creating release_year field in our data frame
# 
# 1. Check data consistency:
#     * Spot intrusive (null, int, floats, etc) data and then drop those rows.
#     * Spot any other row that does not match this pattern "1900-01-01"
# 2. Cast year to create release_year field:
#     * With `pd.Series.dt.year()`. It'll only work once we get consistency format
# 
# 

# In[21]:


# Spoting intrusive data
mask_for_date_intrusives = dtype_checker(movies_df, column="release_date", data_type=str)
# Dropping rows with nulls
movies_df.drop(mask_for_date_intrusives, inplace=True)




# But there are also "numbers" in this field, and I named numbers between parentesis because, literally, there could be numbers like "1", "2", "121", etc.. they are recognized from the dtype checker function as string, and we are still going to get issues when casting data as date type. See how we can spot those string numbers by regex expresion.

# In[24]:


regex_date = r"^\d{4}-\d{2}-\d{2}$" 
#In this lil line, by adding in the begining this ~ to our array of booleans, we get the opposite from pd.Series.str.contains(any_regex_expression)
movies_df.loc[~movies_df["release_date"].str.contains(regex_date)]





# In[26]:


# So that lets impute these values as well.
movies_df.drop(movies_df.loc[~movies_df["release_date"].str.contains(regex_date)].index, inplace=True)



# # And now, no further issues to cast this column as date type and grab the year only to create our year field.

movies_df['release_year'] = pd.to_datetime(movies_df["release_date"]).dt.year

# In[28]:



# ### Filling null values on "revenue" and "budget"
# 
# According to henry's guidance project, We should fill numeric missing data with zeros.
# 
# Steps:
# 
#     1. Check consistency:
#         * Data type
#         * Format (Regex)
#         * Nulls
#     2. Impute and convert





# In[37]:


movies_df["budget"] = movies_df["budget"].astype(float)




# ### Creating "return_on_investment" (ROI) fiel
# 
# This is what is performed.. divide revenue by budget as float, then fill null values with zero.. after filling nulls replace inifinites by zeros.


# In[40]:


movies_df["return_on_investment"] = movies_df["revenue"].div(movies_df["budget"].astype(float)).fillna(0).replace([np.inf, -np.inf], 0)



# ### Flatting Nested Data
# 
# Fields such as belongs_to_collection and genres have relevant data, needed for our API, EDA, and ML model. So lets wrangling and format those fields in order to allow us to query and manipulate as needed.
# 
# Lets begin with belongs_to_collection field. It seems that this is a field with few missing values and this is because not all movies belongs to a collection such as James Bon's movies. Also, each row contains a dictionary.
# 
# #### The approach
# 
# Data Transformation:
# 
#         * Explore data in field (dtype)
#         * Create a new data frame for bellongs_to_collection
#         * Convert data into the original object (dictionary)
# 

# ### Data Modeling on belongs_to_collection Field




# In[45]:


# Building the new data frame for belongs_to_collection
belongs_to_collections_df = movies_df.loc[movies_df["belongs_to_collection"].isnull() == False, "belongs_to_collection"]
belongs_to_collections_df.shape


# Since belongs_to_collection only contains "object" types, we need to convert those pandas' object types into dictionary by `ast.literal_eval()` wich grabs the string and catch it as the data format that seems to be. 
# 
# For example: "{'hello': 2}" string --> {'hello': 2} dict. And it will enable all the dictionary methods needed for this data manipulation

# In[46]:


# This functionn is only for handling any expected error and impute with default empty list
def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return {}


# In[47]:


belongs_to_collections_df = belongs_to_collections_df.apply(safe_literal_eval)

# In[50]:


collections_df = pd.DataFrame(belongs_to_collections_df.tolist())


# In[51]:


collections_df.drop_duplicates(subset=["id"], keep='first', inplace=True)


# In[52]:


collections_df.drop(["poster_path", "backdrop_path"], axis=1, inplace=True)



# In[54]:


collections_df.rename(columns={"id": "collection_id"}, inplace=True)


# In[55]:


collections_df["collection_id"] = collections_df["collection_id"].astype(str)


# ### Getting Id's from belongs_to_collection field

# In[56]:


# In order to deal with null values, impute those records with an default object is conveniant when is time to process each object without further dtype issues.
# Is like adding fake consistency, letter, we can spot those missing data to its origin dtype
default_collection = "{'id': 1234567, 'name': 'Null', 'poster_path': 'im.jpg', 'backdrop_path': 'im.jpg'}"
movies_df.loc[movies_df["belongs_to_collection"].isna(), "belongs_to_collection"] = default_collection
# Getting only the Ids
movies_df["belongs_to_collection"] = movies_df["belongs_to_collection"].apply(ast.literal_eval).apply(lambda x: x["id"]).astype(str)
# Imputing to nulls
movies_df.loc[movies_df["belongs_to_collection"] == "1234567", "belongs_to_collection"] = np.nan
# Renaming belongs_to_collection field to collection_id to get consistency in field name when mergin datasets
movies_df.rename(columns={"belongs_to_collection": "collection_id"}, inplace=True)







# ### Flattent genres field
# 
# This is going to be a bit more complex since in this case we have a many objects related to one row, and many rows releted to one object.. In other words, a reletion ship many to many. 
# 
# #### The Approach:
# 
# Normaliza genres:
# 
#         * Build a new data frame for genres:
#                 * Cast any single row in genres as lists
#                 * Check data consistency and impute
#         * Extract information:
#                 * Explode genres field.
#                 * Drop duplicates
#         * Drop duplicates
# 
# Data Transformation for genres field in movies_df:
# 
#         * Grab only ids




# In[60]:


empty_list_pattern = r'^\[\]$'
genre_empy_mask = movies_df["genres"].astype(str).str.match(empty_list_pattern)
print(genre_empy_mask.sum())


# In[61]:


movies_df.loc[genre_empy_mask, "genres"] = "[{'id': 123456, 'name': 'Unknown'}]"



# In[63]:


movies_df["genres"] = movies_df["genres"].apply(safe_literal_eval)



# In[67]:


genres_df = movies_df.explode('genres', ignore_index=True)[["genres"]].rename(columns={"genres": "genres_info"})



# In[69]:


genres_df["genre_id"] = genres_df["genres_info"].apply(lambda genre: genre["id"])
genres_df["genre_id"] = genres_df["genre_id"].astype(str)

genres_df["genre_name"] = genres_df["genres_info"].apply(lambda genre: genre["name"])
genres_df.drop(columns="genres_info", inplace=True)


# In[70]:


genres_df.drop_duplicates(subset=["genre_id"], keep="first", inplace=True)


# In[71]:


movies_genres_df = movies_df.explode('genres', ignore_index=True)[['id', 'genres']].rename(columns={'genres': 'genre_info'})
movies_genres_df["genre_id"] = movies_genres_df["genre_info"].apply(lambda genre: genre["id"])
movies_genres_df["genre_id"] = movies_genres_df["genre_id"].astype(str)


# In[72]:


movies_genres_df.drop(columns="genre_info", inplace=True)


# In[73]:


movies_genres_df.merge(movies_df[["id", "title"]], on='id').merge(genres_df, on="genre_id")


# In[74]:


movies_df.drop(labels=["genres"], axis=1, inplace=True)





# ### Production Companies Field



# In[77]:


empty_list_pattern = r'^\[\]$'
print(movies_df["production_companies"].str.match(empty_list_pattern).sum())
production_movies_mask = movies_df["production_companies"].str.match(empty_list_pattern)

# In[79]:


movies_df.loc[production_movies_mask, "production_companies"] = "[{'name': 'Unknown', 'id': 123456}]"


# In[81]:


movies_df["production_companies"] = movies_df["production_companies"].apply(safe_literal_eval)


# In[82]:


production_companies_df = movies_df.explode("production_companies", ignore_index=True)[["production_companies"]].rename(columns={"production_companies": "production_companies_info"})


# In[83]:


production_companies_df["company_name"] = production_companies_df["production_companies_info"].apply(lambda prod_company: prod_company["name"])
production_companies_df["company_id"] = production_companies_df["production_companies_info"].apply(lambda prod_company: prod_company["id"])
production_companies_df.drop("production_companies_info", axis=1, inplace=True)


# In[84]:


production_companies_df.drop_duplicates(subset=["company_id"], keep="first", inplace=True)





# In[86]:


production_companies_df["company_id"] = production_companies_df["company_id"].astype(str)





# DataFrame for the many-to-many relationship

# In[88]:


movies_production_companies_df = movies_df.explode("production_companies", ignore_index=True)[["id", "production_companies"]].rename(columns={"production_companies": "production_companies_info"})
movies_production_companies_df["company_id"] = movies_production_companies_df["production_companies_info"].apply(lambda prod_company: prod_company["id"])
movies_production_companies_df.drop("production_companies_info", axis=1, inplace=True)
movies_production_companies_df["company_id"] = movies_production_companies_df["company_id"].astype(str)



# ### Trying it out

# In[90]:


company_demo = movies_production_companies_df.merge(movies_df[["id", "title"]], on = "id").merge(production_companies_df, on="company_id")


# In[91]:


company_demo.loc[(company_demo["title"].str.contains("")) & (company_demo["company_name"] == "Pixar Animation Studios")]


# In[92]:


movies_df.drop("production_companies", axis=1, inplace=True)


# ### Production Countries Field
# 
# For only learning purpuses, the approach to tackle this down will be completely different from normalizing data and building tables. Here the approach is going to be the accesing approach by `.apply()` method and lambda expressions against nested data. This leverage the processecing resources rather than the store recourses because we aren't going to create any additionally dataframe.
# 
# How it works?
# 
# * First of all, we should take care about missing value, in this case they are represented as empy list and levearing this data is currently in string datatype, it will be easier to spot those empty list with regex patterns and then impute a default value to simply label them as "unknowns"
# * Once we get all rows in production_companies field with an a list with at least one dictionary object, we should be able to apply our safe_literal_eval to convert those strings into the appropiate objects
# * Then, we could filter information over these nested data (dictionaries in a list) with `.apply()` method wich applies any function on each record. We can use eaither custom funcions (for example, safe_literal_eval is one custom function applied on each record in the production_companies field) or lamba expressions.




# In[95]:


movies_df["production_countries"].str.match(empty_list_pattern).sum()
production_countries_empties_mask = movies_df["production_countries"].str.match(empty_list_pattern)


# In[96]:


movies_df.loc[production_countries_empties_mask, "production_countries"] = "[{'iso_3166_1': 'Unknown', 'name': 'Unknown'}]"


# In[97]:


movies_df.loc[production_countries_empties_mask, "production_countries"]


# In[98]:


movies_df["production_countries"] = movies_df["production_countries"].apply(safe_literal_eval)




# # Spoken Language field



# In[103]:


spoken_languages_mask = movies_df["spoken_languages"].str.match(empty_list_pattern)


# In[104]:


movies_df.loc[spoken_languages_mask, "spoken_languages"] = "[{'iso_639_1': 'Unknown', 'name': 'Unknown'}]"
movies_df["spoken_languages"] = movies_df["spoken_languages"].apply(safe_literal_eval)


# # Building Functions


# In[106]:


def count_movies_by_original_languages(language: str):

    return {"number of movie": movies_df.loc[movies_df["original_language"] == language].shape[0]}


# In[107]:


def get_runtime_and_release_year(movie_title: str):

    runtime_movie = movies_df.loc[movies_df["title"] == movie_title, ["runtime", "release_year"]].values[0][0]
    release_year_movie = movies_df.loc[movies_df["title"] == movie_title, ["runtime", "release_year"]].values[0][1]
    return {"Duracion": runtime_movie, "Año": release_year_movie}




# In[109]:


def get_collection_information_by_title(title1):

    # Getting the collection id
    id_of_collection = movies_df.loc[movies_df["title"] == title1, "collection_id"].tolist()[0]

    # Usage of the collection id in the collections and movies merge
    movies_in_collection = collections_df.merge(movies_df[["collection_id", "title", "revenue"]], on="collection_id")

    # Getting name, number of movies, total revenue, and mean revenue for output
    collection_name = movies_in_collection[movies_in_collection["collection_id"] == id_of_collection]["name"].unique()[0]
    number_of_movies = movies_in_collection[movies_in_collection["collection_id"] == id_of_collection].shape[0]
    total_revenue = movies_in_collection[movies_in_collection["collection_id"] == id_of_collection]["revenue"].sum()
    mean_revenue = np.mean(movies_in_collection[movies_in_collection["collection_id"] == id_of_collection]["revenue"])

    return {"Collection name": collection_name, "number_of_movies": number_of_movies, "total_revenue": total_revenue, "mean_revenue": mean_revenue}




# In[111]:


def number_of_movies_produced_in_country(country_name):
    
    number_of_movies =  movies_df[movies_df["production_countries"].apply(lambda country: country_name in {item["name"] for item in country})].shape[0]

    return {"country_name": country_name, "numbers_of_movies_produced": number_of_movies}


# In[112]:


def production_company_success(production_company_name):

    production_companies = movies_production_companies_df.merge(movies_df[["id","revenue"]], on = "id").merge(production_companies_df[["company_id"]], on="company_id")
    revenue = production_companies.loc[(company_demo["company_name"] == production_company_name), "revenue"].sum()
    number_of_movies = production_companies.loc[(company_demo["company_name"] == production_company_name)].shape[0]

    return {"production_company_name": production_company_name, "total_revenue": revenue, "number_of_movies_produced": number_of_movies}






# In[116]:


def director_success(director_name):
    
    mask_director = (credits_parquet.merge(movies_df[["id"]], on="id")
                    .loc[:, "directors"].apply(lambda directors: director_name in {director["name"] for director in directors}))
    
    avg_roi = np.mean(credits_parquet.merge(movies_df[["id","return_on_investment"]], on="id").loc[mask_director, "return_on_investment"])
    
    movies = (credits_parquet.merge(movies_df[["id", "title", "release_date", "return_on_investment", "budget", "revenue"]], on="id")
              .loc[mask_director, ["title", "release_date", "return_on_investment", "budget", "revenue"]].to_dict(orient="records"))

    
    return {"director_name": director_name, "avg_roi": avg_roi, "movies": movies}

