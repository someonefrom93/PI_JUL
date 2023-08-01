#!/usr/bin/env python
# coding: utf-8

# # ETL
# 

# In[3]:


import pandas as pd
import numpy as np
import ast


# In[4]:


#movies = pd.read_csv("data/movies_dataset.csv")
#credits = pd.read_csv("data/credits.csv")
#movies["popularity"] = movies["popularity"].astype(str)
#movies.to_parquet("parquet_data/movies_parquet.parquet")
#credits.to_parquet("parquet_data/credits_parquet.parquet")


# In[5]:


movies_parquet = pd.read_parquet("parquet_data/movies_parquet.parquet")


# In[6]:


movies_parquet.loc[711]


# In[7]:


movies_parquet.head()


# In[8]:


movies_df = movies_parquet.copy()


# ### Exploring movies data frame.
# 
# * Null Values
# * Data Types (Its homogeneity)

# In[9]:


# Null values
movies_df.isna().sum()


# In[10]:


# Data Types
movies_df.dtypes


# ### Dropping Fields
# Let's begin with the easiest ones: dropping fields! 

# In[11]:


movies_df.drop(["video", "imdb_id", "adult", "original_title", "poster_path", "homepage"], axis=1, inplace=True)


# In[12]:


movies_df.isnull().sum()


# ### Data Types Consistency
# Sometimes, there could be some unexpected data in a field, either a string datatype in a numeric field or a numeric dtype in a string field. It is important to check datatype consistency on each field. And the bellow ad-hoc function will help us to do that a bit easier. 
# 
# ##### What it does? 
# 
# Answer: It returns the indexes of intrusive rows in the series, which it'll be used as a mask. That will allow us to: 
#  * Have a visualization of the inconsistencies and 
#  * The rows position that we would like to impute them.

# In[13]:


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

# In[14]:


type(movies_df["release_date"][0])


# Once we get to know the dtype the data is readen, we can see the simple output of the function wich is only  the index position of all that rows with different data type from string

# In[15]:


# In the column release_date, the first five instrusive rows position are lited.
dtype_checker(movies_df, column="release_date", data_type=str)[:5] 


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

# In[16]:


# Spoting intrusive data
mask_for_date_intrusives = dtype_checker(movies_df, column="release_date", data_type=str)
# Dropping rows with nulls
movies_df.drop(mask_for_date_intrusives, inplace=True)


# In[17]:


# And, all that null or numeric field are imputed with this default value.
movies_df.head()


# In[18]:


movies_df["release_date"].head()


# But there are also "numbers" in this field, and I named numbers between parentesis because, literally, there could be numbers like "1", "2", "121", etc.. they are recognized from the dtype checker function as string, and we are still going to get issues when casting data as date type. See how we can spot those string numbers by regex expresion.

# In[19]:


regex_date = r"^\d{4}-\d{2}-\d{2}$" 
#In this lil line, by adding in the begining this ~ to our array of booleans, we get the opposite from pd.Series.str.contains(any_regex_expression)
movies_df.loc[~movies_df["release_date"].str.contains(regex_date)]


# In[20]:


movies_df.loc[~movies_df["release_date"].str.contains(regex_date)].index


# In[21]:


# So that lets impute these values as well.
movies_df.drop(movies_df.loc[~movies_df["release_date"].str.contains(regex_date)].index, inplace=True)


# In[22]:


# Trying if it works.
pd.to_datetime(movies_df["release_date"]).dt.year


# In[23]:


# And now, no further issues to cast this column as date type and grab the year only to create our year field.
movies_df['release_year'] = pd.to_datetime(movies_df["release_date"]).dt.year


# In[24]:


# lets take a look at the data frame movies
movies_df.head()


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

# In[25]:


# Revenue field data type
movies_df["revenue"].dtype


# In[26]:


movies_df["revenue"].isnull().sum()


# In[27]:


# Check how many rows aren't: float, int, str. Recall that movies_df has 45466 rows so far.
len(dtype_checker(movies_df, column="revenue", data_type=float)), len(dtype_checker(movies_df, column="revenue", data_type=int)), len(dtype_checker(movies_df, column="revenue", data_type=str))


# Ok so.. We have 0 values that aren't floats, 45466 aren't integers, 45466 aren't str, thus, all values are float. Good

# In[28]:


movies_df["revenue"] / 2


# In[29]:


# Budget field data type
movies_df["budget"].dtype


# In[30]:


movies_df['budget'].isnull().sum()


# In[31]:


len(dtype_checker(movies_df, column="budget", data_type=float)), len(dtype_checker(movies_df, column="budget", data_type=int)), len(dtype_checker(movies_df, column="budget", data_type=str))


# Ok so.. We have 45466 values that aren't floats, 45466 aren't integers, 0 aren't str, thus, all values are string. Not too good. Let's just try to cast this as float and see what will happen 

# In[32]:


movies_df["budget"].astype(float)


# In[33]:


movies_df["budget"] = movies_df["budget"].astype(float)


# In[34]:


movies_df["budget"].dtype, movies_df["budget"].isnull().sum()


# ### Creating "return_on_investment" (ROI) fiel
# 
# This is what is performed.. divide revenue by budget as float, then fill null values with zero.. after filling nulls replace inifinites by zeros.

# In[35]:


movies_df["revenue"].div(movies_df["budget"].astype(float)).fillna(0).replace([np.inf, -np.inf], 0)


# In[36]:


movies_df["return_on_investment"] = movies_df["revenue"].div(movies_df["budget"].astype(float)).fillna(0).replace([np.inf, -np.inf], 0)


# In[37]:


movies_df.head()


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

# In[38]:


movies_df.loc[:,"belongs_to_collection"].dtype


# In[39]:


# Building the new data frame for belongs_to_collection
belongs_to_collections_df = movies_df.loc[movies_df["belongs_to_collection"].isnull() == False, "belongs_to_collection"]
belongs_to_collections_df.shape


# Since belongs_to_collection only contains "object" types, we need to convert those pandas' object types into dictionary by `ast.literal_eval()` wich grabs the string and catch it as the data format that seems to be. 
# 
# For example: "{'hello': 2}" string --> {'hello': 2} dict. And it will enable all the dictionary methods needed for this data manipulation

# In[40]:


# This functionn is only for handling any expected error and impute with default empty list
def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return {}


# In[41]:


belongs_to_collections_df = belongs_to_collections_df.apply(safe_literal_eval)


# In[42]:


belongs_to_collections_df.head()[0]


# In[43]:


movies_df.loc[dtype_checker(movies_df, "release_date", str), "release_date"]


# In[44]:


dtype_checker(belongs_to_collections_df, None, dict)


# In[45]:


belongs_to_collections_df.loc[dtype_checker(belongs_to_collections_df, None, dict)]


# In[46]:


belongs_to_collections_df.drop(dtype_checker(belongs_to_collections_df, None, dict), inplace=True)


# In[47]:


belongs_to_collections_df


# In[48]:


collections_df = pd.DataFrame(belongs_to_collections_df.tolist())


# In[49]:


collections_df


# In[50]:


movies_df.head(2)


# In[51]:


movies_df.drop(["belongs_to_collection"], axis=1, inplace=True)


# In[52]:


movies_df.head(2)


# In[53]:


pd.merge(collections_df, movies_df, left_index=True, right_index=True, how="inner").head(2)


# In[54]:


movies_df["genres"].apply(safe_literal_eval)


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

# In[55]:


movies_df.head(2)


# In[56]:


movies_df["genres"][0]


# In[57]:


empty_list_pattern = r'^\[\]$'
genre_empy_mask = movies_df["genres"].astype(str).str.match(empty_list_pattern)
print(genre_empy_mask.sum())
movies_df[genre_empy_mask].head(2)


# In[58]:


movies_df.loc[genre_empy_mask, "genres"] = "[{'id': 123456, 'name': 'Unknown'}]"


# In[59]:


movies_df.loc[genre_empy_mask, "genres"]


# In[60]:


movies_df["genres"] = movies_df["genres"].apply(safe_literal_eval)


# In[61]:


movies_df["genres"][0]


# In[62]:


movies_df.explode('genres', ignore_index=True).head(2)#[["genres"]].rename(columns={"genres": "genres_info"})


# In[63]:


movies_df.explode('genres', ignore_index=True).tail(2)#[["genres"]].rename(columns={"genres": "genres_info"})


# In[64]:


genres_df = movies_df.explode('genres', ignore_index=True)[["genres"]].rename(columns={"genres": "genres_info"})


# In[65]:


genres_df


# In[66]:


genres_df["genre_id"] = genres_df["genres_info"].apply(lambda genre: genre["id"])
genres_df["genre_id"] = genres_df["genre_id"].astype(str)

genres_df["genre_name"] = genres_df["genres_info"].apply(lambda genre: genre["name"])
genres_df.drop(columns="genres_info", inplace=True)
genres_df


# In[67]:


genres_df.drop_duplicates(subset=["genre_id"], keep="first", inplace=True)


# In[68]:


movies_genres_df = movies_df.explode('genres', ignore_index=True)[['id', 'genres']].rename(columns={'genres': 'genre_info'})
movies_genres_df["genre_id"] = movies_genres_df["genre_info"].apply(lambda genre: genre["id"])
movies_genres_df["genre_id"] = movies_genres_df["genre_id"].astype(str)


# In[69]:


movies_genres_df.drop(columns="genre_info", inplace=True)


# In[70]:


movies_genres_df.merge(movies_df[["id", "title"]], on='id').merge(genres_df, on="genre_id")


# In[71]:


movies_df.drop(labels=["genres"], axis=1, inplace=True)


# In[72]:


movies_df.columns


# ### Production Companies Field

# In[73]:


movies_df["production_companies"][3]


# In[74]:


empty_list_pattern = r'^\[\]$'
print(movies_df["production_companies"].str.match(empty_list_pattern).sum())
production_movies_mask = movies_df["production_companies"].str.match(empty_list_pattern)
production_movies_mask


# In[75]:


movies_df[production_movies_mask].head(2)


# In[76]:


movies_df.loc[production_movies_mask, "production_companies"] = "[{'name': 'Unknown', 'id': 123456}]"


# In[77]:


movies_df.loc[production_movies_mask, "production_companies"]


# In[78]:


movies_df["production_companies"] = movies_df["production_companies"].apply(safe_literal_eval)


# In[79]:


production_companies_df = movies_df.explode("production_companies", ignore_index=True)[["production_companies"]].rename(columns={"production_companies": "production_companies_info"})


# In[80]:


production_companies_df["company_name"] = production_companies_df["production_companies_info"].apply(lambda prod_company: prod_company["name"])
production_companies_df["company_id"] = production_companies_df["production_companies_info"].apply(lambda prod_company: prod_company["id"])
production_companies_df.drop("production_companies_info", axis=1, inplace=True)


# In[81]:


production_companies_df.drop_duplicates(subset=["company_id"], keep="first", inplace=True)


# In[82]:


production_companies_df.dtypes


# In[83]:


production_companies_df["company_id"] = production_companies_df["company_id"].astype(str)


# In[84]:


production_companies_df.dtypes


# DataFrame for the many-to-many relationship

# In[85]:


movies_production_companies_df = movies_df.explode("production_companies", ignore_index=True)[["id", "production_companies"]].rename(columns={"production_companies": "production_companies_info"})
movies_production_companies_df["company_id"] = movies_production_companies_df["production_companies_info"].apply(lambda prod_company: prod_company["id"])
movies_production_companies_df.drop("production_companies_info", axis=1, inplace=True)
movies_production_companies_df["company_id"] = movies_production_companies_df["company_id"].astype(str)
movies_production_companies_df.head()


# In[86]:


movies_production_companies_df.head(2)


# ### Trying it out

# In[87]:


company_demo = movies_production_companies_df.merge(movies_df[["id", "title"]], on = "id").merge(production_companies_df, on="company_id")


# In[88]:


company_demo.loc[(company_demo["title"].str.contains("The")) & (company_demo["company_name"] == "Pixar Animation Studios")]


# In[89]:


movies_df.drop("production_companies", axis=1, inplace=True)


# ### Production Companies Field
# 
# For only learning purpuses, the approach to tackle this down will be completely different from normalizing data and building tables. Here the approach is going to be the accesing approach by `.apply()` method and lambda expressions against nested data. This leverage the processecing resources rather than the store recourses because we aren't going to create any additionally dataframe.
# 
# How it works?
# 
# * First of all, we should take care about missing value, in this case they are represented as empy list and levearing this data is currently in string datatype, it will be easier to spot those empty list with regex patterns and then impute a default value to simply label them as "unknowns"
# * Once we get all rows in production_companies field with an a list with at least one dictionary object, we should be able to apply our safe_literal_eval to convert those strings into the appropiate objects
# * Then, we could filter information over these nested data (dictionaries in a list) with `.apply()` method wich applies any function on each record. We can use eaither custom funcions (for example, safe_literal_eval is one custom function applied on each record in the production_companies field) or lamba expressions.

# In[90]:


movies_df.head(2)


# In[91]:


movies_df["production_countries"].head(10)


# In[92]:


movies_df["production_countries"].str.match(empty_list_pattern).sum()
production_countries_empties_mask = movies_df["production_countries"].str.match(empty_list_pattern)


# In[93]:


movies_df.loc[production_countries_empties_mask, "production_countries"] = "[{'iso_3166_1': 'Unknown', 'name': 'Unknown'}]"


# In[94]:


movies_df.loc[production_countries_empties_mask, "production_countries"]


# In[95]:


movies_df["production_countries"] = movies_df["production_countries"].apply(safe_literal_eval)


# In[96]:


movies_df.head(1)


# In[97]:


movies_df.loc[(movies_df["production_countries"].apply(lambda country: "Mexico" in {item["name"] for item in country}))
              & (movies_df["original_language"] == "en")
              & (movies_df["production_countries"].apply(lambda country_len: len(country_len) > 1))
                , ["original_language", "title", "production_countries"]]


# In[98]:


movies_df.loc[(movies_df["production_countries"].apply(lambda country: "Mexico" in {item["name"] for item in country}))
              
              & (movies_df["production_countries"].apply(lambda country_len: len(country_len) == 1))
                , ["id", "original_language", "title", "production_countries"]]


# # Credits DataSet
# 
# Inconclude

# In[100]:


# credits_parquet = pd.read_parquet("parquet_data/credits_parquet.parquet")


# In[121]:


# import sys
# sys.getsizeof(credits_parquet) / 1000000


# In[120]:


# credits_parquet.dtypes


# # Spoken Language field

# In[122]:


movies_df["spoken_languages"][1]


# In[123]:


spoken_languages_mask = movies_df["spoken_languages"].str.match(empty_list_pattern)


# In[124]:


movies_df.loc[spoken_languages_mask, "spoken_languages"] = "[{'iso_639_1': 'Unknown', 'name': 'Unknown'}]"
movies_df["spoken_languages"] = movies_df["spoken_languages"].apply(safe_literal_eval)


# # Building Functions

# In[125]:


movies_df.head(1)


# In[126]:


def count_movies_by_original_languages(language: str):

    return {"number of movie": movies_df.loc[movies_df["original_language"] == language].shape[0]}


# In[155]:


def get_runtime_and_release_year(movie_title: str):

    runtime_movie = movies_df.loc[movies_df["title"] == movie_title, ["runtime", "release_year"]].values[0][0]
    release_year_movie = movies_df.loc[movies_df["title"] == movie_title, ["runtime", "release_year"]].values[0][1]
    return {"Duracion": runtime_movie, "Año": release_year_movie}


# In[157]:


# get_runtime_and_release_year("Grumpier Old Men")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[399]:


# get_ipython().system('jupyter nbconvert --to script etl.ipynb')

