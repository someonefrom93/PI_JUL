from fastapi import FastAPI
from etl import count_movies_by_original_languages, get_runtime_and_release_year, get_collection_information_by_title

app = FastAPI()

@app.get("/count_of_movies_by_language/{lang}")
def demo_1(lang):
    return count_movies_by_original_languages(language=lang)
    


@app.get("/get_runtime_and_release_year_by_title/{title0}")
def demo_2(title0):
    return get_runtime_and_release_year(movie_title=title0)
    

@app.get("/get_collection_data_by_title/{title2}")
def demo_3(title2):
    return get_collection_information_by_title(title1=title2)
    