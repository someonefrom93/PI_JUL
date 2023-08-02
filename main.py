from fastapi import FastAPI
from etl import count_movies_by_original_languages, get_runtime_and_release_year, get_collection_information_by_title

app = FastAPI()

@app.get("/count_of_movies_by_language/{lang}")
def demo_1(lang):
    try:
        count_movies_by_original_languages(language=lang)
    except (ValueError):
        return "Please provide only strings"


@app.get("/get_runtime_and_release_year_by_title/{title0}")
def demo_2(title0):
    try:
        get_runtime_and_release_year(movie_title=title0)
    except (IndexError, ValueError):
        return "No movie found"

@app.get("/get_collection_data_by_title/{title2}")
def demo_3(title2):
    try:
        get_collection_information_by_title(title1=title2)
    except (IndexError, ValueError):
        return "No movie found"