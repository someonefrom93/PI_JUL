from fastapi import FastAPI
from etl import count_movies_by_original_languages, get_runtime_and_release_year

app = FastAPI()

@app.get("/count_of_movies_by_language/{lang}")
def demo_1(lang):
    return count_movies_by_original_languages(language=lang)

@app.get("/get_runtime_and_release_year_by_title/{title}")
def demo_2(title):
    return get_runtime_and_release_year(movie_title=title)

