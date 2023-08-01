from fastapi import FastAPI
from etl import count_movies_by_original_languages, get_runtime_and_release_year

app = FastAPI()

@app.get("/{lang}")
def demo_1(lang):
    return count_movies_by_original_languages(language=lang)

@app.get("/{title}")
def demo_2(title):
    return get_runtime_and_release_year(movie_title=title)

