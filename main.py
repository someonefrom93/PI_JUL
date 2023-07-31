from fastapi import FastAPI
from etl import count_movies_by_original_languages

app = FastAPI()

@app.get("/{lang}")
def demo_1(lang):
    return count_movies_by_original_languages(language=lang)

