from fastapi import FastAPI
from etl import count_movies_by_original_languages, get_runtime_and_release_year, get_collection_information_by_title, number_of_movies_produced_in_country, production_company_success, director_success

app = FastAPI()

@app.get("/")
def welcome():
    print("Welcome to my API. Just add '/docs' to the url to get to know the end points :)")

@app.get("/count_of_movies_by_language/{lang}")
def demo_1(lang):
    return count_movies_by_original_languages(language=lang)
    

@app.get("/get_runtime_and_release_year_by_title/{title0}")
def demo_2(title0):
    return get_runtime_and_release_year(movie_title=title0)
    

@app.get("/get_collection_data_by_title/{title2}")
def demo_3(title2):
    return get_collection_information_by_title(title1=title2)

@app.get("/number_of_movies_produced_in_country/{country_name1}")
def demo_4(country_name1):
    return number_of_movies_produced_in_country(country_name=country_name1)
    
@app.get("/production_company_success/{company_name}")
def demo_5(company_name):
    return production_company_success(production_company_name=company_name)

@app.get("/director_success/{director_name1}")
def demo_6(director_name1):
    return director_success(director_name=director_name1)