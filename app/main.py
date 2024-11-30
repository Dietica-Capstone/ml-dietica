from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd

# Load dataset dan model
dataset_nutrisi = pd.read_csv('./dataset_nutrisi.csv')
extracted_data = pd.read_csv('./extracted_data.csv')

scaler = MinMaxScaler()
scaler.fit(dataset_nutrisi)
dataset_nutrisi_scaled = scaler.fit_transform(dataset_nutrisi)

model = load_model('./model_rekomendasi_v2.h5')
model_kalori = load_model('./model_rekomendasi_kalori.h5')
predicted_latent_feature = model.predict(dataset_nutrisi_scaled)

# Inisialisasi FastAPI
application = FastAPI()

# Model untuk input data nutrisi
class NutritionalInput(BaseModel):
    Calories: int
    FatContent: int
    SaturatedFatContent: int
    CholesterolContent: int
    SodiumContent: int
    CarbohydrateContent: int
    FiberContent: int
    SugarContent: int
    ProteinContent: int

# Endpoint utama
@application.get("/")
def index():
    return {"message": "DIETICA API"}

# Rekomendasi berdasarkan nutrisi
@application.post("/recommendation")
def food_recommendation(input_data: NutritionalInput):
    # Ambil input data
    input_nutrisi = np.array([[input_data.Calories, input_data.FatContent, input_data.SaturatedFatContent,
                               input_data.CholesterolContent, input_data.SodiumContent, input_data.CarbohydrateContent,
                               input_data.FiberContent, input_data.SugarContent, input_data.ProteinContent]])

    # Skala data input
    input_nutrisi_scaled = scaler.transform(input_nutrisi)

    # Hitung similarity
    similarity = cosine_similarity(input_nutrisi_scaled, predicted_latent_feature)

    # Ambil indeks resep teratas
    top_resep = np.argsort(similarity, axis=1)[0][::-1][:10]

    # Pilih kolom yang relevan
    kolom_pilihan = ['Name', 'Images', 'RecipeIngredientParts',
                     'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                     'SodiumContent', 'CarbohydrateContent', 'FiberContent',
                     'SugarContent', 'ProteinContent', 'RecipeServings', 'RecipeInstructions']
    recommendations = extracted_data.iloc[top_resep][kolom_pilihan]

    # Konversi hasil ke format dictionary
    rekomendasi = recommendations.to_dict(orient="records")

    return {"message": "Predict success!", "status": 200, "data": rekomendasi}

# Rekomendasi berdasarkan kalori
@application.post("/recommendation_calories")
def recommend_calories(calories: float):
    # Prediksi dengan model kalori
    prediction = model_kalori.predict(np.array([[calories]]))
    
    # Hitung jarak Euclidean
    distances = euclidean_distances(extracted_data[['Calories']], prediction)
    extracted_data['Distance'] = distances
    
    # Urutkan dan ambil 15 resep teratas
    sorted_recipes = extracted_data.sort_values('Distance').head(15)
    relevant_columns = ['Name', 'Images', 'Calories', 'FatContent', 'SaturatedFatContent',
                        'CholesterolContent', 'SodiumContent', 'CarbohydrateContent',
                        'FiberContent', 'SugarContent', 'ProteinContent']
    top_recipes = sorted_recipes[relevant_columns]
    top_recipes_dict = top_recipes.to_dict(orient="records")

    return {"message": "Predict success!", "status": 200, "data": top_recipes_dict}

# Mencari resep berdasarkan nama
@application.post("/search")
def search_recipe(name: str):
    # Cari berdasarkan kata kunci dalam nama
    search_results = extracted_data[extracted_data['Name'].str.contains(name, case=False, na=False)]

    if search_results.empty:
        raise HTTPException(status_code=404, detail="No recipes found")

    # Konversi hasil ke format dictionary
    search_data = search_results.to_dict(orient="records")

    return {"message": "Search success!", "status": 200, "data": search_data}

# Mendapatkan data acak
@application.get("/getrecipe")
def get_recipe():
    getrecipe = extracted_data.sample(10)
    data_resep = getrecipe.to_dict(orient="records")

    return {"message": "Get success!", "status": 200, "data": data_resep}