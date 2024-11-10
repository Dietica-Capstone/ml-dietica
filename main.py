from fastapi import FastAPI
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

dataset_nutrisi = pd.read_csv('./dataset_nutrisi.csv')
extracted_data = pd.read_csv('./extracted_data.csv')

scaler = MinMaxScaler()
scaler.fit(dataset_nutrisi)
dataset_nutrisi_scaled = scaler.fit_transform(dataset_nutrisi)

application = FastAPI()
model = load_model('./model_rekomendasi.h5')

@application.get('/')
def dietica():
    return 'dietica'

@application.post('/recommendation')
def food_recommendation(cal:int, fat:int,sat:int, chol:int, sodium:int, carbo:int, fiber:int, sugar:int, pro:int):
    input_nutrisi = np.array([[cal, fat, sat, chol, sodium, carbo, fiber, sugar, pro]])
    input_nutrisi_scaled = scaler.transform(input_nutrisi)
    predicted_latent_feature = model.predict(dataset_nutrisi_scaled)
    similarity = cosine_similarity(input_nutrisi_scaled, predicted_latent_feature)
    top_resep = np.argsort(similarity, axis=1)[0][::-1][:10]
    kolom_pilihan = ['Name', 'Images', 'RecipeIngredientParts',
                 'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                 'SodiumContent', 'CarbohydrateContent', 'FiberContent',
                 'SugarContent', 'ProteinContent', 'RecipeServings', 'RecipeInstructions']
    recommendations = extracted_data.iloc[top_resep][kolom_pilihan]
    rekomendasi = recommendations.head()
    return rekomendasi

