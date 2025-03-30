
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd
import threading


# Chemin  vers les fichiers
CSV_PATH = './small_data.csv'
PIPELINE_PATH = './pipeline.joblib'

# Charger le fichier CSV
df = pd.read_csv(CSV_PATH)

# Charger le pipeline
with open(PIPELINE_PATH, 'rb') as f:
    pipeline = joblib.load(f)

# FastAPI app
app = FastAPI()

#Fonction  pour récupérer les données d'un client
def get_client_data(client_id: int):
    client_row = df[df['SK_ID_CURR'] == client_id]

    if client_row.empty:
        return None  # Client ID non trouvé

    # Extraire les features
    features = client_row.drop(['SK_ID_CURR'], axis=1).values[0]  # - 'client_id'
    return features

@app.get("/")
def read_root():
    return {"message": "API de prédiction de défaut de client"}

@app.get("/predict/{client_id}")
def predict(client_id: int):
    # Vérifier si le client existe
    client_features = get_client_data(client_id)

    if client_features is None:
        raise HTTPException(status_code=404, detail="Client not found")

    # Préparer les données pour la prédiction
    data = np.array([client_features])

    # Prédiction du pipeline
    probabilité = pipeline.predict_proba(data)[:, 1]  # Probabilité de la classe "défaut"

    # Appliquer le seuil optimisé (par exemple, 0.5)
    seuil = 0.06
    classe = "Accepté" if probabilité >= seuil else "Refusé"

    return {"client_id": client_id, "probabilité": probabilité[0], "classe": classe}