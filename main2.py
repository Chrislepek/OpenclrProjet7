
import streamlit as st
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd
import threading


# Chemin  vers les fichiers
CSV_PATH = 'C:/Users/Win10/Documents/OpenclrProjet7/small_data.csv'
PIPELINE_PATH = 'C:/Users/Win10/Documents/OpenclrProjet7/pipeline.joblib'

# Charger le fichier CSV
df = pd.read_csv(CSV_PATH)

# Charger le pipeline
with open(PIPELINE_PATH, 'rb') as f:
    pipeline = joblib.load(f)

#Fonction  pour récupérer les données d'un client
def get_client_data(client_id: int):
    client_row = df[df['SK_ID_CURR'] == client_id]

    if client_row.empty:
        return None  # Client ID non trouvé

    # Extraire les features
    features = client_row.drop(['SK_ID_CURR'], axis=1).values[0]  # - 'client_id'
    return features

# Interface Streamlit
st.title("Prédiction de Défaut de Client")

# Saisie du client_id
client_id = st.number_input("Entrez le Client ID", min_value=0, step=1)

if st.button("Prédire"):
    client_features = get_client_data(client_id)

    if client_features is not None:
        # Préparer les données pour la prédiction
        data = np.array([client_features])

        # Prédiction du pipeline
        probabilité = pipeline.predict_proba(data)[:, 1]  # Probabilité de la classe "défaut"

        # Appliquer le seuil optimisé (par exemple, 0.06)
        seuil = 0.06
        classe = "Accepté" if probabilité <= seuil else "Refusé"

        st.write(f"Client ID: {client_id}")
        st.write(f"Probabilité de défaut: {probabilité[0]:.4f}")
        st.write(f"Classe: {classe}")
    else:
        st.error(f"Client {client_id} not found")