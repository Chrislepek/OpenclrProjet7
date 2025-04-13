import streamlit as st
import requests

# URL de base de votre API FastAPI
BASE_URL = "http://127.0.0.1:8000"  
st.title("Prédiction de Défaut de Client")

# Entrée utilisateur
client_id = st.number_input("Entrez le Client ID", min_value=0, step=1)

# Bouton pour déclencher la prédiction
if st.button("Prédire"):
    # Appeler l'API FastAPI
    response = requests.get(f"{BASE_URL}/predict/{client_id}")

    if response.status_code == 200:
        result = response.json()
        st.write(f"Client ID: {result['client_id']}")
        st.write(f"Probabilité de défaut: {result['probabilité']:.4f}")
        st.write(f"Classe prédite: {result['classe']}")
    elif response.status_code == 404:
        st.error("Client non trouvé")
    else:
        st.error(f"Erreur: {response.status_code}")