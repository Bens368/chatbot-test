import os
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import re

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
CORS(app)  # Autorise les requêtes cross-origin

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Configuration et initialisation de ChromaDB ---
from chromadb.config import Settings
import chromadb

import chromadb

# Remplacer l'initialisation actuelle du client Chroma par :
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("site_content")


def split_text(text, max_length=500):
    """
    Découpe le texte en passages d'environ max_length caractères.
    On utilise ici une découpe basée sur les phrases pour conserver la cohérence.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def build_vector_db():
    """
    Lit le fichier site_content.txt, le découpe en passages, génère les embeddings pour chacun
    et les insère dans ChromaDB.
    """
    try:
        with open("site_content.txt", "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return
    
    chunks = split_text(content)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = []
    for chunk in chunks:
        try:
            embed_response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
            embedding = embed_response["data"][0]["embedding"]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Erreur lors de la génération de l'embedding : {e}")
            embeddings.append([0]*768)  # Valeur par défaut en cas d'erreur (à adapter)
    # Ajoute les passages à la collection
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    print("Base vectorielle construite avec succès.")

# Construire la base vectorielle si elle est vide
if len(collection.get()["ids"]) == 0:
    build_vector_db()

# --- API Flask ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Message manquant'}), 400

    user_message = data['message']
    
    try:
        # Générer l'embedding de la requête utilisateur
        query_embed_response = openai.Embedding.create(input=user_message, model="text-embedding-ada-002")
        query_embedding = query_embed_response["data"][0]["embedding"]
        
        # Interroger ChromaDB pour récupérer les passages les plus pertinents
        query_result = collection.query(query_embeddings=[query_embedding], n_results=3)
        # Concaténer les passages trouvés pour constituer le contexte
        relevant_texts = " ".join(query_result["documents"][0])
        
        # Construire le prompt en fournissant le contexte (passages pertinents) en message système
        messages = [
            {"role": "system", "content": f"Les informations suivantes proviennent d'un site web :\n{relevant_texts}"},
            {"role": "user", "content": user_message}
        ]
        
        # Appeler l'API ChatCompletion de ChatGPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer = response['choices'][0]['message']['content'].strip()
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)