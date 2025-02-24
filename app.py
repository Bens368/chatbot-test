# Forcer l'utilisation de pysqlite3 à la place de sqlite3
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3

import os
import re
import openai
import streamlit as st
from dotenv import load_dotenv
import chromadb

# Charger les variables d'environnement depuis .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialisation du client Chroma avec le répertoire de persistance
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("site_content")

def split_text(text, max_length=500):
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
    try:
        with open("site_content.txt", "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
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
            st.error(f"Erreur lors de la génération de l'embedding pour un chunk : {e}")
            embeddings.append([0] * 1536)  # Ajustez la taille en fonction du modèle
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    st.success("La base vectorielle a été construite avec succès.")

if len(collection.get()["ids"]) == 0:
    st.info("Construction de la base vectorielle...")
    build_vector_db()

def query_chatbot(user_message):
    query_embed_response = openai.Embedding.create(input=user_message, model="text-embedding-ada-002")
    query_embedding = query_embed_response["data"][0]["embedding"]
    
    query_result = collection.query(query_embeddings=[query_embedding], n_results=3)
    relevant_texts = " ".join(query_result["documents"][0])
    
    messages = [
        {"role": "system", "content": f"Les informations suivantes proviennent d'un site web :\n{relevant_texts}"},
        {"role": "user", "content": user_message}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    answer = response['choices'][0]['message']['content'].strip()
    return answer

st.title("Chatbot Intégré au Contenu du Site")
user_input = st.text_input("Posez votre question:")

if st.button("Envoyer"):
    if user_input:
        with st.spinner("Chargement de la réponse..."):
            answer = query_chatbot(user_input)
            st.markdown(f"**Réponse :** {answer}")
    else:
        st.warning("Veuillez entrer une question.")
