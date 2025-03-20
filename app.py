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
    """Découpe le texte en chunks d'environ max_length caractères en conservant la cohérence des phrases."""
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
    """Lit le fichier site_content.txt, découpe le contenu, génère les embeddings et les indexe dans ChromaDB."""
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
            embeddings.append([0] * 1536)
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    st.success("La base vectorielle a été construite avec succès.")

# Construire la base vectorielle si elle n'existe pas déjà
if len(collection.get()["ids"]) == 0:
    st.info("Construction de la base vectorielle...")
    build_vector_db()

def query_chatbot(user_message):
    """
    Génère l'embedding de la requête utilisateur, interroge ChromaDB pour récupérer les passages pertinents,
    puis envoie le prompt à ChatGPT pour générer une réponse.
    """
    # Générer l'embedding de la requête
    query_embed_response = openai.Embedding.create(input=user_message, model="text-embedding-ada-002")
    query_embedding = query_embed_response["data"][0]["embedding"]
    
    # Récupérer les 3 passages les plus pertinents
    query_result = collection.query(query_embeddings=[query_embedding], n_results=3)
    relevant_texts = " ".join(query_result["documents"][0])
    
    # Construire le prompt avec le contexte
    messages = [
        {"role": "system", "content": f"Les informations suivantes proviennent d'un site web :\n{relevant_texts}"},
        {"role": "user", "content": user_message}
    ]
    
    # Appeler ChatGPT en mode streaming
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )
    full_response = ""
    for chunk in response:
        content = chunk.choices[0].delta.get("content", "")
        full_response += content
    return full_response

# Configuration de la page
st.set_page_config(page_title="Chatbot Intégré", layout="wide")
st.title("Chatbot Intégré au Contenu du Site")
st.write("Posez vos questions ci-dessous:")

# Initialiser l'historique des messages si nécessaire
if "messages" not in st.session_state:
    # Charger le contenu des instructions (si disponible)
    try:
        with open("instructions.txt", "r", encoding="utf-8") as file:
            file_content = file.read()
    except Exception as e:
        file_content = ""
    st.session_state.messages = [
        {"role": "system", "content": "Tu es un chatbot qui répond aux questions des clients potentiels du chirurgien esthétique Dr. Laurent Bendadiba en te basant sur le contenu de son site web. Ton objectif est d'aiguiller les clients avec des réponses relativement courtes, ainsi que d'essayer de les faire convertir en les amenant au fur et à mesure vers une prise de rendez-vous. N'insiste pas trop sur la prise de rendez-vous, mais encourage le client si tu sens qu'il est intéressé. Fais comme si tu étais intégré au site web, donc ne propose pas d'aller sur le site web puisque le client y est déjà. Si l'utilisateur parle d'un autre sujet que la chirurgie ou la médecine, ramène-l'au sujet principal."},
        {"role": "user", "content": f"Voici le contenu du site :\n{file_content}"}
    ]

# Afficher les messages existants dans l'historique (à partir du 3e message)
for message in st.session_state.messages[2:]:
    st.chat_message(message["role"]).markdown(message["content"])

# Saisie utilisateur
if prompt := st.chat_input("Votre texte ici :"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    
    # Traitement de la réponse de l'assistant
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            answer = query_chatbot(prompt)
            full_response = answer
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
