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

# --- Interface Chat style ChatGPT ---
st.set_page_config(page_title="Chatbot", layout="wide")

# Injections CSS pour le style de chat
st.markdown(
    """
    <style>
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 10px;
    }
    .message {
        padding: 8px 15px;
        margin: 5px 0;
        border-radius: 15px;
        max-width: 70%;
        line-height: 1.5;
        font-size: 16px;
    }
    .user {
        background-color: #DCF8C6;
        margin-left: auto;
        text-align: right;
    }
    .bot {
        background-color: #E1E1E1;
        margin-right: auto;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Chatbot Intégré au Contenu du Site")

# Initialiser l'historique des messages dans le state
if "messages" not in st.session_state:
    st.session_state.messages = []

def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Zone de chat
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role_class = "user" if msg["role"] == "user" else "bot"
        st.markdown(f'<div class="message {role_class}"><strong>{msg["role"].capitalize()} :</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Formulaire d'envoi
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Votre message :", "")
    submit_button = st.form_submit_button(label="Envoyer")

if submit_button and user_input:
    # Ajouter le message de l'utilisateur à l'historique
    add_message("user", user_input)
    with st.spinner("Chargement de la réponse..."):
        answer = query_chatbot(user_input)
    add_message("bot", answer)
    # Rafraîchir la page pour afficher le nouveau message
    st.experimental_rerun()
