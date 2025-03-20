import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3

import os
import re
import openai
import streamlit as st
from dotenv import load_dotenv
import chromadb
import tempfile
import requests

# Positionnez st.set_page_config avant toute autre instruction Streamlit
st.set_page_config(page_title="Chatbot Intégré", layout="wide")

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
    """
    Lit les fichiers site_content.txt et document_centralise.txt, 
    découpe le contenu, génère les embeddings et les indexe dans ChromaDB.
    """
    try:
        with open("site_content.txt", "r", encoding="utf-8") as f:
            content1 = f.read()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier site_content.txt : {e}")
        content1 = ""
    try:
        with open("document_centralise.txt", "r", encoding="utf-8") as f:
            content2 = f.read()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier document_centralise.txt : {e}")
        content2 = ""
    if not content1 and not content2:
        st.error("Aucun contenu disponible pour construire la base vectorielle.")
        return

    combined_content = content1 + "\n" + content2
    chunks = split_text(combined_content)
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

# Construire la base vectorielle si elle est vide
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
    
    # Construction du prompt en fournissant le contexte
    messages = [
        {"role": "system", "content": f"Les informations suivantes proviennent du site et du document centralisé :\n{relevant_texts}"},
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

# Configuration de la page Streamlit
st.title("Chatbot Intégré au Contenu du Site")
st.write("Posez vos questions ci-dessous:")

# Choix du mode de chat : Texte ou Vocal
mode = st.radio("Choisissez le mode de chat :", options=["Texte", "Vocal"])

# Initialiser l'historique des messages si nécessaire
if "messages" not in st.session_state:
    try:
        with open("instructions.txt", "r", encoding="utf-8") as file:
            file_content = file.read()
    except Exception as e:
        file_content = ""
    st.session_state.messages = [
        {"role": "system", "content": "Tu es un chatbot qui répond aux questions des clients potentiels du chirurgien esthétique Dr. Laurent Bendadiba en te basant sur le contenu de son site et du document centralisé. Ton objectif est d'aiguiller les clients avec des réponses relativement courtes, tout en les orientant progressivement vers une prise de rendez-vous. N'insiste pas trop sur la prise de rendez-vous, mais encourage le client s'il semble intéressé. Si l'utilisateur aborde un sujet non lié à la chirurgie ou à la médecine, recentre la conversation."},
        {"role": "user", "content": f"Voici le contenu du site et du document centralisé :\n{file_content}"}
    ]

if mode == "Texte":
    # Affichage de l'historique des messages (à partir du 3e message)
    for message in st.session_state.messages[2:]:
        st.chat_message(message["role"]).markdown(message["content"])
    
    if prompt := st.chat_input("Votre texte ici :"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                answer = query_chatbot(prompt)
                full_response = answer
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif mode == "Vocal":
    uploaded_audio = st.file_uploader("Enregistrez ou téléversez un fichier audio", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/mp3")
        # Sauvegarde temporaire du fichier audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(uploaded_audio.getvalue())
            temp_audio_path = temp_audio.name
        try:
            with open(temp_audio_path, "rb") as audio_file:
                transcription = openai.Audio.transcribe("whisper-1", audio_file)
            user_message = transcription['text']
            st.write(f"**Transcription :** {user_message}")
        except Exception as e:
            st.error(f"Erreur lors de la transcription de l'audio : {e}")
            user_message = ""
        
        if user_message:
            try:
                answer = query_chatbot(user_message)
                st.write("**Réponse :**")
                st.write(answer)
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {e}")
            
            # Si la clé API ElevenLabs est configurée, générer et diffuser la réponse audio
            elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
            if elevenlabs_api_key:
                voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
                headers = {
                    "xi-api-key": elevenlabs_api_key,
                    "Content-Type": "application/json",
                }
                data = {
                    "text": f"<lang=fr>{answer}</lang>",
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}
                }
                try:
                    audio_response = requests.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                        headers=headers,
                        json=data
                    )
                    if audio_response.status_code == 200:
                        st.audio(audio_response.content, format="audio/mp3")
                    else:
                        st.error(f"Erreur ElevenLabs : {audio_response.status_code}")
                except Exception as e:
                    st.error(f"Erreur lors de la génération de la réponse audio : {e}")
            else:
                st.info("Clé API ElevenLabs non configurée, réponse textuelle uniquement.")
