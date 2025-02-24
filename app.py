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
    """
    Découpe le texte en chunks d'environ max_length caractères en gardant la cohérence des phrases.
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
    Lit le fichier site_content.txt, découpe le contenu en passages,
    génère les embeddings pour chaque passage et les insère dans ChromaDB.
    """
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
            embeddings.append([0] * 768)  # Valeur par défaut en cas d'erreur
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    st.success("La base vectorielle a été construite avec succès.")

# Si la collection est vide, on construit la base vectorielle
if len(collection.get()["ids"]) == 0:
    st.info("Construction de la base vectorielle...")
    build_vector_db()

def query_chatbot(user_message):
    """
    Génère l'embedding de la requête utilisateur, interroge ChromaDB pour récupérer
    les passages pertinents, puis envoie le prompt à ChatGPT.
    """
    # Générer l'embedding pour la requête
    query_embed_response = openai.Embedding.create(input=user_message, model="text-embedding-ada-002")
    query_embedding = query_embed_response["data"][0]["embedding"]
    
    # Recherche dans ChromaDB (les 3 passages les plus pertinents)
    query_result = collection.query(query_embeddings=[query_embedding], n_results=3)
    relevant_texts = " ".join(query_result["documents"][0])
    
    # Construction du prompt en fournissant le contexte en message système
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

# --- Interface Streamlit ---
st.title("Chatbot Intégré au Contenu du Site")

user_input = st.text_input("Posez votre question:")

if st.button("Envoyer"):
    if user_input:
        with st.spinner("Chargement de la réponse..."):
            answer = query_chatbot(user_input)
            st.markdown(f"**Réponse :** {answer}")
    else:
        st.warning("Veuillez entrer une question.")
