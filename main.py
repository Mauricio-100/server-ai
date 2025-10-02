import os
from dotenv import load_dotenv
import mysql.connector
from urllib.parse import urlparse

from fastapi import FastAPI
from pydantic import BaseModel

# --- Imports LangChain & LLM ---
import torch
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun

# === 1) Configuration et Chargement des Modèles ===
print("--- Démarrage de la phase de configuration ---")
load_dotenv()

# --- Configuration Base de Données ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Configuration des modèles ---
LLM_MODEL_NAME = "nickypro/tinyllama-15M"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Un modèle léger pour la recherche sémantique
CORPUS_FILE = "data.txt"

# --- Chargement du LLM (TinyLlama) ---
print(f"Chargement du LLM : {LLM_MODEL_NAME}...")
llm_pipeline = pipeline(
    "text-generation",
    model=LLM_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
llm = HuggingFacePipeline(pipeline=llm_pipeline, model_kwargs={'temperature': 0.7})
print("LLM chargé avec succès.")

# --- Préparation de la recherche dans data.txt (RAG) ---
print(f"Préparation de la base de connaissance depuis '{CORPUS_FILE}'...")
loader = TextLoader(CORPUS_FILE, encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = FAISS.from_documents(docs, embeddings)

# On crée une "chaîne" LangChain qui sait chercher dans data.txt et faire répondre le LLM
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
print("Base de connaissance prête.")

# --- Outil de recherche Web ---
web_search = DuckDuckGoSearchRun()

# === 2) Fonctions pour la Mémoire SQL ===
# NOTE: La base de données est maintenant un simple cache (plus besoin de vecteurs)
def get_db_connection():
    url = urlparse(DATABASE_URL)
    return mysql.connector.connect(
        host=url.hostname, user=url.username, password=url.password,
        database=url.path[1:], port=url.port
    )

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question TEXT,
            answer TEXT
        )
    """)
    conn.commit()
    conn.close()

def search_memory(question: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT answer FROM memory WHERE question = %s", (question,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def add_to_memory(question: str, answer: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory (question, answer) VALUES (%s, %s)", (question, answer))
    conn.commit()
    conn.close()

# === 3) Le Cerveau de l'IA : Le Workflow Complet ===
def ask_ai(question: str):
    # 1. Vérifier la mémoire cache SQL
    print(f"\n[1] Recherche dans la mémoire SQL pour : '{question}'")
    cached_answer = search_memory(question)
    if cached_answer:
        print("-> Réponse trouvée dans le cache SQL.")
        return f"(Mémoire SQL) {cached_answer}"

    # 2. Chercher dans la base de connaissance personnelle (data.txt)
    print(f"[2] Recherche dans la base de connaissance (data.txt)...")
    try:
        # On demande à la chaîne RAG de répondre.
        rag_answer = rag_chain.run(f"En te basant sur le contexte fourni, réponds à la question suivante : {question}. Si l'information n'est pas dans le contexte, dis simplement 'Je ne trouve pas cette information dans mes documents.'.")
        
        # On vérifie si la réponse est utile
        if "je ne trouve pas" not in rag_answer.lower() and "i don't know" not in rag_answer.lower():
            print("-> Réponse trouvée dans data.txt et générée par le LLM.")
            add_to_memory(question, rag_answer) # Apprentissage
            return f"(Connaissance Personnelle - data.txt) {rag_answer}"
    except Exception as e:
        print(f"Erreur pendant la recherche RAG : {e}")

    # 3. Chercher sur le web en dernier recours
    print(f"[3] Recherche sur le web avec DuckDuckGo...")
    web_results = web_search.run(question)
    if not web_results or "no good search result" in web_results:
        print("-> Pas de résultats pertinents sur le web.")
        return "Désolé, je n'ai trouvé aucune information à ce sujet, ni dans mes documents, ni sur le web."

    print("-> Résultats trouvés sur le web. Le LLM va synthétiser une réponse.")
    # On demande au LLM de formuler une réponse propre à partir des résultats web
    prompt = f"En te basant sur les informations suivantes issues d'une recherche web : '{web_results}', réponds à la question : '{question}'"
    web_answer = llm(prompt)
    
    add_to_memory(question, web_answer) # Apprentissage
    return f"(Recherche Web) {web_answer}"

# === 4) API Web avec FastAPI ===
app = FastAPI(title="Serveur IA Augmenté", description="Une IA avec LLM, RAG, Mémoire SQL et Recherche Web.")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(question_data: Question):
    response = ask_ai(question_data.question)
    return {"answer": response}

# === 5) Initialisation au Démarrage ===
print("\n--- Initialisation de la base de données ---")
init_db()
print("Base de données prête.")
print("\n>>> LE SERVEUR EST PRÊT À RECEVOIR DES REQUÊTES <<<")
