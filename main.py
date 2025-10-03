import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import mysql.connector
from urllib.parse import urlparse

# --- Imports LangChain ---
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# === 1) Configuration et Initialisation des Outils ===
load_dotenv()

# Vérifications de sécurité
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("HUGGINGFACEHUB_API_TOKEN n'est pas défini dans le fichier .env")
if not os.getenv("DATABASE_URL"):
    raise ValueError("DATABASE_URL n'est pas défini dans le fichier .env")

PORT = 10000

# --- Initialisation du LLM distant (léger pour notre serveur) ---
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_new_tokens=250,
)

# --- Initialisation de l'outil de recherche web ---
search = DuckDuckGoSearchRun()

# === 2) Fonctions pour la Mémoire MySQL ===
def get_db_connection():
    """Établit la connexion à la base de données MySQL."""
    url = urlparse(os.getenv("DATABASE_URL"))
    return mysql.connector.connect(
        host=url.hostname, user=url.username, password=url.password,
        database=url.path[1:], port=url.port
    )

def init_db():
    """Crée la table 'memory' si elle n'existe pas."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def search_memory(question: str):
    """Cherche une réponse exacte dans la base de données."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT answer FROM memory WHERE question = %s", (question,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def add_to_memory(question: str, answer: str):
    """Ajoute une nouvelle paire question/réponse à la mémoire."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory (question, answer) VALUES (%s, %s)", (question, answer))
    conn.commit()
    conn.close()

# === 3) Le Cerveau de l'IA : Le Workflow Complet (Cache + LangChain) ===

# Le workflow LangChain pour générer une NOUVELLE réponse
def retriever(query):
    return search.run(query)

prompt_template = ChatPromptTemplate.from_template(
    "En te basant sur ce contexte : {context}\n\nRéponds à cette question : {question}"
)
generation_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

def ask_ai(question: str):
    """Orchestre la recherche de réponse en utilisant le cache MySQL en priorité."""
    print(f"[1] Vérification du cache MySQL pour : '{question}'")
    # Étape 1 : Vérifier la mémoire
    cached_answer = search_memory(question)
    if cached_answer:
        print("-> Cache HIT! Réponse trouvée dans MySQL.")
        return f"(Mémoire SQL) {cached_answer}"

    print("-> Cache MISS. Lancement de la recherche web et de la génération par l'IA.")
    # Étape 2 : Si rien en mémoire, générer une nouvelle réponse
    new_answer = generation_chain.invoke(question)

    print("[3] Sauvegarde de la nouvelle réponse dans MySQL.")
    # Étape 3 : Apprendre en sauvegardant la nouvelle réponse
    add_to_memory(question, new_answer)

    return f"(Recherche Web & IA) {new_answer}"

# === 4) Le Serveur Web avec FastAPI ===
app = FastAPI(title="Serveur IA avec Mémoire SQL", description="Intègre LangChain, DuckDuckGo et un cache MySQL.")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(question_data: Question):
    try:
        response = ask_ai(question_data.question)
        return {"answer": response}
    except Exception as e:
        print(f"ERREUR MAJEURE : {e}")
        return {"error": "Une erreur critique est survenue."}

# === 5) Initialisation au Démarrage ===
print("Initialisation de la base de données...")
init_db()
print("Base de données prête.")
print(f"\n>>> Serveur prêt à démarrer sur le port {PORT} <<<")
