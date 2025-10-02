import os
from dotenv import load_dotenv
import fasttext
import mysql.connector
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from urllib.parse import urlparse

# === 1) Charger variables d'environnement ===
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
# MODIFICATION 1: Remplacement de CORPUS_FILE par le chemin du modèle pré-entraîné
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "cc.fr.300.bin")

# === 2) fastText : Chargement d'un modèle pré-entraîné ===
# MODIFICATION 2: On charge un modèle existant au lieu d'en entraîner un à chaque démarrage.
# C'est ce qui corrige l'erreur "ValueError: empty vocabulary".
if not os.path.exists(FASTTEXT_MODEL_PATH):
    print(f"ERREUR : Le modèle fastText '{FASTTEXT_MODEL_PATH}' n'a pas été trouvé.")
    print("Veuillez le télécharger et le placer dans le bon répertoire.")
    exit() # Stoppe le script si le modèle est manquant

print("Chargement du modèle fastText...")
model = fasttext.load_model(FASTTEXT_MODEL_PATH)
print("Modèle chargé avec succès.")

# === 3) DuckDuckGo Tool ===
search = DuckDuckGoSearchRun()

# === 4) MySQL connection ===
def get_db_connection():
    url = urlparse(DATABASE_URL)
    return mysql.connector.connect(
        host=url.hostname,
        user=url.username,
        password=url.password,
        database=url.path[1:],
        port=url.port
    )

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question TEXT,
            answer TEXT,
            vector BLOB
        )
    """)
    conn.commit()
    conn.close()

# === 5) Recherche mémoire MySQL ===
def search_memory(question):
    conn = get_db_connection()
    cursor = conn.cursor()
    q_vec = model.get_sentence_vector(question)
    cursor.execute("SELECT question, answer, vector FROM memory")
    rows = cursor.fetchall()
    best_answer, best_sim = None, -1
    for q, a, v in rows:
        # Les vecteurs sont stockés en float32
        vec = np.frombuffer(v, dtype=np.float32)
        # Calcul de la similarité cosinus
        sim = float(np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-9))
        if sim > best_sim:
            best_sim = sim
            best_answer = a
    conn.close()
    # Seuil de similarité pour retourner une réponse de la mémoire
    if best_sim > 0.8: # J'ai augmenté un peu le seuil pour plus de pertinence
        return best_answer
    return None

# === 6) Ajouter à la mémoire ===
def add_to_memory(question, answer):
    conn = get_db_connection()
    cursor = conn.cursor()
    # MODIFICATION 3: On stocke le vecteur de la QUESTION, pas de la réponse.
    # C'est plus logique pour retrouver des questions similaires par la suite.
    vec = model.get_sentence_vector(question)
    cursor.execute(
        "INSERT INTO memory (question, answer, vector) VALUES (%s,%s,%s)",
        (question, answer, vec.tobytes())
    )
    conn.commit()
    conn.close()

# === 7) fastText réponse simple ===
def fasttext_answer(question):
    # Prototype : ici tu peux améliorer avec un mini index vectoriel
    return None  # pour l’instant MySQL/DuckDuckGo prend le relais

# === 8) Reformulation ===
prompt_template = """
Tu es un assistant IA. Reformule la réponse suivante de manière claire et naturelle:

Réponse brute: {answer}
"""
prompt = PromptTemplate(input_variables=["answer"], template=prompt_template)
chain = LLMChain(prompt=prompt, llm=None)  # llm=None pour fastText/local
def reformulate(answer):
    # Prototype : retourne la réponse brute pour l'instant
    return answer

# === 9) Workflow hybride ===
def ask_ai(question):
    # 1) MySQL (Mémoire)
    answer = search_memory(question)
    if answer:
        return f"(Mémoire) {reformulate(answer)}"
    
    # 2) DuckDuckGo (Recherche externe)
    print(f"Recherche sur le web pour : {question}")
    answer = search.run(question)
    add_to_memory(question, answer) # On mémorise la nouvelle connaissance
    return f"(DuckDuckGo) {reformulate(answer)}"

# === 10) FastAPI ===
app = FastAPI()
class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(question: Question):
    response = ask_ai(question.question)
    return {"answer": response}

# === 11) Initialisation DB ===
print("Initialisation de la base de données...")
init_db()
print("Base de données prête.")
