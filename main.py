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

# === 1) Charger variables d'environnement ===
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
CORPUS_FILE = os.getenv("CORPUS_FILE", "data.txt")
FASTTEXT_DIM = int(os.getenv("FASTTEXT_DIM", 50))

# === 2) fastText ===
model = fasttext.train_unsupervised(CORPUS_FILE, model="skipgram", dim=FASTTEXT_DIM)

# === 3) DuckDuckGo Tool ===
search = DuckDuckGoSearchRun()

# === 4) MySQL connection ===
def get_db_connection():
    # Parse DATABASE_URL si nécessaire
    # Exemple simple pour Aiven
    from urllib.parse import urlparse
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
        vec = np.frombuffer(v, dtype=np.float32)
        sim = float(np.dot(q_vec, vec)/(np.linalg.norm(q_vec)*np.linalg.norm(vec)+1e-9))
        if sim > best_sim:
            best_sim = sim
            best_answer = a
    conn.close()
    if best_sim > 0.7:
        return best_answer
    return None

# === 6) Ajouter à la mémoire ===
def add_to_memory(question, answer):
    conn = get_db_connection()
    cursor = conn.cursor()
    vec = model.get_sentence_vector(answer)
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
    # 1) fastText
    answer = fasttext_answer(question)
    if answer:
        return f"(fastText) {reformulate(answer)}"
    # 2) MySQL
    answer = search_memory(question)
    if answer:
        return f"(Mémoire) {reformulate(answer)}"
    # 3) DuckDuckGo
    answer = search.run(question)
    add_to_memory(question, answer)
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
init_db()
