from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.sql_database import SQLDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import fasttext
import mysql.connector

# === 1) Connexion à MySQL ===
db = SQLDatabase.from_uri(
    "mysql://avnadmini:AVNS_BvVULOCxM7CcMQd0Aqw@mysql-1a36101-botwii.c.aivencloud.com:14721/defaultdb?ssl-mode=false"
)

# === 2) Init fastText (mini modèle, <20 Mo) ===
# Créer un modèle vide (non supervisé) si pas encore entraîné
model = fasttext.train_unsupervised("data.txt", model='skipgram', dim=50)  # "data.txt" = corpus initial minimal

# === 3) Outil de recherche DuckDuckGo ===
search = DuckDuckGoSearchRun()

# === 4) Mémoire conversationnelle (buffer en RAM + MySQL) ===
memory = ConversationBufferMemory()

# === 5) Prompt pour l’agent ===
prompt = PromptTemplate(
    input_variables=["question"],
    template="Tu es une IA légère. Utilise fastText + DuckDuckGo + ta mémoire MySQL pour répondre à: {question}"
)

# === 6) Chaîne (ici tu pourrais brancher un mini modèle texte local si besoin) ===
chain = LLMChain(prompt=prompt, llm=None, memory=memory)  # llm=None car tu utilises fastText

# === 7) Exemple de workflow ===
def ask_ai(question):
    # Recherche web via DuckDuckGo
    result = search.run(question)

    # Encoder résultat avec fastText
    vector = model.get_sentence_vector(result)

    # Sauvegarder dans MySQL
    conn = mysql.connector.connect(
        host="mysql-1a36101-botwii.c.aivencloud.com",
        user="avnadmini",
        password="AVNS_BvVULOCxM7CcMQd0Aqw",
        database="defaultdb",
        port=14721
    )
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS memory (question TEXT, result TEXT, vector BLOB)")
    cursor.execute("INSERT INTO memory (question, result, vector) VALUES (%s, %s, %s)",
                   (question, result, vector.tobytes()))
    conn.commit()
    conn.close()

    # Réponse simple basée sur DuckDuckGo + mémoire
    response = f"Réponse trouvée: {result[:300]}..."
    return response

# === 8) Test ===
print(ask_ai("Qui est Albert Einstein ?"))
