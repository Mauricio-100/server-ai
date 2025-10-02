#!/bin/bash

# Nom du fichier du modèle
MODEL_FILE="cc.fr.300.bin"
# URL pour le télécharger (version compressée .gz)
MODEL_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz"

# Vérifie si le fichier du modèle n'existe PAS
if [ ! -f "$MODEL_FILE" ]; then
  echo "--- Le modèle fastText n'a pas été trouvé. Téléchargement en cours... ---"
  
  # Télécharge le fichier compressé
  wget -O ${MODEL_FILE}.gz $MODEL_URL
  
  echo "--- Décompression du modèle... ---"
  # Décompresse le fichier
  gunzip ${MODEL_FILE}.gz
  
  echo "--- Modèle prêt. Lancement du serveur... ---"
else
  echo "--- Le modèle fastText existe déjà. Lancement du serveur... ---"
fi

# Lance l'application FastAPI (votre commande de démarrage originale)
uvicorn main:app --host 0.0.0.0 --port 10000
