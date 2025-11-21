import requests
import json
import time  


queries = [
    "deep learning",
    "computer vision",
    "reinforcement learning",
    "transformers",
    "NLP",
    "machine learning",
    "graph neural networks",
    "medical AI",
    "image segmentation",
    "speech recognition"
]


LIMIT = 100                 # nombre d articles par appel
NB_PAGES_PER_QUERY = 10     # nombre de pages par mot cle
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

all_papers = []

print("🚀 Début de la collecte...\n")


for q in queries:
    print(f"📌 Collecte pour le thème : {q}")
    for page in range(NB_PAGES_PER_QUERY):
        offset = page * LIMIT
        
        params = {
            "query": q,
            "limit": LIMIT,
            "offset": offset,
            "fields": "title"
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            # Ajouter les résultats
            all_papers.extend(data.get("data", []))
            
            print(f"  ➕ Page {page+1}/{NB_PAGES_PER_QUERY} récupérée")
        
        except Exception as e:
            print(f"⚠ Erreur sur {q} page {page+1} → {e}")
        
        # Pause pour éviter rate-limit
        time.sleep(1)

print("\n📊 FIN DE COLLECTE")
print(f"Nombre total d'articles récupérés : {len(all_papers)}")

# 💾 Sauvegarde
with open("papers_list_raw.json", "w", encoding="utf-8") as f:
    json.dump(all_papers, f, ensure_ascii=False, indent=2)

print("✅ Fichier 'papers_list_raw.json' créé.")
