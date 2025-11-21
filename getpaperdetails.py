import json
import requests
import time


with open("papers_list_raw.json", "r", encoding="utf-8") as f:
    papers_list = json.load(f)

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/"
FIELDS = "title,year,authors,venue,fieldsOfStudy"

detailed_papers = []

print(f"🚀 Début récupération détails pour {len(papers_list)} articles...\n")

for i, p in enumerate(papers_list):

    paper_id = p.get("paperId")
    if not paper_id:
        print(" Article sans paperId, ignoré.")
        continue

    url = f"{BASE_URL}{paper_id}"
    params = {"fields": FIELDS}

    print(f"[{i+1}/{len(papers_list)}] Récupération de {paper_id}...")

    success = False

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            detailed_papers.append(resp.json())
            success = True
            break  # sortie du retry
        except Exception as e:
            print(f"⚠️ Tentative {attempt+1}/3 échouée : {e}")
            time.sleep(1)  # attendre avant retry

    if not success:
        print(f"❌ Échec final pour {paper_id}, ignoré.")
        continue

    if (i+1) % 100 == 0:
        with open("papers_details_progress.json", "w", encoding="utf-8") as f:
            json.dump(detailed_papers, f, ensure_ascii=False, indent=2)
        print("📌 Sauvegarde intermédiaire effectuée (progress).")

  
    time.sleep(0.7)

with open("papers_details_raw.json", "w", encoding="utf-8") as f:
    json.dump(detailed_papers, f, ensure_ascii=False, indent=2)


print(f"Nombre total d'articles détaillés : {len(detailed_papers)}")
print(" Fichier 'papers_details_raw.json' créé.")
