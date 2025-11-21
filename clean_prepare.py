import json
import pandas as pd

# Charger les données brutes détaillées
with open("papers_details_raw.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

authors_rows = []
papers_rows = []
authorship_rows = []

print(f"📌 Nombre d'articles chargés : {len(papers)}")

for p in papers:

    paper_id = p.get("paperId")
    if not paper_id:
        continue

    title = p.get("title")
    year = p.get("year")
    venue = p.get("venue")
    fields = p.get("fieldsOfStudy", [])

    # Ajouter l'article
    papers_rows.append({
        "paperId": paper_id,
        "title": title,
        "year": int(year) if year not in [None, ""] else None,
        "venue": venue,
        "fieldsOfStudy": ",".join(fields) if isinstance(fields, list) else fields
    })

    # Ajouter les auteurs + relations
    for a in p.get("authors", []):
        author_id = a.get("authorId")
        name = a.get("name")

        if not author_id or not name:
            continue

        authors_rows.append({
            "authorId": author_id,
            "name": name.strip()
        })

        authorship_rows.append({
            "authorId": author_id,
            "paperId": paper_id
        })



df_authors = pd.DataFrame(authors_rows)
df_papers = pd.DataFrame(papers_rows)
df_authorships = pd.DataFrame(authorship_rows)

# Nettoyage
df_authors = df_authors.drop_duplicates(subset=["authorId"])
df_papers = df_papers.drop_duplicates(subset=["paperId"])

# Filtrer (année >= 2015)
df_papers = df_papers[df_papers["year"].fillna(0) >= 2015]

# Garder seulement relations provenant d'articles valides
valid_papers = set(df_papers["paperId"])
df_authorships = df_authorships[df_authorships["paperId"].isin(valid_papers)]

print("\n📊 STATISTIQUES FINALES")
print("Auteurs :", df_authors.shape)
print("Papers  :", df_papers.shape)
print("Liens   :", df_authorships.shape)

# Export CSV
df_authors.to_csv("authors.csv", index=False)
df_papers.to_csv("papers.csv", index=False)
df_authorships.to_csv("authorships.csv", index=False)

print("\n✅ Fichiers CSV créés avec succès.")
print("📁 authors.csv")
print("📁 papers.csv")
print("📁 authorships.csv")
