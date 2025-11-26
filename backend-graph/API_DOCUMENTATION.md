# Documentation API - Backend Graph Neo4j

## Base URL
```
http://localhost:5000
```

## Endpoints disponibles

### 🔍 Authors (Auteurs)

#### 1. Tester la connexion Neo4j
```
GET /researchers/ping
GET /authors/ping
```
**Réponse:** `{ "status": "connected", "neo4j": true }`

#### 2. Lister tous les auteurs
```
GET /authors
GET /researchers
```
**Réponse:** Tableau de tous les auteurs

#### 3. Obtenir un auteur par ID
```
GET /authors/:id
GET /researchers/:id
```
**Exemple:** `GET /authors/34917892`

**Réponse:**
```json
{
  "authorId": 34917892,
  "name": "Djork-Arné Clevert"
}
```

#### 4. Obtenir les papiers d'un auteur
```
GET /authors/:id/papers
GET /researchers/:id/papers
```
**Exemple:** `GET /authors/34917892/papers`

**Réponse:** Tableau des papiers de l'auteur

#### 5. Créer un auteur
```
POST /authors
POST /researchers
```
**Body (JSON):**
```json
{
  "authorId": 12345678,
  "name": "John Doe"
}
```

#### 6. Modifier un auteur
```
PUT /authors/:id
PUT /researchers/:id
```
**Body (JSON):**
```json
{
  "name": "John Doe Updated"
}
```

#### 7. Supprimer un auteur
```
DELETE /authors/:id
DELETE /researchers/:id
```

---

### 📄 Papers (Papiers)

#### 1. Lister tous les papiers
```
GET /papers
GET /publications
```

#### 2. Obtenir un papier par ID
```
GET /papers/:id
GET /publications/:id
```
**Exemple:** `GET /papers/f63e917638553414526a0cc8550de4ad2d83fe7a`

**Réponse:**
```json
{
  "paperId": "f63e917638553414526a0cc8550de4ad2d83fe7a",
  "title": "Fast and Accurate Deep Network Learning...",
  "year": 2015,
  "venue": "International Conference on Learning Representations",
  "fieldsOfStudy": "Computer Science,Mathematics"
}
```

#### 3. Obtenir les auteurs d'un papier
```
GET /papers/:id/authors
GET /publications/:id/authors
```
**Exemple:** `GET /papers/f63e917638553414526a0cc8550de4ad2d83fe7a/authors`

**Réponse:** Tableau des auteurs du papier

#### 4. Créer un papier
```
POST /papers
POST /publications
```
**Body (JSON):**
```json
{
  "paperId": "abc123def456",
  "title": "Mon titre de papier",
  "year": 2024,
  "venue": "Conference Name",
  "fieldsOfStudy": "Computer Science"
}
```

#### 5. Modifier un papier
```
PUT /papers/:id
PUT /publications/:id
```
**Body (JSON):** (au moins un champ requis)
```json
{
  "title": "Nouveau titre",
  "year": 2025,
  "venue": "Nouvelle conférence",
  "fieldsOfStudy": "Mathematics"
}
```

#### 6. Supprimer un papier
```
DELETE /papers/:id
DELETE /publications/:id
```

---

### 🔗 Authorships (Relations Auteur-Papier)

#### 1. Créer une relation (auteur écrit un papier)
```
POST /authorships
```
**Body (JSON):**
```json
{
  "authorId": 34917892,
  "paperId": "f63e917638553414526a0cc8550de4ad2d83fe7a"
}
```

#### 2. Supprimer une relation
```
DELETE /authorships/:authorId/:paperId
```
**Exemple:** `DELETE /authorships/34917892/f63e917638553414526a0cc8550de4ad2d83fe7a`

---

## Exemples de tests Postman

### Test 1: Obtenir un auteur existant
```
GET http://localhost:5000/authors/34917892
```

### Test 2: Obtenir les papiers d'un auteur
```
GET http://localhost:5000/authors/34917892/papers
```

### Test 3: Obtenir un papier
```
GET http://localhost:5000/papers/f63e917638553414526a0cc8550de4ad2d83fe7a
```

### Test 4: Obtenir les auteurs d'un papier
```
GET http://localhost:5000/papers/f63e917638553414526a0cc8550de4ad2d83fe7a/authors
```

### Test 5: Créer une relation
```
POST http://localhost:5000/authorships
Content-Type: application/json

{
  "authorId": 34917892,
  "paperId": "f63e917638553414526a0cc8550de4ad2d83fe7a"
}
```

---

## Codes de réponse HTTP

- `200 OK` - Requête réussie
- `201 Created` - Ressource créée avec succès
- `400 Bad Request` - Données invalides
- `404 Not Found` - Ressource non trouvée
- `500 Internal Server Error` - Erreur serveur

---

## Notes importantes

1. **Types de données:**
   - `authorId` peut être un nombre (comme dans le CSV) ou une string
   - `paperId` est toujours une string (hash)
   - `year` peut être un float (2015.0) mais sera converti en entier

2. **Sanitisation automatique:**
   - Les IDs sont automatiquement nettoyés (suppression des caractères de nouvelle ligne)
   - Les IDs numériques sont convertis en nombres si possible

3. **Alias de routes:**
   - `/authors` et `/researchers` pointent vers les mêmes endpoints
   - `/papers` et `/publications` pointent vers les mêmes endpoints

4. **Ordre des routes:**
   - Les routes spécifiques (`/ping`, `/:id/papers`) doivent être définies avant les routes paramétrées (`/:id`)

