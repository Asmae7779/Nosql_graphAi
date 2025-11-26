# Guide de test API dans Postman

## Base URL
```
http://localhost:5000
```

---

## 🔍 1. TEST DE CONNEXION

### Test Neo4j
```
GET http://localhost:5000/researchers/ping
```
**Réponse attendue:**
```json
{
  "status": "connected",
  "neo4j": true
}
```

---

## 👥 2. AUTHORS (AUTEURS)

### 2.1. Lister tous les auteurs
```
GET http://localhost:5000/authors
```
ou
```
GET http://localhost:5000/researchers
```

**Réponse:** Tableau de tous les auteurs
```json
[
  {
    "authorId": 34917892,
    "name": "Djork-Arné Clevert"
  },
  ...
]
```

---

### 2.2. Obtenir un auteur par ID
```
GET http://localhost:5000/authors/34917892
```
ou
```
GET http://localhost:5000/researchers/34917892
```

**Réponse:**
```json
{
  "authorId": 34917892,
  "name": "Djork-Arné Clevert"
}
```

**Autres IDs à tester:**
- `3462562`
- `2465270`
- `3308557`

---

### 2.3. Obtenir les papiers d'un auteur
```
GET http://localhost:5000/authors/34917892/papers
```
ou
```
GET http://localhost:5000/researchers/34917892/papers
```

**Réponse:** Tableau des papiers de l'auteur
```json
[
  {
    "paperId": "f63e917638553414526a0cc8550de4ad2d83fe7a",
    "title": "Fast and Accurate Deep Network Learning...",
    "year": 2015,
    "venue": "International Conference on Learning Representations",
    "fieldsOfStudy": "Computer Science,Mathematics"
  },
  ...
]
```

---

### 2.4. Créer un auteur
```
POST http://localhost:5000/authors
```
**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "authorId": 99999999,
  "name": "John Doe"
}
```

**Réponse (201 Created):**
```json
{
  "message": "Author created",
  "data": {
    "authorId": 99999999,
    "name": "John Doe"
  }
}
```

---

### 2.5. Modifier un auteur
```
PUT http://localhost:5000/authors/34917892
```
**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "name": "Djork-Arné Clevert Updated"
}
```

**Réponse:**
```json
{
  "message": "Updated",
  "data": {
    "authorId": 34917892,
    "name": "Djork-Arné Clevert Updated"
  }
}
```

---

### 2.6. Supprimer un auteur
```
DELETE http://localhost:5000/authors/99999999
```

**Réponse:**
```json
{
  "message": "Deleted"
}
```

---

## 📄 3. PAPERS (PAPIERS)

### 3.1. Lister tous les papiers
```
GET http://localhost:5000/papers
```
ou
```
GET http://localhost:5000/publications
```

**Réponse:** Tableau de tous les papiers

---

### 3.2. Obtenir un papier par ID
```
GET http://localhost:5000/papers/f63e917638553414526a0cc8550de4ad2d83fe7a
```
ou
```
GET http://localhost:5000/publications/f63e917638553414526a0cc8550de4ad2d83fe7a
```

**Réponse:**
```json
{
  "paperId": "f63e917638553414526a0cc8550de4ad2d83fe7a",
  "title": "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)",
  "year": 2015,
  "venue": "International Conference on Learning Representations",
  "fieldsOfStudy": "Computer Science,Mathematics"
}
```

**Autres IDs de papiers à tester:**
- `a6ffce203cb7e587a5f4d36ca7442a7b26c65b07`
- `e02f91d625cd32290d4ede0f31284da115844316`

---

### 3.3. Obtenir les auteurs d'un papier
```
GET http://localhost:5000/papers/f63e917638553414526a0cc8550de4ad2d83fe7a/authors
```
ou
```
GET http://localhost:5000/publications/f63e917638553414526a0cc8550de4ad2d83fe7a/authors
```

**Réponse:** Tableau des auteurs du papier
```json
[
  {
    "authorId": 34917892,
    "name": "Djork-Arné Clevert"
  },
  {
    "authorId": 2465270,
    "name": "Thomas Unterthiner"
  },
  ...
]
```

---

### 3.4. Créer un papier
```
POST http://localhost:5000/papers
```
**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "paperId": "test123456789",
  "title": "Mon nouveau papier de recherche",
  "year": 2024,
  "venue": "Conference Test",
  "fieldsOfStudy": "Computer Science"
}
```

**Réponse (201 Created):**
```json
{
  "message": "Paper created",
  "data": {
    "paperId": "test123456789",
    "title": "Mon nouveau papier de recherche",
    "year": 2024,
    "venue": "Conference Test",
    "fieldsOfStudy": "Computer Science"
  }
}
```

---

### 3.5. Modifier un papier
```
PUT http://localhost:5000/papers/f63e917638553414526a0cc8550de4ad2d83fe7a
```
**Headers:**
```
Content-Type: application/json
```

**Body (JSON):** (au moins un champ requis)
```json
{
  "title": "Nouveau titre",
  "year": 2025,
  "venue": "Nouvelle conférence",
  "fieldsOfStudy": "Mathematics,Physics"
}
```

**Réponse:**
```json
{
  "message": "Updated",
  "data": {
    "paperId": "f63e917638553414526a0cc8550de4ad2d83fe7a",
    "title": "Nouveau titre",
    "year": 2025,
    "venue": "Nouvelle conférence",
    "fieldsOfStudy": "Mathematics,Physics"
  }
}
```

---

### 3.6. Supprimer un papier
```
DELETE http://localhost:5000/papers/test123456789
```

**Réponse:**
```json
{
  "message": "Deleted"
}
```

---

## 🔗 4. AUTHORSHIPS (RELATIONS AUTEUR-PAPIER)

### 4.1. Créer une relation (auteur écrit un papier)
```
POST http://localhost:5000/authorships
```
**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "authorId": 34917892,
  "paperId": "f63e917638553414526a0cc8550de4ad2d83fe7a"
}
```

**Réponse (201 Created):**
```json
{
  "message": "Authorship created",
  "data": {
    "author": {
      "authorId": 34917892,
      "name": "Djork-Arné Clevert"
    },
    "paper": {
      "paperId": "f63e917638553414526a0cc8550de4ad2d83fe7a",
      "title": "Fast and Accurate Deep Network Learning...",
      "year": 2015,
      "venue": "International Conference on Learning Representations",
      "fieldsOfStudy": "Computer Science,Mathematics"
    }
  }
}
```

---

### 4.2. Supprimer une relation
```
DELETE http://localhost:5000/authorships/34917892/f63e917638553414526a0cc8550de4ad2d83fe7a
```

**Format:** `DELETE /authorships/:authorId/:paperId`

**Réponse:**
```json
{
  "message": "Authorship deleted"
}
```

---

## 📋 5. COLLECTION POSTMAN RECOMMANDÉE

### Ordre de test suggéré:

1. **Test de connexion**
   - `GET /researchers/ping`

2. **Tests Authors (lecture)**
   - `GET /authors` - Liste tous les auteurs
   - `GET /authors/34917892` - Auteur spécifique
   - `GET /authors/34917892/papers` - Papiers d'un auteur

3. **Tests Papers (lecture)**
   - `GET /papers` - Liste tous les papiers
   - `GET /papers/f63e917638553414526a0cc8550de4ad2d83fe7a` - Papier spécifique
   - `GET /papers/f63e917638553414526a0cc8550de4ad2d83fe7a/authors` - Auteurs d'un papier

4. **Tests Authorships (lecture)**
   - Vérifier les relations existantes via les endpoints ci-dessus

5. **Tests CRUD (création/modification/suppression)**
   - `POST /authors` - Créer un auteur
   - `POST /papers` - Créer un papier
   - `POST /authorships` - Créer une relation
   - `PUT /authors/:id` - Modifier un auteur
   - `PUT /papers/:id` - Modifier un papier
   - `DELETE /authorships/:authorId/:paperId` - Supprimer une relation
   - `DELETE /authors/:id` - Supprimer un auteur
   - `DELETE /papers/:id` - Supprimer un papier

---

## ⚠️ 6. CODES DE RÉPONSE HTTP

- **200 OK** - Requête réussie
- **201 Created** - Ressource créée avec succès
- **400 Bad Request** - Données invalides (champs manquants, format incorrect)
- **404 Not Found** - Ressource non trouvée
- **500 Internal Server Error** - Erreur serveur

---

## 🐛 7. DÉPANNAGE

### Erreur ECONNRESET
- Vérifiez que l'URL ne contient pas de caractères invisibles
- Tapez l'URL manuellement au lieu de copier-coller
- Vérifiez que le serveur est bien démarré

### Erreur 404
- Vérifiez l'orthographe de l'endpoint
- Vérifiez que l'ID existe dans la base de données
- Utilisez `/authors` ou `/researchers` (les deux fonctionnent)

### Erreur 400
- Vérifiez que tous les champs requis sont présents
- Vérifiez le format JSON du body
- Vérifiez les types de données (year doit être un nombre)

---

## 📝 8. EXEMPLES DE DONNÉES RÉELLES

### Auteur existant:
- ID: `34917892` - Djork-Arné Clevert
- ID: `3462562` - G. Beroza
- ID: `2465270` - Thomas Unterthiner

### Papier existant:
- ID: `f63e917638553414526a0cc8550de4ad2d83fe7a` - Fast and Accurate Deep Network Learning...
- ID: `a6ffce203cb7e587a5f4d36ca7442a7b26c65b07` - Deep-learning seismology
- ID: `e02f91d625cd32290d4ede0f31284da115844316` - DeepXDE: A Deep Learning Library...

---

## ✅ 9. CHECKLIST DE TEST

- [ ] Test de connexion Neo4j fonctionne
- [ ] Liste tous les auteurs fonctionne
- [ ] Obtenir un auteur par ID fonctionne
- [ ] Obtenir les papiers d'un auteur fonctionne
- [ ] Liste tous les papiers fonctionne
- [ ] Obtenir un papier par ID fonctionne
- [ ] Obtenir les auteurs d'un papier fonctionne
- [ ] Créer un auteur fonctionne
- [ ] Créer un papier fonctionne
- [ ] Créer une relation fonctionne
- [ ] Modifier un auteur fonctionne
- [ ] Modifier un papier fonctionne
- [ ] Supprimer une relation fonctionne
- [ ] Supprimer un auteur fonctionne
- [ ] Supprimer un papier fonctionne

