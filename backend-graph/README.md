# Backend Graph - Neo4j API

API backend pour gérer les chercheurs et publications dans une base de données Neo4j.

## Configuration Neo4j sans authentification

Pour utiliser ce projet sans authentification, vous devez configurer Neo4j pour désactiver l'authentification :

### Option 1 : Configuration dans neo4j.conf

1. Trouvez le fichier de configuration Neo4j (généralement dans `conf/neo4j.conf`)
2. Ajoutez ou modifiez la ligne suivante :
   ```
   dbms.security.auth_enabled=false
   ```
3. Redémarrez Neo4j

### Option 2 : Variables d'environnement

Si vous préférez utiliser l'authentification, créez un fichier `.env` dans le dossier `backend-graph` :

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=votre_mot_de_passe
```

## Installation

```bash
npm install
```

## Démarrage

```bash
npm start
```

Le serveur démarre sur le port 5000 par défaut.

## Endpoints

### Researchers
- `GET /researchers` - Liste tous les chercheurs
- `GET /researchers/:id` - Récupère un chercheur par ID
- `POST /researchers` - Crée un chercheur
- `PUT /researchers/:id` - Met à jour un chercheur
- `DELETE /researchers/:id` - Supprime un chercheur
- `GET /researchers/ping` - Teste la connexion Neo4j

### Publications
- `GET /publications` - Liste toutes les publications
- `GET /publications/:id` - Récupère une publication par ID
- `POST /publications` - Crée une publication
- `PUT /publications/:id` - Met à jour une publication
- `DELETE /publications/:id` - Supprime une publication

