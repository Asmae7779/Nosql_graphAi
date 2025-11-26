# Guide de Diagnostic - Problème GET ne retourne rien

## Étape 1: Vérifier la structure de la base de données

Testez cet endpoint dans Postman pour voir ce qui existe réellement dans Neo4j:

```
GET http://localhost:5000/researchers/debug
```

Cet endpoint vous montrera:
- Les labels existants dans la base
- Le nombre de nœuds par label
- Un exemple de nœud Author et Paper
- Le nombre de relations

## Étape 2: Vérifier l'import CSV

Lors de l'import CSV dans Neo4j, vous devez spécifier les labels. Vérifiez que vous avez utilisé:

### Pour authors.csv:
```cypher
LOAD CSV WITH HEADERS FROM 'file:///authors.csv' AS row
CREATE (a:Author {authorId: toInteger(row.authorId), name: row.name})
```

### Pour papers.csv:
```cypher
LOAD CSV WITH HEADERS FROM 'file:///papers.csv' AS row
CREATE (p:Paper {
  paperId: row.paperId, 
  title: row.title, 
  year: toFloat(row.year), 
  venue: row.venue, 
  fieldsOfStudy: row.fieldsOfStudy
})
```

### Pour authorships.csv:
```cypher
LOAD CSV WITH HEADERS FROM 'file:///authorships.csv' AS row
MATCH (a:Author {authorId: toInteger(row.authorId)})
MATCH (p:Paper {paperId: row.paperId})
CREATE (a)-[:AUTHORED]->(p)
```

## Étape 3: Si les labels n'existent pas

Si les données ont été importées sans labels, vous pouvez les ajouter avec ces requêtes Cypher dans Neo4j Browser:

### Ajouter le label Author:
```cypher
MATCH (n)
WHERE n.authorId IS NOT NULL
SET n:Author
RETURN count(n) as authors_labeled
```

### Ajouter le label Paper:
```cypher
MATCH (n)
WHERE n.paperId IS NOT NULL
SET n:Paper
RETURN count(n) as papers_labeled
```

### Créer les relations AUTHORED:
```cypher
LOAD CSV WITH HEADERS FROM 'file:///authorships.csv' AS row
MATCH (a:Author {authorId: toInteger(row.authorId)})
MATCH (p:Paper {paperId: row.paperId})
MERGE (a)-[:AUTHORED]->(p)
RETURN count(*) as relationships_created
```

## Étape 4: Tester les endpoints

Après avoir vérifié/corrigé les labels, testez:

1. **Debug:**
   ```
   GET http://localhost:5000/researchers/debug
   ```

2. **Liste des auteurs:**
   ```
   GET http://localhost:5000/authors
   ```

3. **Liste des papiers:**
   ```
   GET http://localhost:5000/papers
   ```

4. **Auteur spécifique:**
   ```
   GET http://localhost:5000/authors/34917892
   ```

## Solutions temporaires

Le code a été modifié pour chercher aussi sans labels si les labels ne sont pas trouvés. Mais il est recommandé d'avoir les bons labels pour de meilleures performances.

