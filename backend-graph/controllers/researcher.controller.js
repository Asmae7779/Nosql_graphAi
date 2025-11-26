const { getSession } = require("../config/neo4j");

module.exports = {

  // Lister tous les auteurs
  getAllAuthors: async (req, res) => {
    const session = getSession();
    try {
      // Essayer d'abord avec le label Author, puis sans label si ça ne fonctionne pas
      let result = await session.run("MATCH (a:Author) RETURN a LIMIT 100");
      
      // Si aucun résultat, essayer sans label (chercher par propriété authorId)
      if (result.records.length === 0) {
        console.log("No Author label found, trying without label...");
        result = await session.run("MATCH (a) WHERE a.authorId IS NOT NULL RETURN a LIMIT 100");
      }
      
      const data = result.records.map(rec => rec.get("a").properties);
      console.log(`Found ${data.length} authors`);
      res.json(data);
    } catch (error) {
      console.error('Error in getAllAuthors:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  // Auteur par ID
  getAuthorById: async (req, res) => {
    const session = getSession();
    try {
      // Sanitisation de l'ID (supprime les caractères de nouvelle ligne et espaces)
      let authorId = req.params.id.trim().replace(/[\r\n]/g, '');
      
      if (!authorId) {
        return res.status(400).json({ error: "Invalid author ID" });
      }
      
      // Convertir en nombre si c'est un nombre valide (pour compatibilité avec CSV)
      const authorIdNum = Number(authorId);
      if (!isNaN(authorIdNum) && authorIdNum.toString() === authorId) {
        authorId = authorIdNum;
      }
      
      // Essayer d'abord avec le label Author
      let result = await session.run(
        "MATCH (a:Author {authorId: $authorId}) RETURN a",
        { authorId }
      );
      
      // Si aucun résultat, essayer sans label
      if (result.records.length === 0) {
        result = await session.run(
          "MATCH (a) WHERE a.authorId = $authorId RETURN a",
          { authorId }
        );
      }
      
      if (result.records.length === 0) return res.status(404).json({ error: "Not found" });
      res.json(result.records[0].get("a").properties);
    } catch (error) {
      console.error('Error in getAuthorById:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  // Obtenir les papiers d'un auteur
  getAuthorPapers: async (req, res) => {
    const session = getSession();
    try {
      // Sanitisation de l'ID
      let authorId = req.params.id.trim().replace(/[\r\n]/g, '');
      
      if (!authorId) {
        return res.status(400).json({ error: "Invalid author ID" });
      }
      
      // Convertir en nombre si c'est un nombre valide
      const authorIdNum = Number(authorId);
      if (!isNaN(authorIdNum) && authorIdNum.toString() === authorId) {
        authorId = authorIdNum;
      }
      
      // Essayer d'abord avec les labels
      let result = await session.run(
        `MATCH (a:Author {authorId: $authorId})-[:AUTHORED]->(p:Paper)
         RETURN p ORDER BY p.year DESC, p.title`,
        { authorId }
      );
      
      // Si aucun résultat, essayer sans labels
      if (result.records.length === 0) {
        result = await session.run(
          `MATCH (a {authorId: $authorId})-[r]->(p)
           WHERE p.paperId IS NOT NULL
           RETURN p ORDER BY p.year DESC, p.title`,
          { authorId }
        );
      }
      const data = result.records.map(rec => rec.get("p").properties);
      res.json(data);
    } catch (error) {
      console.error('Error in getAuthorPapers:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  // Créer un auteur
  createAuthor: async (req, res) => {
    const { authorId, name } = req.body;
    
    // Validation des données
    if (!authorId || !name) {
      return res.status(400).json({ error: "Les champs 'authorId' et 'name' sont requis" });
    }
    
    const session = getSession();
    try {
      const result = await session.run(
        "CREATE (a:Author {authorId: $authorId, name: $name}) RETURN a",
        { authorId, name }
      );
      res.status(201).json({ 
        message: "Author created", 
        data: result.records[0]?.get("a").properties 
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  // Modifier auteur
  updateAuthor: async (req, res) => {
    const { name } = req.body;
    
    // Validation des données
    if (!name) {
      return res.status(400).json({ error: "Le champ 'name' est requis" });
    }
    
    const session = getSession();
    try {
      // Sanitisation de l'ID
      let authorId = req.params.id.trim().replace(/[\r\n]/g, '');
      
      if (!authorId) {
        return res.status(400).json({ error: "Invalid author ID" });
      }
      
      // Convertir en nombre si c'est un nombre valide
      const authorIdNum = Number(authorId);
      if (!isNaN(authorIdNum) && authorIdNum.toString() === authorId) {
        authorId = authorIdNum;
      }
      
      const result = await session.run(
        "MATCH (a:Author {authorId: $authorId}) SET a.name = $name RETURN a",
        { authorId, name }
      );
      
      if (result.records.length === 0) {
        return res.status(404).json({ error: "Author not found" });
      }
      
      res.json({ 
        message: "Updated", 
        data: result.records[0].get("a").properties 
      });
    } catch (error) {
      console.error('Error in updateAuthor:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  // Supprimer auteur
  deleteAuthor: async (req, res) => {
    const session = getSession();
    try {
      // Sanitisation de l'ID
      let authorId = req.params.id.trim().replace(/[\r\n]/g, '');
      
      if (!authorId) {
        return res.status(400).json({ error: "Invalid author ID" });
      }
      
      // Convertir en nombre si c'est un nombre valide
      const authorIdNum = Number(authorId);
      if (!isNaN(authorIdNum) && authorIdNum.toString() === authorId) {
        authorId = authorIdNum;
      }
      
      const result = await session.run(
        "MATCH (a:Author {authorId: $authorId}) DETACH DELETE a RETURN a",
        { authorId }
      );
      
      if (result.records.length === 0) {
        return res.status(404).json({ error: "Author not found" });
      }
      
      res.json({ message: "Deleted" });
    } catch (error) {
      console.error('Error in deleteAuthor:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },
};
