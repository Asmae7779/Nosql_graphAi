const { getSession } = require("../config/neo4j");

module.exports = {
  // Créer une relation Authorship (auteur écrit un papier)
  createAuthorship: async (req, res) => {
    const { authorId, paperId } = req.body;
    
    // Validation des données
    if (!authorId || !paperId) {
      return res.status(400).json({ error: "Les champs 'authorId' et 'paperId' sont requis" });
    }
    
    const session = getSession();
    try {
      // Sanitisation et conversion des IDs
      let cleanAuthorId = String(authorId).trim().replace(/[\r\n]/g, '');
      const cleanPaperId = String(paperId).trim().replace(/[\r\n]/g, '');
      
      // Convertir authorId en nombre si c'est un nombre valide
      const authorIdNum = Number(cleanAuthorId);
      if (!isNaN(authorIdNum) && authorIdNum.toString() === cleanAuthorId) {
        cleanAuthorId = authorIdNum;
      }
      
      // Vérifier que l'auteur et le papier existent
      const authorCheck = await session.run(
        "MATCH (a:Author {authorId: $authorId}) RETURN a",
        { authorId: cleanAuthorId }
      );
      if (authorCheck.records.length === 0) {
        return res.status(404).json({ error: "Author not found" });
      }
      
      const paperCheck = await session.run(
        "MATCH (p:Paper {paperId: $paperId}) RETURN p",
        { paperId: cleanPaperId }
      );
      if (paperCheck.records.length === 0) {
        return res.status(404).json({ error: "Paper not found" });
      }
      
      // Créer la relation
      const result = await session.run(
        `MATCH (a:Author {authorId: $authorId}), (p:Paper {paperId: $paperId})
         MERGE (a)-[r:AUTHORED]->(p)
         RETURN a, p, r`,
        { authorId: cleanAuthorId, paperId: cleanPaperId }
      );
      
      res.status(201).json({ 
        message: "Authorship created",
        data: {
          author: result.records[0]?.get("a").properties,
          paper: result.records[0]?.get("p").properties
        }
      });
    } catch (error) {
      console.error('Error in createAuthorship:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  // Supprimer une relation Authorship
  deleteAuthorship: async (req, res) => {
    const session = getSession();
    try {
      // Sanitisation et conversion des IDs
      let cleanAuthorId = String(req.params.authorId).trim().replace(/[\r\n]/g, '');
      const cleanPaperId = String(req.params.paperId).trim().replace(/[\r\n]/g, '');
      
      if (!cleanAuthorId || !cleanPaperId) {
        return res.status(400).json({ error: "Invalid authorId or paperId" });
      }
      
      // Convertir authorId en nombre si c'est un nombre valide
      const authorIdNum = Number(cleanAuthorId);
      if (!isNaN(authorIdNum) && authorIdNum.toString() === cleanAuthorId) {
        cleanAuthorId = authorIdNum;
      }
      
      const result = await session.run(
        `MATCH (a:Author {authorId: $authorId})-[r:AUTHORED]->(p:Paper {paperId: $paperId})
         DELETE r
         RETURN a, p`,
        { authorId: cleanAuthorId, paperId: cleanPaperId }
      );
      
      if (result.records.length === 0) {
        return res.status(404).json({ error: "Authorship not found" });
      }
      
      res.json({ message: "Authorship deleted" });
    } catch (error) {
      console.error('Error in deleteAuthorship:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },
};

