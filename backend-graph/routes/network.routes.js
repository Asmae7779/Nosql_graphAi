const express = require("express");
const router = express.Router();
const { getSession } = require("../config/neo4j");

// Statistiques du réseau
router.get("/stats", async (req, res) => {
  const session = getSession();
  
  try {
    const result = await session.run(`
      // Nombre total d'auteurs
      MATCH (a:authors)
      WITH count(a) as totalAuthors
      
      // Nombre total de publications
      MATCH (p:papers)
      WITH totalAuthors, count(p) as totalPapers
      
      // Nombre total de collaborations
      MATCH (a1:authors)-[:AUTHORED]->(p:papers)<-[:AUTHORED]-(a2:authors)
      WHERE a1.authorId < a2.authorId
      WITH totalAuthors, totalPapers, count(DISTINCT [a1.authorId, a2.authorId]) as totalCollaborations
      
      // Année la plus récente
      MATCH (p:papers)
      WITH totalAuthors, totalPapers, totalCollaborations, max(p.year) as latestYear
      
      RETURN totalAuthors, totalPapers, totalCollaborations, latestYear
    `);
    
    if (result.records.length === 0) {
      return res.status(404).json({ error: "No data found" });
    }
    
    const record = result.records[0];
    const stats = {
      totalAuthors: record.get('totalAuthors').toNumber(),
      totalPapers: record.get('totalPapers').toNumber(),
      totalCollaborations: record.get('totalCollaborations').toNumber(),
      latestYear: record.get('latestYear')
    };
    
    res.json(stats);
  } catch (error) {
    console.error("Network stats error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

module.exports = router;