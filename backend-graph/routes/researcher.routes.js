const express = require("express");
const router = express.Router();
const controller = require("../controllers/researcher.controller");
const { getSession } = require("../config/neo4j");


// Middleware pour nettoyer les paramètres après le routage
router.use((req, res, next) => {
  // Nettoyer les paramètres de route après le routage
  if (req.params) {
    Object.keys(req.params).forEach(key => {
      if (typeof req.params[key] === 'string') {
        req.params[key] = req.params[key].replace(/[\r\n\t]/g, '').replace(/%0A/gi, '').replace(/%0D/gi, '').trim();
      }
    });
  }
  next();
});

// Route de test de connexion (doit être avant les routes paramétrées)
router.get("/ping", async (req, res) => {
  const session = getSession();
  try {
    await session.run("RETURN 1 AS test");
    res.json({ status: "connected", neo4j: true });
  } catch (error) {
    console.error('Error in /ping:', error);
    res.status(500).json({ status: "error", neo4j: false, error: error.message });
  } finally {
    await session.close();
  }
});

// Route de diagnostic pour vérifier la structure de la base de données
router.get("/debug", async (req, res) => {
  const session = getSession();
  try {
    // Vérifier les labels existants
    const labelsResult = await session.run("CALL db.labels()");
    const labels = labelsResult.records.map(r => r.get(0));
    
    // Vérifier les nœuds avec différents labels possibles
    const authorCounts = {};
    const paperCounts = {};
    
    // Tester différents labels possibles
    const possibleAuthorLabels = ['Author', 'author', 'authors', 'Author'];
    const possiblePaperLabels = ['Paper', 'paper', 'papers', 'Publication', 'publication'];
    
    for (const label of possibleAuthorLabels) {
      try {
        const result = await session.run(`MATCH (n:${label}) RETURN count(n) as count`);
        authorCounts[label] = result.records[0]?.get('count').toNumber() || 0;
      } catch (e) {
        authorCounts[label] = 0;
      }
    }
    
    for (const label of possiblePaperLabels) {
      try {
        const result = await session.run(`MATCH (n:${label}) RETURN count(n) as count`);
        paperCounts[label] = result.records[0]?.get('count').toNumber() || 0;
      } catch (e) {
        paperCounts[label] = 0;
      }
    }
    
    // Vérifier les propriétés d'un nœud Author s'il existe
    let authorSample = null;
    try {
      const result = await session.run("MATCH (n) WHERE n.authorId IS NOT NULL RETURN n LIMIT 1");
      if (result.records.length > 0) {
        authorSample = result.records[0].get('n').properties;
        authorSample.labels = result.records[0].get('n').labels;
      }
    } catch (e) {
      // Ignore
    }
    
    // Vérifier les propriétés d'un nœud Paper s'il existe
    let paperSample = null;
    try {
      const result = await session.run("MATCH (n) WHERE n.paperId IS NOT NULL RETURN n LIMIT 1");
      if (result.records.length > 0) {
        paperSample = result.records[0].get('n').properties;
        paperSample.labels = result.records[0].get('n').labels;
      }
    } catch (e) {
      // Ignore
    }
    
    // Compter les relations
    let relationshipCount = 0;
    try {
      const result = await session.run("MATCH ()-[r]->() RETURN count(r) as count");
      relationshipCount = result.records[0]?.get('count').toNumber() || 0;
    } catch (e) {
      // Ignore
    }
    
    res.json({
      labels: labels,
      authorCounts: authorCounts,
      paperCounts: paperCounts,
      authorSample: authorSample,
      paperSample: paperSample,
      relationshipCount: relationshipCount
    });
  } catch (error) {
    console.error('Error in /debug:', error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

// Routes CRUD (routes spécifiques avant routes paramétrées)
router.get("/", controller.getAllAuthors);
router.post("/", controller.createAuthor);
router.get("/:id/papers", controller.getAuthorPapers); // Route spécifique avant :id
router.get("/:id", controller.getAuthorById);
router.put("/:id", controller.updateAuthor);
router.delete("/:id", controller.deleteAuthor);

module.exports = router;
