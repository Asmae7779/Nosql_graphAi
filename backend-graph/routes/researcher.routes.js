const express = require("express");
const router = express.Router();
const controller = require("../controllers/researcher.controller");
const { getSession } = require("../config/neo4j");

router.use((req, res, next) => {
  if (req.params) {
    Object.keys(req.params).forEach(key => {
      if (typeof req.params[key] === 'string') {
        req.params[key] = req.params[key].replace(/[\r\n\t]/g, '').replace(/%0A/gi, '').replace(/%0D/gi, '').trim();
      }
    });
  }
  next();
});

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

router.get("/debug", async (req, res) => {
  const session = getSession();
  try {
    const labelsResult = await session.run("CALL db.labels()");
    const labels = labelsResult.records.map(r => r.get(0));
    
    const authorCounts = {};
    const paperCounts = {};
    const possibleAuthorLabels = ['Author', 'author', 'authors'];
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
    
    let authorSample = null;
    try {
      const result = await session.run("MATCH (n) WHERE n.authorId IS NOT NULL RETURN n LIMIT 1");
      if (result.records.length > 0) {
        authorSample = result.records[0].get('n').properties;
        authorSample.labels = result.records[0].get('n').labels;
      }
    } catch (e) {}
    
    let paperSample = null;
    try {
      const result = await session.run("MATCH (n) WHERE n.paperId IS NOT NULL RETURN n LIMIT 1");
      if (result.records.length > 0) {
        paperSample = result.records[0].get('n').properties;
        paperSample.labels = result.records[0].get('n').labels;
      }
    } catch (e) {}
    
    let relationshipCount = 0;
    try {
      const result = await session.run("MATCH ()-[r]->() RETURN count(r) as count");
      relationshipCount = result.records[0]?.get('count').toNumber() || 0;
    } catch (e) {}
    
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

router.get("/", controller.getAllAuthors);
router.post("/", controller.createAuthor);
router.get("/:id/papers", controller.getAuthorPapers);
router.get("/:id", controller.getAuthorById);
router.put("/:id", controller.updateAuthor);
router.delete("/:id", controller.deleteAuthor);

module.exports = router;
