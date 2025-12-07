const express = require("express");
const router = express.Router();
const controller = require("../controllers/publication.controller");
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

// Recherche de publications 
router.get("/search", async (req, res) => {
  const { q } = req.query;
  const session = getSession();
  
  try {
    const result = await session.run(
      `MATCH (p:papers)
       WHERE toLower(p.title) CONTAINS toLower($query)
          OR toLower(p.venue) CONTAINS toLower($query)
          OR toLower(p.fieldsOfStudy) CONTAINS toLower($query)
          OR EXISTS((:authors)-[:AUTHORED]->(p) AND 
             ANY(author IN [(a:authors)-[:AUTHORED]->(p) | a] 
             WHERE toLower(author.name) CONTAINS toLower($query)))
       RETURN DISTINCT p.paperId as paperId, p.title as title, p.year as year,
              p.venue as venue, p.fieldsOfStudy as fieldsOfStudy
       ORDER BY p.year DESC
       LIMIT 200`, 
      { query: q }
    );
    
    const papers = result.records.map(record => ({
      paperId: record.get('paperId'),
      title: record.get('title'),
      year: record.get('year'),
      venue: record.get('venue'),
      fieldsOfStudy: record.get('fieldsOfStudy')
    }));
    
    res.json(papers);
  } catch (error) {
    console.error("Search papers error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

// Auteurs d'un papier
router.get("/:id/authors", async (req, res) => {
  const { id } = req.params;
  const session = getSession();
  
  try {
    const result = await session.run(
      `MATCH (p:papers {paperId: $paperId})<-[:AUTHORED]-(a:authors)
       RETURN a.authorId as authorId, a.name as name
       ORDER BY a.name`,
      { paperId: id }
    );
    
    const authors = result.records.map(record => ({
      authorId: record.get('authorId'),
      name: record.get('name')
    }));
    
    res.json(authors);
  } catch (error) {
    console.error("Paper authors error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

// Papiers similaires
router.get("/:id/related", async (req, res) => {
  const { id } = req.params;
  const session = getSession();
  
  try {
    const result = await session.run(
      `MATCH (p:papers {paperId: $paperId})<-[:AUTHORED]-(author:authors)-[:AUTHORED]->(related:papers)
       WHERE related.paperId <> $paperId
       WITH related, count(author) as commonAuthors
       RETURN related.paperId as paperId, related.title as title, 
              related.year as year, related.venue as venue, commonAuthors
       ORDER BY commonAuthors DESC
       LIMIT 10`,
      { paperId: id }
    );
    
    const relatedPapers = result.records.map(record => ({
      paperId: record.get('paperId'),
      title: record.get('title'),
      year: record.get('year'),
      venue: record.get('venue'),
      commonAuthors: record.get('commonAuthors').low
    }));
    
    res.json(relatedPapers);
  } catch (error) {
    console.error("Related papers error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

// Top publications
router.get("/top", async (req, res) => {
  const { limit = 10 } = req.query;
  const session = getSession();
  
  try {
    const result = await session.run(
      `MATCH (p:papers)
       RETURN p.paperId as paperId, p.title as title, p.year as year,
              p.venue as venue, p.fieldsOfStudy as fieldsOfStudy
       ORDER BY p.year DESC
       LIMIT $limit`,
      { limit: parseInt(limit) }
    );
    
    const topPapers = result.records.map(record => ({
      paperId: record.get('paperId'),
      title: record.get('title'),
      year: record.get('year'),
      venue: record.get('venue'),
      fieldsOfStudy: record.get('fieldsOfStudy')
    }));
    
    res.json(topPapers);
  } catch (error) {
    console.error("Top papers error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

// Détails d'un papier 
router.get("/:id", async (req, res) => {
  const { id } = req.params;
  const session = getSession();
  
  try {
    const result = await session.run(
      `MATCH (p:papers {paperId: $paperId})
       RETURN p.paperId as paperId, p.title as title, p.year as year,
              p.venue as venue, p.fieldsOfStudy as fieldsOfStudy`,
      { paperId: id }
    );
    
    if (result.records.length === 0) {
      return res.status(404).json({ error: "Paper not found" });
    }
    
    const paper = {
      paperId: result.records[0].get('paperId'),
      title: result.records[0].get('title'),
      year: result.records[0].get('year'),
      venue: result.records[0].get('venue'),
      fieldsOfStudy: result.records[0].get('fieldsOfStudy')
    };
    
    res.json(paper);
  } catch (error) {
    console.error("Paper details error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

module.exports = router;