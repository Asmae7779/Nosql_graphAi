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

// RECHERCHE D'AUTEURS - VERSION CORRIGÉE
router.get("/search", async (req, res) => {
  const { q } = req.query;
  const session = getSession();

  try {
    console.log(`🔍 [SEARCH] Requête originale: "${q}"`);
    
    // Normalisation plus intelligente
    const normalizedQuery = q.toLowerCase()
      .replace(/[^\w\s\.]/g, "") // Garde les points pour "S." etc.
      .trim();
    
    // Sépare les mots mais garde les mots courts (comme "S.")
    const queryWords = normalizedQuery.split(/\s+/).filter(w => w.length > 0);
    
    console.log(`🔍 [SEARCH] Mots de recherche:`, queryWords);

    let cypher;
    let params;

    if (queryWords.length > 1) {
      // Pour plusieurs mots, cherche CHAQUE mot séparément
      cypher = `
        MATCH (a:authors)
        WHERE ${queryWords.map((w, i) =>
          `toLower(a.name) CONTAINS toLower($word${i})`
        ).join(" AND ")}
        RETURN DISTINCT a.authorId AS authorId, a.name AS name
        ORDER BY a.name
        LIMIT 100
      `;

      params = Object.fromEntries(queryWords.map((w, i) => [`word${i}`, w]));
    } else {
      // Pour un seul mot, cherche avec LIKE pour plus de flexibilité
      cypher = `
        MATCH (a:authors)
        WHERE toLower(a.name) CONTAINS toLower($query)
        RETURN DISTINCT a.authorId AS authorId, a.name AS name
        ORDER BY a.name
        LIMIT 100
      `;

      params = { query: normalizedQuery };
    }

    console.log(`🔍 [SEARCH] Requête Cypher:`, cypher);
    console.log(`🔍 [SEARCH] Paramètres:`, params);

    const result = await session.run(cypher, params);

    console.log(`✅ [SEARCH] Auteurs trouvés: ${result.records.length}`);

    const authors = result.records.map(r => {
      const id = r.get("authorId");
      return {
        authorId: id.low ?? id,
        name: r.get("name")
      };
    });

    res.json(authors);

  } catch (e) {
    console.error("Search error:", e);
    res.status(500).json({ error: e.message });
  } finally {
    await session.close();
  }
});

// RECHERCHE UNIFIÉE
router.get("/unified-search", async (req, res) => {
  const { q } = req.query;
  const session = getSession();

  try {
    console.log("🔍 [UNIFIED SMART] Requête originale:", q);

    const normalizedQuery = q.toLowerCase()
      .replace(/[^\w\s\.]/g, "")
      .trim();
    
    const queryWords = normalizedQuery.split(/\s+/).filter(w => w.length > 0);

    console.log(`🔍 [UNIFIED SMART] Mots de recherche:`, queryWords);

    /*
     * AUTEURS - Trouve les auteurs correspondants
     */
    let authorsQuery, authorsParams;

    if (queryWords.length > 1) {
      authorsQuery = `
        MATCH (a:authors)
        WHERE ${queryWords.map((w, i) =>
          `toLower(a.name) CONTAINS toLower($word${i})`
        ).join(" AND ")}
        RETURN DISTINCT a.authorId AS authorId, a.name AS name
        ORDER BY a.name
        LIMIT 100
      `;
      authorsParams = Object.fromEntries(queryWords.map((w, i) => [`word${i}`, w]));
    } else {
      authorsQuery = `
        MATCH (a:authors)
        WHERE toLower(a.name) CONTAINS toLower($query)
        RETURN DISTINCT a.authorId AS authorId, a.name AS name
        ORDER BY a.name
        LIMIT 100
      `;
      authorsParams = { query: normalizedQuery };
    }

    const authorsRes = await session.run(authorsQuery, authorsParams);
    const authors = authorsRes.records.map(r => ({
      authorId: r.get("authorId").low ?? r.get("authorId"),
      name: r.get("name")
    }));

    console.log(`✅ [UNIFIED SMART] Auteurs trouvés: ${authors.length}`);

    /*
     * PUBLICATIONS - LOGIQUE AMÉLIORÉE
     */
    let papers = [];

    if (authors.length > 0) {
      // ✅ CAS 1: Si on a trouvé des auteurs, cherche LEURS publications
      console.log(`🔍 [UNIFIED SMART] Recherche publications des auteurs trouvés`);
      
      const authorIds = authors.map(a => a.authorId);
      
      const papersQuery = `
        MATCH (a:authors)-[:AUTHORED]->(p:papers)
        WHERE a.authorId IN $authorIds
        RETURN DISTINCT p.paperId AS paperId, p.title AS title, p.year AS year,
               p.venue AS venue, p.fieldsOfStudy AS fieldsOfStudy
        ORDER BY p.year DESC
        LIMIT 200
      `;
      
      const papersRes = await session.run(papersQuery, { authorIds });
      papers = papersRes.records.map(r => ({
        paperId: r.get("paperId"),
        title: r.get("title"),
        year: r.get("year"),
        venue: r.get("venue"),
        fieldsOfStudy: r.get("fieldsOfStudy")
      }));

      console.log(`✅ [UNIFIED SMART] Publications des auteurs: ${papers.length}`);

    } else {
      // ✅ CAS 2: Si aucun auteur trouvé, cherche dans les titres/domaines
      console.log(`🔍 [UNIFIED SMART] Aucun auteur trouvé, recherche dans titres/domaines`);
      
      let papersQuery, papersParams;

      if (queryWords.length > 1) {
        papersQuery = `
          MATCH (p:papers)
          WHERE ${queryWords.map((w, i) =>
            `(toLower(p.title) CONTAINS toLower($word${i}) OR 
              toLower(p.venue) CONTAINS toLower($word${i}) OR 
              toLower(p.fieldsOfStudy) CONTAINS toLower($word${i}))`
          ).join(" OR ")}
          RETURN DISTINCT p.paperId AS paperId, p.title AS title, p.year AS year,
                 p.venue AS venue, p.fieldsOfStudy AS fieldsOfStudy
          ORDER BY p.year DESC
          LIMIT 200
        `;
        papersParams = Object.fromEntries(queryWords.map((w, i) => [`word${i}`, w]));
      } else {
        papersQuery = `
          MATCH (p:papers)
          WHERE toLower(p.title) CONTAINS toLower($query)
             OR toLower(p.venue) CONTAINS toLower($query)
             OR toLower(p.fieldsOfStudy) CONTAINS toLower($query)
          RETURN DISTINCT p.paperId AS paperId, p.title AS title, p.year AS year,
                 p.venue AS venue, p.fieldsOfStudy AS fieldsOfStudy
          ORDER BY p.year DESC
          LIMIT 200
        `;
        papersParams = { query: normalizedQuery };
      }

      const papersRes = await session.run(papersQuery, papersParams);
      papers = papersRes.records.map(r => ({
        paperId: r.get("paperId"),
        title: r.get("title"),
        year: r.get("year"),
        venue: r.get("venue"),
        fieldsOfStudy: r.get("fieldsOfStudy")
      }));

      console.log(`✅ [UNIFIED SMART] Publications génériques: ${papers.length}`);
    }

    res.json({
      authors,
      papers,
      stats: {
        authorsCount: authors.length,
        papersCount: papers.length
      }
    });

  } catch (e) {
    console.error("UNIFIED SEARCH ERROR:", e);
    res.status(500).json({ error: e.message });
  } finally {
    await session.close();
  }
});


// Publications d'un auteur
router.get("/:id/publications", async (req, res) => {
  const { id } = req.params;
  const session = getSession();
  
  try {
    const result = await session.run(
      `MATCH (a:authors {authorId: $authorId})-[:AUTHORED]->(p:papers)
       RETURN p.paperId as paperId, p.title as title, p.year as year, 
              p.venue as venue, p.fieldsOfStudy as fieldsOfStudy
       ORDER BY p.year DESC`,
      { authorId: id }
    );
    
    const publications = result.records.map(record => ({
      paperId: record.get('paperId'),
      title: record.get('title'),
      year: record.get('year'),
      venue: record.get('venue'),
      fieldsOfStudy: record.get('fieldsOfStudy')
    }));
    
    res.json(publications);
  } catch (error) {
    console.error("Publications error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

router.get("/:id/publication-count", async (req, res) => {
  const { id } = req.params;
  const session = getSession();
  
  try {
    console.log(`🔢 [COUNT FIXED] authorId: ${id}, type: ${typeof id}`);
    
    // Convertir en number pour éviter les problèmes de notation scientifique
    const authorIdNum = Number(id);
    
    const result = await session.run(
      `MATCH (a:authors)-[:AUTHORED]->(p:papers)
       WHERE a.authorId = $authorId
       RETURN count(p) as publicationCount`,
      { authorId: authorIdNum }
    );
    
    const count = result.records[0]?.get('publicationCount').toNumber() || 0;
    console.log(`✅ [COUNT FIXED] Résultat: ${count} publications`);
    
    res.json({ count });
  } catch (error) {
    console.error("Publication count error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

// Top auteurs
router.get("/top", async (req, res) => {
  const { limit = 10 } = req.query;
  const session = getSession();
  
  try {
    const result = await session.run(
      `MATCH (a:authors)-[:AUTHORED]->(p:papers)
       RETURN a.authorId as authorId, a.name as name, count(p) as publicationCount
       ORDER BY publicationCount DESC
       LIMIT toInteger($limit)`,
      { limit: parseInt(limit) }
    );
    
    const topAuthors = result.records.map(record => ({
      authorId: record.get('authorId'),
      name: record.get('name'),
      publicationCount: record.get('publicationCount').toNumber() // .toNumber() au lieu de .low
    }));
    
    res.json(topAuthors);
  } catch (error) {
    console.error("Top authors error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});


// Détails d'un auteur
router.get("/:id", async (req, res) => {
  const { id } = req.params;
  const session = getSession();
  
  try {
    console.log(`🔍 [GET AUTHOR] Recherche de l'auteur avec ID: ${id}, type: ${typeof id}`);
    
    // Essayer de convertir en nombre si possible
    let authorIdParam;
    try {
      authorIdParam = parseInt(id);
      console.log(`🔍 [GET AUTHOR] ID converti en int: ${authorIdParam}`);
    } catch (e) {
      authorIdParam = id;
      console.log(`🔍 [GET AUTHOR] ID gardé comme string: ${authorIdParam}`);
    }
    
    const result = await session.run(
      `MATCH (a:authors)
       WHERE a.authorId = $authorId
       RETURN a.authorId as authorId, a.name as name`,
      { authorId: authorIdParam }
    );
    
    if (result.records.length === 0) {
      console.log(`❌ [GET AUTHOR] Auteur non trouvé pour ID: ${id}`);
      return res.status(404).json({ error: "Author not found" });
    }
    
    const record = result.records[0];
    const authorId = record.get('authorId');
    
    console.log(`✅ [GET AUTHOR] Auteur trouvé: ${authorId} - ${record.get('name')}`);
    
    const author = {
      authorId: authorId.low ?? authorId,
      name: record.get('name')
    };
    
    res.json(author);
  } catch (error) {
    console.error("Author details error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

// Collaborateurs d'un auteur 
router.get("/:id/collaborators", async (req, res) => {
  const { id } = req.params;
  const session = getSession();
  
  try {
    const result = await session.run(
      `MATCH (a:authors {authorId: $authorId})-[:AUTHORED]->(p:papers)<-[:AUTHORED]-(collab:authors)
       WHERE collab.authorId <> $authorId
       RETURN collab.authorId as authorId, collab.name as name, 
              count(p) as collaborationCount
       ORDER BY collaborationCount DESC`,
      { authorId: id }
    );
    
    const collaborators = result.records.map(record => ({
      authorId: record.get('authorId'),
      name: record.get('name'),
      collaborationCount: record.get('collaborationCount').low
    }));
    
    res.json(collaborators);
  } catch (error) {
    console.error("Collaborators error:", error);
    res.status(500).json({ error: error.message });
  } finally {
    await session.close();
  }
});

module.exports = router;