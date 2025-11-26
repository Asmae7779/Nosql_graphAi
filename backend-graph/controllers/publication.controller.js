const { getSession } = require("../config/neo4j");

const sanitizePaperId = (id) => {
  const paperId = String(id).trim().replace(/[\r\n]/g, '');
  return paperId || null;
};

const validateYear = (year) => {
  const yearNum = typeof year === 'string' ? parseFloat(year) : year;
  const yearInt = Math.floor(yearNum);
  return !isNaN(yearNum) && yearInt >= 1000 && yearInt <= 9999 ? yearInt : null;
};

module.exports = {
  getAllPapers: async (req, res) => {
    const session = getSession();
    try {
      let result = await session.run("MATCH (p:Paper) RETURN p LIMIT 100");
      if (result.records.length === 0) {
        console.log("No Paper label found, trying without label...");
        result = await session.run("MATCH (p) WHERE p.paperId IS NOT NULL RETURN p LIMIT 100");
      }
      const data = result.records.map(r => r.get("p").properties);
      console.log(`Found ${data.length} papers`);
      res.json(data);
    } catch (error) {
      console.error('Error in getAllPapers:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  createPaper: async (req, res) => {
    const { paperId, title, year, venue, fieldsOfStudy } = req.body;
    
    if (!paperId || !title || !year) {
      return res.status(400).json({ error: "Les champs 'paperId', 'title' et 'year' sont requis" });
    }
    
    const yearInt = validateYear(year);
    if (!yearInt) {
      return res.status(400).json({ error: "Le champ 'year' doit être un nombre entre 1000 et 9999" });
    }
    
    const session = getSession();
    try {
      const result = await session.run(
        `CREATE (p:Paper {
          paperId: $paperId, 
          title: $title, 
          year: $year, 
          venue: $venue, 
          fieldsOfStudy: $fieldsOfStudy
        }) RETURN p`,
        { paperId, title, year: yearInt, venue: venue || null, fieldsOfStudy: fieldsOfStudy || null }
      );
      res.status(201).json({ 
        message: "Paper created", 
        data: result.records[0]?.get("p").properties 
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  getPaperById: async (req, res) => {
    const session = getSession();
    try {
      const paperId = sanitizePaperId(req.params.id);
      if (!paperId) {
        return res.status(400).json({ error: "Invalid paper ID" });
      }
      
      let result = await session.run(
        "MATCH (p:Paper {paperId: $paperId}) RETURN p",
        { paperId }
      );
      
      if (result.records.length === 0) {
        result = await session.run(
          "MATCH (p) WHERE p.paperId = $paperId RETURN p",
          { paperId }
        );
      }
      
      if (result.records.length === 0) return res.status(404).json({ error: "Not found" });
      res.json(result.records[0].get("p").properties);
    } catch (error) {
      console.error('Error in getPaperById:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  getPaperAuthors: async (req, res) => {
    const session = getSession();
    try {
      const paperId = sanitizePaperId(req.params.id);
      if (!paperId) {
        return res.status(400).json({ error: "Invalid paper ID" });
      }
      
      let result = await session.run(
        `MATCH (a:Author)-[:AUTHORED]->(p:Paper {paperId: $paperId})
         RETURN a`,
        { paperId }
      );
      
      if (result.records.length === 0) {
        result = await session.run(
          `MATCH (a)-[r]->(p {paperId: $paperId})
           WHERE a.authorId IS NOT NULL
           RETURN a`,
          { paperId }
        );
      }
      
      const data = result.records.map(rec => rec.get("a").properties);
      res.json(data);
    } catch (error) {
      console.error('Error in getPaperAuthors:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  updatePaper: async (req, res) => {
    const { title, year, venue, fieldsOfStudy } = req.body;
    
    if (!title && !year && !venue && !fieldsOfStudy) {
      return res.status(400).json({ error: "Au moins un champ est requis" });
    }
    
    const session = getSession();
    try {
      const paperId = sanitizePaperId(req.params.id);
      if (!paperId) {
        return res.status(400).json({ error: "Invalid paper ID" });
      }
      
      const updates = [];
      const params = { paperId };
      
      if (title !== undefined) {
        updates.push("p.title = $title");
        params.title = title;
      }
      if (year !== undefined) {
        const yearInt = validateYear(year);
        if (!yearInt) {
          return res.status(400).json({ error: "Le champ 'year' doit être un nombre entre 1000 et 9999" });
        }
        updates.push("p.year = $year");
        params.year = yearInt;
      }
      if (venue !== undefined) {
        updates.push("p.venue = $venue");
        params.venue = venue;
      }
      if (fieldsOfStudy !== undefined) {
        updates.push("p.fieldsOfStudy = $fieldsOfStudy");
        params.fieldsOfStudy = fieldsOfStudy;
      }
      
      const result = await session.run(
        `MATCH (p:Paper {paperId: $paperId}) SET ${updates.join(", ")} RETURN p`,
        params
      );
      
      if (result.records.length === 0) {
        return res.status(404).json({ error: "Paper not found" });
      }
      
      res.json({ 
        message: "Updated", 
        data: result.records[0].get("p").properties 
      });
    } catch (error) {
      console.error('Error in updatePaper:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },

  deletePaper: async (req, res) => {
    const session = getSession();
    try {
      const paperId = sanitizePaperId(req.params.id);
      if (!paperId) {
        return res.status(400).json({ error: "Invalid paper ID" });
      }
      
      const result = await session.run(
        "MATCH (p:Paper {paperId: $paperId}) DETACH DELETE p RETURN p",
        { paperId }
      );
      
      if (result.records.length === 0) {
        return res.status(404).json({ error: "Paper not found" });
      }
      
      res.json({ message: "Deleted" });
    } catch (error) {
      console.error('Error in deletePaper:', error);
      res.status(500).json({ error: error.message });
    } finally {
      await session.close();
    }
  },
};
