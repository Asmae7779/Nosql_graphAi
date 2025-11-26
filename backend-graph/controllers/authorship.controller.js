const { getSession } = require("../config/neo4j");

const sanitizeAuthorId = (id) => {
  let authorId = String(id).trim().replace(/[\r\n]/g, '');
  if (!authorId) return null;
  const authorIdNum = Number(authorId);
  if (!isNaN(authorIdNum) && authorIdNum.toString() === authorId) {
    return authorIdNum;
  }
  return authorId;
};

const sanitizePaperId = (id) => {
  const paperId = String(id).trim().replace(/[\r\n]/g, '');
  return paperId || null;
};

module.exports = {
  createAuthorship: async (req, res) => {
    const { authorId, paperId } = req.body;
    
    if (!authorId || !paperId) {
      return res.status(400).json({ error: "Les champs 'authorId' et 'paperId' sont requis" });
    }
    
    const session = getSession();
    try {
      const cleanAuthorId = sanitizeAuthorId(authorId);
      const cleanPaperId = sanitizePaperId(paperId);
      
      if (!cleanAuthorId || !cleanPaperId) {
        return res.status(400).json({ error: "Invalid authorId or paperId" });
      }
      
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

  deleteAuthorship: async (req, res) => {
    const session = getSession();
    try {
      const cleanAuthorId = sanitizeAuthorId(req.params.authorId);
      const cleanPaperId = sanitizePaperId(req.params.paperId);
      
      if (!cleanAuthorId || !cleanPaperId) {
        return res.status(400).json({ error: "Invalid authorId or paperId" });
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

