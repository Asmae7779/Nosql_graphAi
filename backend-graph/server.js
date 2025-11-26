require("dotenv").config();
const express = require("express");
const cors = require("cors");

const researcherRoutes = require("./routes/researcher.routes");
const publicationRoutes = require("./routes/publication.routes");
const authorshipRoutes = require("./routes/authorship.routes");
const { getSession } = require("./config/neo4j");

const app = express();
app.use(cors());

app.use((req, res, next) => {
  try {
    const originalUrl = req.url;
    
    if (req.url) {
      req.url = req.url
        .replace(/%0A/gi, '')
        .replace(/%0D/gi, '')
        .replace(/%09/gi, '')
        .replace(/[\r\n\t]/g, '')
        .trim();
    }
    
    if (req.originalUrl) {
      req.originalUrl = req.originalUrl
        .replace(/%0A/gi, '')
        .replace(/%0D/gi, '')
        .replace(/%09/gi, '')
        .replace(/[\r\n\t]/g, '')
        .trim();
    }
    
    if (req.path) {
      req.path = req.path
        .replace(/%0A/gi, '')
        .replace(/%0D/gi, '')
        .replace(/%09/gi, '')
        .replace(/[\r\n\t]/g, '')
        .trim();
    }
    
    if (originalUrl !== req.url) {
      console.log(`URL cleaned: ${originalUrl} -> ${req.url}`);
    }
    
    next();
  } catch (error) {
    console.warn('URL cleaning warning:', error.message);
    next();
  }
});

app.use((req, res, next) => {
  if (['GET', 'HEAD', 'OPTIONS', 'DELETE'].includes(req.method)) {
    return next();
  }
  express.json({ limit: '10mb' })(req, res, next);
});

app.use("/authors", researcherRoutes);
app.use("/researchers", researcherRoutes);
app.use("/papers", publicationRoutes);
app.use("/publications", publicationRoutes);
app.use("/authorships", authorshipRoutes);

app.use((err, req, res, next) => {
  if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
    return res.status(400).json({ 
      error: "Invalid JSON format", 
      message: err.message 
    });
  }
  
  if (err) {
    console.error('Error:', err);
    if (!res.headersSent) {
      return res.status(err.status || 500).json({ 
        error: err.message || "Internal server error" 
      });
    }
  }
  
  next();
});

app.use((req, res) => {
  res.status(404).json({ error: "Route not found" });
});

const PORT = process.env.PORT || 5000;

const server = app.listen(PORT, async () => {
  console.log(`Server running on port ${PORT}`);
  
  const session = getSession();
  try {
    const result = await session.run("RETURN 1 AS test");
    console.log("✅ Connexion Neo4j réussie !");
  } catch (error) {
    console.error("❌ Erreur de connexion Neo4j:", error.message);
  } finally {
    await session.close();
  }
});

server.on('clientError', (err, socket) => {
  console.warn('Client error:', err.message);
  if (!socket.destroyed) {
    socket.end('HTTP/1.1 400 Bad Request\r\n\r\n');
  }
});

server.on('error', (err) => {
  console.error('Server error:', err.message);
});
