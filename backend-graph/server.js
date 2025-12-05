require("dotenv").config();
const express = require("express");
const cors = require("cors");

const researcherRoutes = require("./routes/researcher.routes");
const publicationRoutes = require("./routes/publication.routes");
const authorshipRoutes = require("./routes/authorship.routes");
const communitiesRoutes = require('./routes/communities');
const recommendationsRoutes = require('./routes/recommendations.routes');
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
app.use('/communities', communitiesRoutes);
app.use('/recommendations', recommendationsRoutes);

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

app.get("/model", (req, res) => {
  const modelPath = path.join(__dirname, "../results/link_prediction_model.pt");

  // Vérifier si le fichier existe
  fs.readFile(modelPath, "utf8", (err, data) => {
    if (err) {
      console.error('Erreur lors de la lecture du fichier:', err);
      return res.status(500).json({ error: "Erreur lors de la lecture du modèle" });
    }
    
    // Retourner le fichier ou un résumé des données (dépend de votre utilisation)
    res.json({ message: "Modèle chargé avec succès", data: data });
  });
});

app.get("/communities", (req, res) => {
  const partitionPath = path.join(__dirname, "../results/communities/louvain_partition.pkl");

  fs.readFile(partitionPath, "utf8", (err, data) => {
    if (err) {
      console.error('Erreur lors de la lecture du fichier:', err);
      return res.status(500).json({ error: "Erreur lors de la lecture des partitions de la communauté" });
    }

    res.json({ message: "Communautés chargées avec succès", data: JSON.parse(data) });
  });
});

// Endpoint pour récupérer le résumé de l'entraînement
app.get("/training-summary", (req, res) => {
  const summaryPath = path.join(__dirname, "../results/training_summary.json");

  fs.readFile(summaryPath, "utf8", (err, data) => {
    if (err) {
      console.error('Erreur lors de la lecture du fichier:', err);
      return res.status(500).json({ error: "Erreur lors de la lecture du résumé d'entraînement" });
    }

    res.json({ message: "Résumé d'entraînement chargé avec succès", data: JSON.parse(data) });
  });
});

// Endpoint pour récupérer les chercheurs d'exemple
app.get("/example-researchers", (req, res) => {
  const exampleResearchersPath = path.join(__dirname, "../results/example_researchers.json");

  fs.readFile(exampleResearchersPath, "utf8", (err, data) => {
    if (err) {
      console.error('Erreur lors de la lecture du fichier:', err);
      return res.status(500).json({ error: "Erreur lors de la lecture des chercheurs d'exemple" });
    }

    res.json({ message: "Chercheurs d'exemple chargés avec succès", data: JSON.parse(data) });
  });
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
