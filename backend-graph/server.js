require("dotenv").config();
const express = require("express");
const cors = require("cors");

const researcherRoutes = require("./routes/researcher.routes");
const publicationRoutes = require("./routes/publication.routes");
const authorshipRoutes = require("./routes/authorship.routes");
const { getSession } = require("./config/neo4j");

const app = express();
app.use(cors());

// Middleware de nettoyage des URLs (supprime les caractères indésirables comme %0A)
// Nettoie l'URL brute avant qu'Express ne la parse
app.use((req, res, next) => {
  try {
    // Logger l'URL originale pour debug
    const originalUrl = req.url;
    
    // Nettoyer l'URL en supprimant %0A, %0D et autres caractères indésirables
    if (req.url) {
      // Remplacer directement les caractères encodés problématiques AVANT le décodage
      req.url = req.url
        .replace(/%0A/gi, '')
        .replace(/%0D/gi, '')
        .replace(/%09/gi, '') // tab
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
    
    // Nettoyer aussi req.path si disponible
    if (req.path) {
      req.path = req.path
        .replace(/%0A/gi, '')
        .replace(/%0D/gi, '')
        .replace(/%09/gi, '')
        .replace(/[\r\n\t]/g, '')
        .trim();
    }
    
    // Logger si l'URL a été modifiée
    if (originalUrl !== req.url) {
      console.log(`URL cleaned: ${originalUrl} -> ${req.url}`);
    }
    
    next();
  } catch (error) {
    // Si l'URL est malformée, continuer quand même
    console.warn('URL cleaning warning:', error.message);
    next();
  }
});

// Middleware JSON conditionnel - ne parse pas pour GET, HEAD, OPTIONS, DELETE
app.use((req, res, next) => {
  // Pour les méthodes qui n'ont pas besoin de body, on skip le parsing JSON
  if (['GET', 'HEAD', 'OPTIONS', 'DELETE'].includes(req.method)) {
    return next();
  }
  // Pour POST, PUT, PATCH, on parse le JSON
  express.json({ limit: '10mb' })(req, res, next);
});

// Routes
app.use("/authors", researcherRoutes); // Garde /authors pour compatibilité avec l'ancien code
app.use("/researchers", researcherRoutes); // Alias pour /authors
app.use("/papers", publicationRoutes); // Garde /papers pour compatibilité
app.use("/publications", publicationRoutes); // Alias pour /papers
app.use("/authorships", authorshipRoutes);

// Middleware de gestion d'erreur global (doit être après les routes)
app.use((err, req, res, next) => {
  // Erreur de parsing JSON
  if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
    return res.status(400).json({ 
      error: "Invalid JSON format", 
      message: err.message 
    });
  }
  
  // Autres erreurs
  if (err) {
    console.error('Error:', err);
    // S'assurer qu'une réponse n'a pas déjà été envoyée
    if (!res.headersSent) {
      return res.status(err.status || 500).json({ 
        error: err.message || "Internal server error" 
      });
    }
  }
  
  next();
});

// Route 404 pour les routes non trouvées
app.use((req, res) => {
  res.status(404).json({ error: "Route not found" });
});

// Lancer serveur
const PORT = process.env.PORT || 5000;

const server = app.listen(PORT, async () => {
  console.log(`Server running on port ${PORT}`);
  
  // Test de connexion à Neo4j
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

// Gestion des erreurs de connexion (ECONNRESET, etc.)
server.on('clientError', (err, socket) => {
  console.warn('Client error:', err.message);
  if (!socket.destroyed) {
    socket.end('HTTP/1.1 400 Bad Request\r\n\r\n');
  }
});

// Gestion des erreurs de requête
server.on('error', (err) => {
  console.error('Server error:', err.message);
});
