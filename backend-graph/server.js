const axios = require("axios");
require("dotenv").config();
const express = require("express");
const cors = require("cors");

const { createProxyMiddleware } = require("http-proxy-middleware");

const researcherRoutes = require("./routes/researcher.routes");
const publicationRoutes = require("./routes/publication.routes");
const authorshipRoutes = require("./routes/authorship.routes");
const networkRoutes = require("./routes/network.routes");
const { getSession } = require("./config/neo4j");

const app = express();

// ========== 1. CORS ==========
app.use(
  cors({
    origin: "http://localhost:3000",
    credentials: true,
  })
);

// ========== 2. JSON ==========
app.use(express.json({ limit: "10mb" }));

// ========== 3. ROUTES NEO4J (IMPORTANT : AVANT LES PROXYS ML !) ==========
app.use("/api/researchers", researcherRoutes);
app.use("/api/publications", publicationRoutes);
app.use("/api/authorships", authorshipRoutes);
app.use("/api/network", networkRoutes);

app.get("/api/test", (req, res) => {
  res.json({
    status: "OK",
    message: "Backend Neo4j API running",
    timestamp: new Date().toISOString(),
  });
});

// ========== 4. PROXYS ML (APRÈS LES ROUTES NEO4J) ==========
app.use(
  "/communities",
  createProxyMiddleware({
    target: "http://localhost:8000",
    changeOrigin: true,
    pathRewrite: { "^/communities": "/api/communities" },
  })
);

app.use(
  "/recommendations",
  createProxyMiddleware({
    target: "http://localhost:8000",
    changeOrigin: true,
  })
);

app.use(
  "/analytics",
  createProxyMiddleware({
    target: "http://localhost:8000",
    changeOrigin: true,
  })
);

// ========== 5. 404 ==========
app.use((req, res) => {
  res.status(404).json({ error: "Not found" });
});

// ========== 6. START ==========
app.listen(5000, () => {
  console.log("🚀 Backend running on http://localhost:5000");
});
