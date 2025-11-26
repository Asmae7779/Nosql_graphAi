const express = require("express");
const router = express.Router();
const controller = require("../controllers/publication.controller");

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

// Routes CRUD (routes spécifiques avant routes paramétrées)
router.get("/", controller.getAllPapers);
router.post("/", controller.createPaper);
router.get("/:id/authors", controller.getPaperAuthors); // Route spécifique avant :id
router.get("/:id", controller.getPaperById);
router.put("/:id", controller.updatePaper);
router.delete("/:id", controller.deletePaper);

module.exports = router;
