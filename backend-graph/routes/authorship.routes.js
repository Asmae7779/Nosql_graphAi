const express = require("express");
const router = express.Router();
const controller = require("../controllers/authorship.controller");

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

// Routes pour gérer les relations Authorship
router.post("/", controller.createAuthorship);
router.delete("/:authorId/:paperId", controller.deleteAuthorship);

module.exports = router;

