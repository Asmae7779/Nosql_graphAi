const express = require("express");
const router = express.Router();
const controller = require("../controllers/publication.controller");

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

router.get("/", controller.getAllPapers);
router.post("/", controller.createPaper);
router.get("/:id/authors", controller.getPaperAuthors);
router.get("/:id", controller.getPaperById);
router.put("/:id", controller.updatePaper);
router.delete("/:id", controller.deletePaper);

module.exports = router;
