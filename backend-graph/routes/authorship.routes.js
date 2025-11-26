const express = require("express");
const router = express.Router();
const controller = require("../controllers/authorship.controller");

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

router.post("/", controller.createAuthorship);
router.delete("/:authorId/:paperId", controller.deleteAuthorship);

module.exports = router;

