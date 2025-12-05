const express = require('express');
const router = express.Router();
const controller = require('../controllers/recommendations.controller');


router.get('/analytics/researcher/:id', controller.getResearcherAnalytics);
router.get('/:researcher_id', controller.getRecommendations);

module.exports = router;