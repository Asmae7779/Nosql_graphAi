const express = require('express');
const router = express.Router();
const fs = require('fs').promises;
const path = require('path');
const controller = require('../controllers/communities.controller');
// Chemin vers les fichiers JSON
const COMMUNITIES_PATH = path.join(__dirname, '../python-results/communities_api.json');
const AUTHOR_MAP_PATH = path.join(__dirname, '../python-results/author_community_map.json');
const SUMMARY_PATH = path.join(__dirname, '../python-results/summary.json');

// Cache des données (évite de lire le fichier à chaque requête)
let communitiesCache = null;
let authorMapCache = null;
let summaryCache = null;

// Fonction pour charger les données
async function loadData() {
  if (!communitiesCache) {
    const data = await fs.readFile(COMMUNITIES_PATH, 'utf8');
    communitiesCache = JSON.parse(data);
  }
  if (!authorMapCache) {
    const data = await fs.readFile(AUTHOR_MAP_PATH, 'utf8');
    authorMapCache = JSON.parse(data);
  }
  if (!summaryCache) {
    const data = await fs.readFile(SUMMARY_PATH, 'utf8');
    summaryCache = JSON.parse(data);
  }
}

// Route 1 : GET /communities - Liste toutes les communautés
router.get('/', async (req, res) => {
  try {
    await loadData();
    res.json(communitiesCache);
  } catch (error) {
    res.status(500).json({ 
      error: 'Erreur lors du chargement des communautés',
      details: error.message 
    });
  }
});

// Route 2 : GET /communities/summary - Résumé des communautés
router.get('/summary', async (req, res) => {
  try {
    await loadData();
    res.json(summaryCache);
  } catch (error) {
    res.status(500).json({ 
      error: 'Erreur lors du chargement du résumé',
      details: error.message 
    });
  }
});

// Route 3 : GET /communities/:id - Détails d'une communauté spécifique
router.get('/:id', async (req, res) => {
  try {
    await loadData();
    const communityId = parseInt(req.params.id);
    
    const community = communitiesCache.communities.find(
      c => c.id === communityId
    );
    
    if (!community) {
      return res.status(404).json({ 
        error: 'Communauté non trouvée',
        communityId 
      });
    }
    
    res.json(community);
  } catch (error) {
    res.status(500).json({ 
      error: 'Erreur lors du chargement de la communauté',
      details: error.message 
    });
  }
});

// Route 4 : GET /communities/author/:authorId - Communauté d'un auteur
router.get('/author/:authorId', async (req, res) => {
  try {
    await loadData();
    const authorId = req.params.authorId;
    
    const authorInfo = authorMapCache[authorId];
    
    if (!authorInfo) {
      return res.status(404).json({ 
        error: 'Auteur non trouvé',
        authorId 
      });
    }
    
    // Récupérer la communauté complète
    const community = communitiesCache.communities.find(
      c => c.id === authorInfo.community_id
    );
    
    res.json({
      author: {
        authorId,
        name: authorInfo.name,
        communityId: authorInfo.community_id
      },
      community
    });
  } catch (error) {
    res.status(500).json({ 
      error: 'Erreur lors du chargement de la communauté de l\'auteur',
      details: error.message 
    });
  }
});



// Route 5 : GET /communities/largest/:n - Les n plus grandes communautés
router.get('/largest/:n', async (req, res) => {
  try {
    await loadData();
    const n = parseInt(req.params.n) || 10;
    
    const sortedCommunities = [...communitiesCache.communities]
      .sort((a, b) => b.size - a.size)
      .slice(0, n);
    
    res.json({
      count: sortedCommunities.length,
      communities: sortedCommunities
    });
  } catch (error) {
    res.status(500).json({ 
      error: 'Erreur lors du chargement des plus grandes communautés',
      details: error.message 
    });
  }
});

// Route 6 : POST /communities/reload - Recharger les données
router.post('/reload', async (req, res) => {
  try {
    communitiesCache = null;
    authorMapCache = null;
    summaryCache = null;
    
    await loadData();
    
    res.json({ 
      message: 'Données rechargées avec succès',
      summary: summaryCache
    });
  } catch (error) {
    res.status(500).json({ 
      error: 'Erreur lors du rechargement des données',
      details: error.message 
    });
  }
});

router.get('/:communityId/graph', controller.getCommunityGraph);


module.exports = router;