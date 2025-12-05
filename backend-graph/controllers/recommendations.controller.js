const axios = require('axios');
const ML_API_URL = 'http://localhost:8000';



exports.getRecommendations = async (req, res) => {
  try {
    const { researcher_id } = req.params;  // Ou renommer en author_id
    const { top_k = 10, strategy } = req.query;

    const response = await axios.get(
     `${ML_API_URL}/recommendations/${researcher_id}`,
     { 
       params: { 
         top_k, 
         filter_hubs: true,
         strategy 
       }, 
       timeout: 30000 
     }
    ); 

    res.json(response.data);
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'API ML non démarrée',
        message: 'Lancez: cd ml-api && python run_api.py'
      });
    }
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.detail || error.message
    });
  }
};

exports.getResearcherAnalytics = async (req, res) => {
  try {
    const { id } = req.params;
    const response = await axios.get(
      `${ML_API_URL}/analytics/author/${id}`,  // ✅ Changé de researcher à author
      { timeout: 30000 }
    );
    res.json(response.data);
  } catch (error) {
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.detail || error.message
    });
  }
};