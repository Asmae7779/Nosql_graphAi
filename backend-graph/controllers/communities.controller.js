const axios = require('axios');
const ML_API_URL = 'http://localhost:8000';

exports.getCommunityGraph = async (req, res) => {
  try {
    const { communityId } = req.params;

    const response = await axios.get(
      `${ML_API_URL}/communities/${communityId}/graph`,
      { timeout: 30000 }
    );

    res.json(response.data);

  } catch (error) {
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.detail || error.message
    });
  }
};
