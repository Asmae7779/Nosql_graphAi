import axios from 'axios'

// ==================== CONFIG ====================
const BACKEND_BASE = 'http://localhost:5000'
const ML_API_BASE = 'http://localhost:8000'

// Backend Neo4j
const backend = axios.create({
  baseURL: BACKEND_BASE,
  headers: { 'Content-Type': 'application/json' }
})

// ML API directe
const ml = axios.create({
  baseURL: ML_API_BASE,
  headers: { 'Content-Type': 'application/json' }
})

// Logging
backend.interceptors.request.use(c => {
  console.log(`➡️ BACKEND ${c.method.toUpperCase()} ${c.url}`)
  return c
})

ml.interceptors.request.use(c => {
  console.log(`➡️ ML API ${c.method.toUpperCase()} ${c.url}`)
  return c
})

// ==================== API ====================
export const api = {

  // ----- TEST -----
  testBackend: () => backend.get('/api/test'),
  healthCheck: () => backend.get('/api/health'),

  // ----- NEO4J -----
  searchAuthors: (q) => backend.get(`/api/researchers/search?q=${encodeURIComponent(q)}`),
  searchPapers:  (q) => backend.get(`/api/publications/search?q=${encodeURIComponent(q)}`),
  unifiedSearch: (query) =>
  backend.get(`/api/researchers/unified-search?q=${encodeURIComponent(query)}`),


  getAuthor: (id) => backend.get(`/api/researchers/${id}`),
  getPaper:  (id) => backend.get(`/api/publications/${id}`),
  getPaperAuthors: (id) => backend.get(`/api/publications/${id}/authors`),
  getAuthorPublicationCount: (authorId) => backend.get(`api/researchers/${authorId}/publication-count`),
  getTopAuthors: (limit = 10) => backend.get(`api/researchers/top?limit=${limit}`),
  getTopPapers: (limit = 10) => backend.get(`api/publications/top?limit=${limit}`),
  getNetworkStats: () => backend.get(`/api/network/stats`),

  // ----- ML API DIRECT -----
  getCommunities: () => ml.get('/api/communities'),
  getCommunitySummary: () => ml.get('/api/communities/summary'),
  getCommunity: (id) => ml.get(`/api/communities/${id}`),

  getCollaborationRecommendations: (id) =>
  ml.get(`/recommendations/${id}?filter_hubs=true&top_k=10`),

  getResearcherAnalytics: (id) =>
    ml.get(`/analytics/author/${id}`),

  // Health direct ML
  testML: () => ml.get('/api/health')
}
