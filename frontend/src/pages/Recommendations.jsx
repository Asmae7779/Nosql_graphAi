import React, { useState } from 'react'
import SearchBar from '../components/SearchBar'
import AuthorCard from '../components/AuthorCard'
import axios from 'axios'
import { api } from '../services/api'
import './Recommendations.css'

const Recommendations = () => {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedAuthor, setSelectedAuthor] = useState(null)
  const [searchResults, setSearchResults] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingRecs, setLoadingRecs] = useState(false)
  const [error, setError] = useState(null)

  const DEMO_AUTHORS = [
    { authorId: '3308557', name: 'Sepp Hochreiter' },
    { authorId: '34917892', name: 'Djork-Arné Clevert' },
    { authorId: '2465270', name: 'Thomas Unterthiner' },
    { authorId: '145845739', name: 'S. Mousavi' },
    { authorId: '3462562', name: 'G. Beroza' }
  ]

  const handleSearch = async (query) => {
    setSearchQuery(query)
    setLoading(true)
    setError(null)
    
    try {
      if (!query.trim()) {
        setSearchResults([])
        return
      }
      
      const response = await api.searchAuthors(query)
      const authors = response.data || []
      
      setSearchResults(authors)
      
      if (authors.length === 0) {
        setError(`Aucun auteur trouvé pour "${query}". Essayez avec un nom complet.`)
      }
    } catch (error) {
      console.error('Erreur recherche:', error)
      const filtered = DEMO_AUTHORS.filter(author => 
        author.name.toLowerCase().includes(query.toLowerCase())
      )
      setSearchResults(filtered)
      
      if (filtered.length === 0) {
        setError(`Aucun auteur trouvé. Essayez: ${DEMO_AUTHORS.slice(0, 3).map(a => a.name).join(', ')}`)
      }
    } finally {
      setLoading(false)
    }
  }

  const handleAuthorSelect = async (author) => {
    console.log(`🎯 Sélection de l'auteur: ${author.authorId} - ${author.name}`)
    setSelectedAuthor(author)
    setLoadingRecs(true)
    setError(null)
    setRecommendations([])
    setAnalytics(null)
    
    try {
      const recsResponse = await axios.get(
        `http://localhost:8000/recommendations/${author.authorId}?filter_hubs=true&top_k=10`
      )
      
      const recsData = recsResponse.data.recommendations || []
      
      if (recsData.length === 0) {
        setError('Aucune recommandation disponible pour cet auteur')
        return
      }
      
      let analyticsData = null
      try {
        const analyticsResponse = await axios.get(`http://localhost:8000/analytics/author/${author.authorId}`)
        analyticsData = analyticsResponse.data
        console.log('✅ Analytics reçus')
      } catch (analyticsError) {
        console.warn('⚠️ Analytics non disponibles:', analyticsError.message)
      }

      const communityPromises = recsData.map(async (rec) => {
        let communityName = `Communauté ${rec.community_id || 'Inconnue'}`
        
        // Essayer de récupérer le nom de la communauté
        try {
          if (rec.community_id !== undefined && rec.community_id !== null) {
            // Appeler l'API pour obtenir les infos de la communauté
            const communityResponse = await axios.get(`http://localhost:8000/api/communities/${rec.community_id}`)
            if (communityResponse.data && communityResponse.data.name) {
              communityName = communityResponse.data.name
            }
          }
        } catch (e) {
          console.log(`⚠️ Impossible de récupérer la communauté ${rec.community_id}:`, e.message)
        }
        
        return communityName
      })
      
      const communityNames = await Promise.all(communityPromises)
      
      const enrichedRecs = recsData.map((rec, index) => ({
        authorId: rec.researcher_id,
        name: rec.name || `Researcher ${rec.researcher_id}`,
        score: rec.collaboration_score || rec.ml_score || 0,
        mlScore: rec.ml_score || 0,
        reason: getRecommendationReason(rec),
        degree: rec.degree || 0,
        degreePercentile: rec.degree_percentile || 0,
        community: rec.community_id || 'N/A',
        communityName: communityNames[index],
        sameCommunity: rec.same_community || false,
        commonPapers: rec.common_neighbors || 0,
        pagerank: rec.pagerank || 0,
        penalties: rec.penalties_applied || {}
      }))
      
      setRecommendations(enrichedRecs)
      setAnalytics(analyticsData)
      
    } catch (error) {
      console.error('❌ Erreur complète:', error)
      setError(`Erreur: ${error.message}`)
      setRecommendations([])
      setAnalytics(null)
    } finally {
      setLoadingRecs(false)
    }
  }

  const getRecommendationReason = (rec) => {
    if (rec.reason) return rec.reason
    if (rec.same_community) return 'Membre de la même communauté de recherche'
    if (rec.common_neighbors > 0) return `${rec.common_neighbors} collaborateurs en commun`
    return 'Recommandation basée sur l\'analyse du réseau de collaborations'
  }

  const getScoreCategory = (score) => {
    if (score >= 0.8) return 'Très élevé'
    if (score >= 0.6) return 'Élevé'
    if (score >= 0.4) return 'Moyen'
    return 'Faible'
  }

  const getScoreColor = (score) => {
    if (score >= 0.8) return '#2a824eff'
    if (score >= 0.6) return '#2ecc71'
    if (score >= 0.4) return '#f39c12'
    return '#e74c3c'
  }

  return (
    <div className="recommendations-container">
      {/* Header élégant */}
      <div className="recommendations-header">
        <div className="header-content">
          <h1 className="page-title">Recommandations de Collaborations Scientifiques</h1>
          <p className="page-subtitle">
            {selectedAuthor 
              ? `Analyse des collaborations potentielles pour le chercheur ${selectedAuthor.name}`
              : 'Sélectionnez un chercheur pour découvrir des collaborations potentielles'
            }
          </p>
        </div>
        
        {error && (
          <div className="error-alert">
            <p className="error-message">{error}</p>
          </div>
        )}
      </div>

      {/* Section de recherche */}
      <div className="search-section">
        <div className="search-card">
          <div className="search-header">
            <div className="step-indicator">
              <span className="step-number">1</span>
              <h2 className="section-title">Rechercher un Chercheur</h2>
            </div>
            <p className="search-instruction">
              Entrez le nom d'un chercheur pour analyser ses collaborations potentielles
            </p>
          </div>
          
          <div className="search-input-container">
            <SearchBar 
              onSearch={handleSearch}
              placeholder="Exemple: 'Hochreiter', 'Clevert', 'Mousavi'..."
              size="large"
            />
          </div>
          
          {loading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <p>Recherche en cours...</p>
            </div>
          )}
          
          {searchResults.length > 0 && (
            <div className="search-results">
              <h3 className="results-title">
              {searchResults.length} chercheur(s) trouvé(s)
              </h3>
               <p className="results-subtitle">
               Sélectionnez un chercheur pour voir les recommandations
               </p>
            <div className="authors-grid">
              {searchResults.map(author => (
                <div key={author.authorId} className="author-result-card">
                  <div className="author-card-content">
                    <div className="author-info">
                      <span className="author-name">{author.name}</span>
                        <span className="author-id">ID: {author.authorId}</span>
                       </div>
                     <button 
                      onClick={() => handleAuthorSelect(author)}
                     className="select-author-btn"
                    >
                      Analyser
                 </button>
                  </div>
               </div>
               ))}
            </div>
           </div>
          )}
          
          {!loading && searchQuery && searchResults.length === 0 && (
            <div className="no-results">
              <p className="no-results-text">Aucun résultat pour "{searchQuery}"</p>
              <div className="suggestions">
                <p className="suggestions-title">Suggestions :</p>
                <div className="suggestion-tags">
                  {DEMO_AUTHORS.map(author => (
                    <button
                      key={author.authorId}
                      className="suggestion-tag"
                      onClick={() => {
                        setSearchQuery(author.name)
                        handleSearch(author.name)
                      }}
                    >
                      {author.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Section des recommandations */}
      {selectedAuthor && (
        <div className="recommendations-section">
          <div className="recommendations-card">
            <div className="recommendations-header">
              <div className="recommendations-title-section">
                <div className="step-indicator">
                  <span className="step-number">2</span>
                  <h2 className="section-title">Recommandations de Collaboration</h2>
                </div>
                <p className="selected-author">
                  Pour le chercheur : <strong>{selectedAuthor.name}</strong>
                </p>
              </div>
              
              <button 
                onClick={() => {
                  setSelectedAuthor(null)
                  setRecommendations([])
                  setAnalytics(null)
                  setError(null)
                }}
                className="change-author-btn"
              >
                Changer de chercheur
              </button>
            </div>

            {loadingRecs ? (
              <div className="loading-recommendations">
                <div className="spinner large"></div>
                <div className="loading-text">
                  <p className="loading-title">Analyse en cours</p>
                  <p className="loading-subtitle">
                    Calcul des recommandations basées sur le réseau de collaborations...
                  </p>
                </div>
              </div>
            ) : (
              <div className="recommendations-content">
                {/* Résumé du chercheur */}
                {analytics && (
                  <div className="author-summary">
                    <h3 className="summary-title">Profil du Chercheur</h3>
                    <div className="summary-stats">
                      <div className="stat">
                        <div className="stat-value">{analytics.network_stats?.total_collaborators || 0}</div>
                        <div className="stat-label">Collaborateurs</div>
                      </div>
                      <div className="stat">
                        <div className="stat-value">
                          {analytics.influence_metrics?.degree || 0}
                        </div>
                        <div className="stat-label">Degré d'influence</div>
                      </div>
                      <div className="stat">
                        <div className="stat-value">
                          {analytics.network_stats?.community_size || 'N/A'}
                        </div>
                        <div className="stat-label">Taille communauté</div>
                      </div>
                      <div className="stat">
                        <div className="stat-value">
                          {analytics.influence_metrics?.pagerank 
                            ? (analytics.influence_metrics.pagerank * 10000).toFixed(1)
                            : 'N/A'}
                        </div>
                        <div className="stat-label">Score d'influence</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Légende des scores */}
                <div className="score-legend">
                  <h4 className="legend-title">Interprétation des scores :</h4>
                  <div className="legend-items">
                    <div className="legend-item">
                      <span className="legend-dot" style={{ backgroundColor: '#2ecc71' }}></span>
                      <span className="legend-text">Très élevé (80-100%)</span>
                    </div>
                    <div className="legend-item">
                      <span className="legend-dot" style={{ backgroundColor: '#27ae60' }}></span>
                      <span className="legend-text">Élevé (60-80%)</span>
                    </div>
                    <div className="legend-item">
                      <span className="legend-dot" style={{ backgroundColor: '#f39c12' }}></span>
                      <span className="legend-text">Moyen (40-60%)</span>
                    </div>
                    <div className="legend-item">
                      <span className="legend-dot" style={{ backgroundColor: '#e74c3c' }}></span>
                      <span className="legend-text">Faible (0-40%)</span>
                    </div>
                  </div>
                </div>

                {/* Statistiques globales */}
                {recommendations.length > 0 && (
                  <div className="global-stats">
                    <div className="stats-grid">
                      <div className="stat-card">
                        <div className="stat-card-value">
                          {(recommendations.reduce((a, b) => a + b.score, 0) / recommendations.length * 100).toFixed(1)}%
                        </div>
                        <div className="stat-card-label">Score moyen</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-card-value">{recommendations.length}</div>
                        <div className="stat-card-label">Recommandations</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-card-value">
                          {new Set(recommendations.map(r => r.community)).size}
                        </div>
                        <div className="stat-card-label">Communautés différentes</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-card-value">
                          {Math.max(...recommendations.map(r => r.commonPapers))}
                        </div>
                        <div className="stat-card-label">Max collaborateurs communs</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Liste des recommandations */}
                {recommendations.length > 0 ? (
                  <div className="recommendations-list">
                    <h3 className="list-title">Recommandations Classées</h3>
                    <p className="list-subtitle">Classement basé sur le potentiel de collaboration</p>
                    
                    <div className="recommendations-grid">
                      {recommendations.map((rec, index) => (
                        <RecommendationCard 
                          key={`${rec.authorId}-${index}`} 
                          recommendation={rec} 
                          rank={index + 1}
                          selectedAuthor={selectedAuthor}
                        />
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="no-recommendations">
                    <div className="no-data-icon">💡</div>
                    <h3>Aucune recommandation disponible</h3>
                    <p>
                      Nous n'avons pas pu générer de recommandations pour ce chercheur. 
                      Cela peut être dû à un manque de données de collaborations.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Section informative */}
      {!selectedAuthor && (
        <div className="info-section">
          <div className="info-card">
            <h3 className="info-title">Comment fonctionne notre système ?</h3>
            <div className="info-grid">
              <div className="info-item">
                <div className="info-item-header">
                  <div className="info-item-icon">📊</div>
                  <h4>Analyse du Réseau</h4>
                </div>
                <p>Nous analysons les réseaux de collaborations existants entre chercheurs</p>
              </div>
              <div className="info-item">
                <div className="info-item-header">
                  <div className="info-item-icon">🤝</div>
                  <h4>Communautés Scientifiques</h4>
                </div>
                <p>Identification des communautés de recherche via l'algorithme Louvain</p>
              </div>
              <div className="info-item">
                <div className="info-item-header">
                  <div className="info-item-icon">🎯</div>
                  <h4>Apprentissage Automatique</h4>
                </div>
                <p>Utilisation de GraphSAGE pour prédire les collaborations potentielles</p>
              </div>
              <div className="info-item">
                <div className="info-item-header">
                  <div className="info-item-icon">⚖️</div>
                  <h4>Diversification</h4>
                </div>
                <p>Évite les recommandations évidentes et favorise la diversité</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Composant de carte de recommandation élégante
const RecommendationCard = ({ recommendation, rank, selectedAuthor }) => {
  const getScoreColor = (score) => {
    if (score >= 0.8) return '#2ecc71'
    if (score >= 0.6) return '#27ae60'
    if (score >= 0.4) return '#f39c12'
    return '#e74c3c'
  }

  const getScoreCategory = (score) => {
    if (score >= 0.8) return 'Très élevé'
    if (score >= 0.6) return 'Élevé'
    if (score >= 0.4) return 'Moyen'
    return 'Faible'
  }

  const getCommunityColor = (communityId) => {
    const colors = ['#3498db', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    return colors[communityId % colors.length] || '#95a5a6'
  }

  return (
    <div className="recommendation-card">
      {/* En-tête avec rang et nom */}
      <div className="card-header">
        <div className="card-rank">
          <span className="rank-number">#{rank}</span>
        </div>
        <div className="card-author">
          <h3 className="author-name">{recommendation.name}</h3>
          <div className="author-id">ID: {recommendation.authorId}</div>
        </div>
      </div>
      
      {/* Score principal */}
      <div className="card-score-section">
        <div className="score-display">
          <div 
            className="score-circle"
            style={{ 
              backgroundColor: getScoreColor(recommendation.score),
              borderColor: getScoreColor(recommendation.score)
            }}
          >
            <span className="score-value">{(recommendation.score * 100).toFixed(1)}%</span>
          </div>
          <div className="score-info">
            <div className="score-category">{getScoreCategory(recommendation.score)}</div>
            <div className="score-label">Potentiel de collaboration</div>
          </div>
        </div>
        
        {/* Barre de progression */}
        <div className="score-bar-container">
          <div className="score-bar">
            <div 
              className="score-bar-fill"
              style={{ 
                width: `${recommendation.score * 100}%`,
                backgroundColor: getScoreColor(recommendation.score)
              }}
            ></div>
          </div>
          <div className="score-bar-labels">
            <span>0%</span>
            <span>100%</span>
          </div>
        </div>
      </div>
      
      {/* Métriques détaillées */}
      <div className="card-metrics">
        <div className="metrics-grid">
          <div className="metric">
            <div className="metric-header">
              <span className="metric-title">Collaborateurs Communs</span>
            </div>
            <div className="metric-value">{recommendation.commonPapers}</div>
            <div className="metric-description">Auteurs en commun</div>
          </div>
          
          <div className="metric">
            <div className="metric-header">
              <span className="metric-title">Degré d'Influence</span>
            </div>
            <div className="metric-value">{recommendation.degree}</div>
            <div className="metric-description">
              {recommendation.degreePercentile > 0 && `Top ${recommendation.degreePercentile.toFixed(1)}%`}
            </div>
          </div>
          
          <div className="metric">
            <div className="metric-header">
              <span className="metric-title">Confiance Modèle</span>
            </div>
            <div className="metric-value">{(recommendation.mlScore * 100).toFixed(1)}%</div>
            <div className="metric-description">Score d'apprentissage</div>
          </div>
          
          {recommendation.community !== 'N/A' && (
            <div className="metric">
              <div className="metric-header">
                <span className="metric-title">Communauté</span>
                {recommendation.sameCommunity && (
                  <span className="community-badge">Même domaine</span>
                )}
              </div>
              <div 
                className="metric-value community-value"
                style={{ color: getCommunityColor(recommendation.community) }}
              >
                {recommendation.communityName || `#${recommendation.community}`}
              </div>
              <div className="metric-description">
                {recommendation.sameCommunity 
                  ? `Similaire à ${selectedAuthor?.name?.split(' ')[0] || 'vous'}`
                  : 'Domaine complémentaire'
                }
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Raison de la recommandation */}
      <div className="card-reason">
        <div className="reason-header">
          <h4 className="reason-title">Pourquoi cette recommandation ?</h4>
        </div>
        <p className="reason-text">{recommendation.reason}</p>
      </div>
      
      {/* Métriques supplémentaires */}
      <div className="card-additional-metrics">
        <div className="additional-metrics-grid">
          <div className="additional-metric">
            <span className="additional-label">PageRank :</span>
            <span className="additional-value">
              {recommendation.pagerank ? (recommendation.pagerank * 10000).toFixed(2) : 'N/A'}
            </span>
          </div>
          <div className="additional-metric">
            <span className="additional-label">Score ML brut :</span>
            <span className="additional-value">{(recommendation.mlScore * 100).toFixed(1)}%</span>
          </div>
          {recommendation.penalties && recommendation.penalties.hub_penalty > 0 && (
            <div className="additional-metric">
              <span className="additional-label">Ajustement :</span>
              <span className="additional-value">-{(recommendation.penalties.hub_penalty * 100).toFixed(1)}%</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Recommendations