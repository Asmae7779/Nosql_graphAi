import React, { useState, useEffect } from 'react'
import SearchBar from '../components/SearchBar'
import { api } from '../services/api'

const Home = () => {
  const [stats, setStats] = useState(null)
  const [topAuthors, setTopAuthors] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      try {
        const [statsResponse, authorsResponse] = await Promise.all([
          api.getNetworkStats(),
          api.getTopAuthors(5)
        ])
        
        setStats(statsResponse.data)
        setTopAuthors(authorsResponse.data)
      } catch (error) {
        console.error('Error loading data:', error)
        // Garde les données mockées en fallback
        setStats({
          totalAuthors: 25000,
          totalPapers: 5000,
          totalCollaborations: 50000,
          latestYear: 2023,
          topFields: [{field: 'Computer Science', count: 1500}]
        })
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  const handleSearch = (query) => {
    window.location.href = `/explorer?search=${encodeURIComponent(query)}`
  }

  return (
    <div className="home-container">
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            Science Collaboration Network
          </h1>
          <p className="hero-subtitle">
            Explorez {stats ? stats.totalAuthors.toLocaleString() : '25,000'} chercheurs et {stats ? stats.totalPapers.toLocaleString() : '5,000'} publications
          </p>
          
          <div className="search-container">
            <SearchBar 
              onSearch={handleSearch} 
              placeholder="Rechercher un auteur, un papier, un domaine..." 
            />
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="stats-section">
        <h2 className="section-title">Notre Base de Données</h2>
        {loading ? (
          <div className="loading-state">
            <div className="loading-spinner"></div>
            <p>Chargement des données...</p>
          </div>
        ) : (
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-icon">👥</div>
              <div className="stat-number">{stats.totalAuthors.toLocaleString()}</div>
              <div className="stat-label">Chercheurs</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📄</div>
              <div className="stat-number">{stats.totalPapers.toLocaleString()}</div>
              <div className="stat-label">Publications</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🔗</div>
              <div className="stat-number">{stats.totalCollaborations.toLocaleString()}+</div>
              <div className="stat-label">Collaborations</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📅</div>
              <div className="stat-number">{stats.latestYear}</div>
              <div className="stat-label">Dernière année</div>
            </div>
          </div>
        )}
      </div>

      {/* Top Authors Section */}
      <div className="features-section">
        <h2 className="section-title">Auteurs les Plus Prolifiques</h2>
        {loading ? (
          <div className="loading-state">
            <p>Chargement...</p>
          </div>
        ) : (
          <div className="top-authors-grid">
            {topAuthors.map((author, index) => (
              <div key={author.authorId} className="top-author-card">
                <div className="author-rank">#{index + 1}</div>
                <h3 className="author-name">{author.name}</h3>
                <p className="author-stats">{author.publicationCount} publications</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Features Section */}
      <div className="features-section">
        <h2 className="section-title">Découvrez Notre Plateforme</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🔍</div>
            <h3>Exploration Avancée</h3>
            <p>Parcourez les chercheurs, publications et leurs relations complexes</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🏢</div>
            <h3>Communautés Scientifiques</h3>
            <p>Découvrez les groupes de recherche détectés par IA</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🤝</div>
            <h3>Recommandations Intelligentes</h3>
            <p>Trouvez des collaborations potentielles basées sur vos intérêts</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📈</div>
            <h3>Analyses Graphiques</h3>
            <p>Visualisez les réseaux de collaboration avec des métriques avancées</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Home