import React, { useState, useEffect } from 'react'
import SearchBar from '../components/SearchBar'
import AuthorCard from '../components/AuthorCard'
import PaperCard from '../components/PaperCard'
import { api } from '../services/api'

const Explorer = () => {
  const [searchResults, setSearchResults] = useState({ authors: [], papers: [] })
  const [loading, setLoading] = useState(false)
  const [searchPerformed, setSearchPerformed] = useState(false)

  const handleSearch = async (query) => {
    if (!query.trim()) return
    
    setLoading(true)
    setSearchPerformed(true)
    try {
      // Utilise la recherche unifiée
      const response = await api.unifiedSearch(query)
      
      console.log('🔍 [Explorer] Unified search response:', response.data)
      
      setSearchResults({
        authors: response.data.authors || [],
        papers: response.data.papers || []
      })
    } catch (error) {
      console.error('Erreur de recherche unifiée, fallback vers recherche normale:', error)
      
      // Fallback: utilise les recherches séparées
      try {
        const [authorsResponse, papersResponse] = await Promise.all([
          api.searchAuthors(query),
          api.searchPapers(query)
        ])
        
        setSearchResults({
          authors: authorsResponse.data || [],
          papers: papersResponse.data || []
        })
      } catch (fallbackError) {
        console.error('Erreur fallback:', fallbackError)
        setSearchResults({
          authors: [],
          papers: []
        })
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      <div className="explorer-header">
        <h1>🔍 Explorer la Base de Données</h1>
        <p>Recherchez parmi 24,905 chercheurs et 4,424 publications</p>
      </div>
      
      <div className="search-section">
        <SearchBar 
          onSearch={handleSearch} 
          placeholder="Rechercher un auteur, un papier, un domaine..." 
          size="large"
        />
      </div>
      
      {loading && (
        <div className="loading-state">
          <div className="loading-spinner"></div>
          <p>Recherche en cours...</p>
        </div>
      )}
      
      {searchPerformed && !loading && (
        <div className="results-section">
          <div className="results-grid">
            <div className="results-column">
              <h2>Auteurs ({searchResults.authors.length})</h2>
              {searchResults.authors.length > 0 ? (
                <div className="authors-list">
                  {searchResults.authors.map(author => (
                    <AuthorCard 
                      key={author.authorId} 
                      author={author} 
                    />
                  ))}
                </div>
              ) : (
                <div className="no-results">
                  <p>Aucun auteur trouvé</p>
                </div>
              )}
            </div>
            
            <div className="results-column">
              <h2>Publications ({searchResults.papers.length})</h2>
              {searchResults.papers.length > 0 ? (
                <div className="papers-list">
                  {searchResults.papers.map(paper => (
                    <PaperCard 
                      key={paper.paperId} 
                      paper={paper} 
                    />
                  ))}
                </div>
              ) : (
                <div className="no-results">
                  <p>Aucune publication trouvée</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {!searchPerformed && (
        <div className="welcome-section">
          <div className="welcome-card">
            <h3>💡 Comment effectuer une recherche ?</h3>
            <ul>
              <li>🔍 <strong>Par nom d'auteur</strong> : "Hochreiter", "Clevert", etc.</li>
              <li>📄 <strong>Par titre de publication</strong> : "Deep Learning", "Neural Networks"</li>
              <li>🏷️ <strong>Par domaine</strong> : "Computer Science", "Mathematics"</li>
              <li>📚 <strong>Par venue</strong> : "ICLR", "Science", etc.</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}

export default Explorer