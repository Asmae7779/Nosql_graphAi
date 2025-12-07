import React, { useState, useEffect } from 'react'
import { api } from '../services/api'

const PaperCard = ({ paper, onClick }) => {
  const [authors, setAuthors] = useState([])
  const [loadingAuthors, setLoadingAuthors] = useState(false)
  const [showAllAuthors, setShowAllAuthors] = useState(false)

  // Charge les auteurs quand le composant est monté
  useEffect(() => {
    const loadAuthors = async () => {
      if (!paper.paperId) return
      
      setLoadingAuthors(true)
      try {
        const response = await api.getPaperAuthors(paper.paperId)
        setAuthors(response.data || [])
      } catch (error) {
        console.error('Error loading authors:', error)
        setAuthors([])
      } finally {
        setLoadingAuthors(false)
      }
    }

    loadAuthors()
  }, [paper.paperId])

  const toggleAuthors = () => {
    setShowAllAuthors(!showAllAuthors)
  }

  const getVisibleAuthors = () => {
    if (showAllAuthors || authors.length <= 5) {
      return authors
    }
    return authors.slice(0, 3)
  }

  const hasMoreAuthors = authors.length > 5 && !showAllAuthors

  return (
    <div 
      className="paper-card"
      onClick={() => onClick && onClick(paper)}
    >
      <h3 className="paper-title">{paper.title}</h3>
      
      <div className="paper-meta">
        <div className="meta-item">
          <span className="meta-label"> Année:</span>
          <span className="meta-value">{paper.year || 'N/A'}</span>
        </div>
        
        {paper.venue && (
          <div className="meta-item">
            <span className="meta-label"> Venue:</span>
            <span className="meta-value">{paper.venue}</span>
          </div>
        )}
        
        {paper.fieldsOfStudy && (
          <div className="meta-item">
            <span className="meta-label"> Domaines:</span>
            <span className="meta-value">{paper.fieldsOfStudy}</span>
          </div>
        )}
      </div>

      {/* Section Auteurs */}
      <div className="authors-section">
        <div className="authors-header">
          <span className="meta-label"> Auteurs ({authors.length}):</span>
          {loadingAuthors && <span className="loading-text">Chargement...</span>}
        </div>
        
        {!loadingAuthors && authors.length > 0 && (
          <div className="authors-container">
            <div className="authors-list">
              {getVisibleAuthors().map((author, index) => (
                <span key={`${author.authorId}-${index}`} className="author-tag">
                  {author.name}
                </span>
              ))}
            </div>
            
            {/* Bouton pour voir plus/moins d'auteurs */}
            {authors.length > 5 && (
              <button 
                className="toggle-authors-btn"
                onClick={(e) => {
                  e.stopPropagation() // Empêche le clic de déclencher onClick du paper
                  toggleAuthors()
                }}
              >
                {showAllAuthors ? (
                  <>
                    <span className="toggle-icon">▲</span>
                    Voir moins
                  </>
                ) : (
                  <>
                    <span className="toggle-icon">▼</span>
                    Voir les {authors.length - 3} autres auteurs
                  </>
                )}
              </button>
            )}
          </div>
        )}
        
        {!loadingAuthors && authors.length === 0 && (
          <span className="no-authors">Aucun auteur trouvé</span>
        )}
      </div>
    </div>
  )
}

export default PaperCard