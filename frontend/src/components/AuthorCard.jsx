import React, { useState, useEffect } from 'react'
import { api } from '../services/api'

const AuthorCard = ({ author, onClick }) => {
  const [publicationCount, setPublicationCount] = useState(0)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const loadPublicationCount = async () => {
      if (!author.authorId) return
      
      setLoading(true)
      try {
        console.log(`🔍 [AuthorCard] Chargement count pour authorId: ${author.authorId}, type: ${typeof author.authorId}`);
        
        const response = await api.getAuthorPublicationCount(author.authorId)
        
        setPublicationCount(response.data?.count || 0)
      } catch (error) {
        setPublicationCount(0)
      } finally {
        setLoading(false)
      }
    }

    loadPublicationCount()
  }, [author.authorId])

  return (
    <div 
      className="author-card"
      onClick={() => onClick && onClick(author)}
    >
      <h3 className="author-name">{author.name}</h3>
      
      <div className="author-meta">
        <div className="meta-item">
          <span className="meta-label"> ID:</span>
          <span className="meta-value">{author.authorId}</span>
        </div>
        
        <div className="meta-item">
          <span className="meta-label"> Publications:</span>
          <span className={`publication-count ${loading ? 'loading' : ''}`}>
            {loading ? (
              <span className="loading-dots">...</span>
            ) : (
              <span className="count-number">{publicationCount}</span>
            )}
          </span>
        </div>
      </div>
    </div>
  )
}

export default AuthorCard