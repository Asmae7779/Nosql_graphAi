import React, { useState } from 'react'

const SearchBar = ({ onSearch, placeholder = "Rechercher...", size = "medium" }) => {
  const [query, setQuery] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (query.trim()) {
      onSearch(query.trim())
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit(e)
    }
  }

  const sizes = {
    small: { padding: '0.5rem 1rem', fontSize: '0.9rem' },
    medium: { padding: '0.75rem 1.5rem', fontSize: '1rem' },
    large: { padding: '1rem 2rem', fontSize: '1.1rem' }
  }

  return (
    <form onSubmit={handleSubmit} className="search-bar">
      <div className="search-container">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={placeholder}
          className="search-input"
          style={sizes[size]}
        />
        <button 
          type="submit" 
          className="search-button"
          style={sizes[size]}
          disabled={!query.trim()}
        >
          <span className="search-icon"></span>
          Rechercher
        </button>
      </div>
    </form>
  )
}

export default SearchBar