import React from 'react'
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import Home from './pages/Home'
import Explorer from './pages/Explorer'
import Recommendations from './pages/Recommendations'
import Communities from './pages/Communities'
import './styles/globals.css'

function App() {
  return (
    <Router>
      <div className="app">
        {/* Nouveau Header Élégant */}
        <nav className="navbar">
          <div className="nav-container">
            <div className="nav-logo">
              <h1>Science Collaboration Network</h1>
            </div>
            <div className="nav-menu">
              <NavLink 
                to="/" 
                className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
              >
                Accueil
              </NavLink>
              <NavLink 
                to="/explorer" 
                className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
              >
                Explorer
              </NavLink>
              <NavLink 
                to="/communities" 
                className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
              >
                Communautés
              </NavLink>
              <NavLink 
                to="/recommendations" 
                className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
              >
                Recommandations
              </NavLink>
            </div>
          </div>
        </nav>
        
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/communities" element={<Communities />} />
          <Route path="/recommendations" element={<Recommendations />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App