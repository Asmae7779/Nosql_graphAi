import React, { useState, useEffect } from 'react'
import { api } from '../services/api'
import axios from 'axios'

const Communities = () => {
  const [communities, setCommunities] = useState([])
  const [selectedCommunity, setSelectedCommunity] = useState(null)
  const [communityDetails, setCommunityDetails] = useState(null)
  const [loading, setLoading] = useState(true)
  const [loadingDetails, setLoadingDetails] = useState(false)
  const [viewMode, setViewMode] = useState('list')
  const [summary, setSummary] = useState(null)

  // Fonction utilitaire pour fallback
  const fetchWithFallback = async (apiCall, directCall) => {
    try {
      const response = await apiCall()
      return response.data
    } catch (error) {
      console.warn('Proxy failed, trying direct connection...')
      try {
        const response = await directCall()
        return response.data
      } catch (directError) {
        console.error('Direct connection also failed:', directError)
        throw directError
      }
    }
  }

  // Fonction pour enrichir les données avec les vrais noms
const fetchAuthorName = async (authorId) => {
  try {
    // Essayer de trouver le nom dans différentes sources
    console.log(`🔍 Recherche nom pour ${authorId}`)
    
    // 1. Essayer avec l'ID tel quel
    try {
      const response = await api.getAuthor(authorId)
      if (response.data && response.data.name) {
        console.log(`✅ Nom trouvé via ID: ${response.data.name}`)
        return response.data.name
      }
    } catch (error) {
      console.log(`❌ ID ${authorId} non trouvé dans Neo4j`)
    }
    
    // 2. Si l'ID est numérique, essayer comme int
    if (!isNaN(authorId)) {
      try {
        const response = await api.getAuthor(parseInt(authorId))
        if (response.data && response.data.name) {
          console.log(`✅ Nom trouvé via int: ${response.data.name}`)
          return response.data.name
        }
      } catch (error) {
        // Ignorer
      }
    }
    
    // 3. Chercher par nom si disponible dans les données ML
    // (Vous devriez stocker les noms dans votre modèle ML)
    
    // Fallback: retourner un nom générique
    return `Researcher ${authorId}`
    
  } catch (error) {
    console.error(`❌ Erreur recherche nom ${authorId}:`, error)
    return `Researcher ${authorId}`
  }
}

// Modifiez enrichCommunityData :
const enrichCommunityData = async (communitiesData) => {
  try {
    if (!communitiesData || !communitiesData.communities) {
      return communitiesData
    }

    // Parcourir chaque communauté et enrichir les membres
    const enrichedCommunities = await Promise.all(
      communitiesData.communities.map(async (community) => {
        const enrichedMembers = await Promise.all(
          (community.top_members || []).map(async (member) => {
            try {
              const name = await fetchAuthorName(member.author_id)
              return {
                ...member,
                name: name
              }
            } catch (error) {
              return {
                ...member,
                name: `Researcher ${member.author_id}`
              }
            }
          })
        )
        
        return {
          ...community,
          top_members: enrichedMembers
        }
      })
    )

    return { ...communitiesData, communities: enrichedCommunities }
  } catch (error) {
    console.error('Erreur enrichissement:', error)
    return communitiesData
  }
}

  const generateCommunityName = (community, authorNames = {}) => {
    // Si la communauté a déjà un nom personnalisé, le garder
    if (community.name && 
        !community.name.includes('Research Community') && 
        !community.name.includes('Author')) {
      return community.name
    }

    // Utiliser les noms enrichis si disponibles
    const topMembers = community.top_members || []
    const topNames = topMembers.slice(0, 3).map(m => 
      authorNames[m.author_id] || m.name || `Author ${m.author_id}`
    )

    // Chercher des mots-clés dans les noms
    const keywords = {
      'AI Research': ['hinton', 'bengio', 'lecun', 'goodfellow', 'silver'],
      'Deep Learning': ['hochreiter', 'schmidhuber', 'krizhevsky'],
      'Computer Vision': ['krizhevsky', 'szegedy', 'zisserman'],
      'NLP': ['manning', 'collobert', 'mikolov', 'pennington'],
      'Physics': ['einstein', 'feynman', 'hawking', 'schrödinger'],
      'Mathematics': ['gauss', 'euler', 'newton', 'leibniz'],
      'Neuroscience': ['cajal', 'hubel', 'wiesel', 'kandel'],
      'Bioinformatics': ['venter', 'collins', 'watson', 'crick']
    }

    for (const [domain, names] of Object.entries(keywords)) {
      for (const name of topNames) {
        if (names.some(keyword => name.toLowerCase().includes(keyword.toLowerCase()))) {
          return `${domain} Network`
        }
      }
    }

    // Sinon, utiliser le domaine principal ou le premier auteur
    if (community.topics && community.topics.length > 0) {
      return `${community.topics[0]} Research Community`
    }

    if (topNames.length > 0) {
      const lastName = topNames[0].split(' ').pop()
      return `${lastName} Research Group`
    }

    return `Research Community ${community.id}`
  }

  // Charger la liste des communautés
  useEffect(() => {
const loadCommunities = async () => {
  setLoading(true)
  console.log('🏢 Début chargement communautés...')
  
  try {
    // Appel direct à l'API ML (port 8000)
    console.log('🔄 Tentative via API ML (port 8000)...')
    
    const [communitiesResponse, summaryResponse] = await Promise.all([
      axios.get('http://localhost:8000/api/communities'),
      axios.get('http://localhost:8000/api/communities/summary')
    ])
    
    console.log('✅ Données reçues de l\'API ML:')
    
    let communitiesData = communitiesResponse.data.communities || []
    const summaryData = summaryResponse.data
    
    // Filtrer pour avoir des communautés significatives
    const filteredCommunities = communitiesData
      .filter(comm => comm.size >= 5 && comm.density > 0.1)
      .sort((a, b) => b.size - a.size)
    
    console.log(`📊 ${filteredCommunities.length} communautés significatives`)
    
    setCommunities(filteredCommunities)
    setSummary(summaryData)
    
  } catch (error) {
    console.error('❌ Erreur chargement communautés:', error)
    // Fallback avec les données de l'exemple curl
    const mockData = {
      communities: [
        {
          id: 0,
          name: "AI Research Network",
          size: 959,
          density: 0.15,
          topics: ["Computer Science", "Artificial Intelligence", "Mathematics"],
          top_members: [
            { author_id: "3308557", name: "Sepp Hochreiter", publication_count: 17, centrality_score: 0.95 },
            { author_id: "34917892", name: "Djork-Arné Clevert", publication_count: 15, centrality_score: 0.88 },
            { author_id: "2465270", name: "Thomas Unterthiner", publication_count: 27, centrality_score: 0.85 }
          ],
          description: "Large AI research community with 959 members"
        }
      ],
      total_communities: 1,
      total_authors: 959
    }
    
    setCommunities(mockData.communities)
    setSummary({
      total_communities: 1,
      total_authors: 959,
      average_size: 959,
      average_density: 0.15
    })
  } finally {
    setLoading(false)
  }
}

    loadCommunities()
  }, [])

  // Charger les détails d'une communauté
  const loadCommunityDetails = async (community) => {
    setSelectedCommunity(community)
    setLoadingDetails(true)
    try {
      const response = await api.getCommunity(community.id)
      const details = response.data
      
      console.log('📋 Détails communauté bruts:', details)
      
      // Enrichir les noms des membres
      const enrichedMembers = await enrichMembersWithNames(details.top_members || [])
      
      setCommunityDetails({
        members: enrichedMembers,
        collaborationNetwork: {
          nodes: details.size || 0,
          edges: details.internal_edges || 0,
          averageDegree: details.average_degree || details.density ? Math.round(details.density * 10) / 10 : 0
        },
        metrics: details.metrics || {}
      })
      
    } catch (error) {
      console.error('Erreur détails communauté:', error)
      // Fallback direct
      try {
        const response = await axios.get(`http://localhost:8000/api/communities/${community.id}`, { timeout: 5000 })
        const details = response.data
        
        const enrichedMembers = await enrichMembersWithNames(details.top_members || [])
        
        setCommunityDetails({
          members: enrichedMembers,
          collaborationNetwork: {
            nodes: details.size || 0,
            edges: details.internal_edges || 0,
            averageDegree: details.average_degree || 0
          }
        })
      } catch (fallbackError) {
        console.error('Fallback échoué:', fallbackError)
        setCommunityDetails(getMockCommunityDetails(community.id))
      }
    } finally {
      setLoadingDetails(false)
    }
  }

  // Fonction pour enrichir les membres avec leurs vrais noms
  const enrichMembersWithNames = async (members) => {
    if (!members || members.length === 0) return members
    
    try {
      const enrichedMembers = []
      
      for (const member of members.slice(0, 20)) { // Limiter à 20 membres
        try {
          const authorId = member.author_id || member.authorId
          if (authorId) {
            const response = await api.getAuthor(authorId)
            enrichedMembers.push({
              authorId: authorId,
              name: response.data.name || member.name || `Author ${authorId}`,
              papers: member.publication_count || member.papers || 0,
              centrality: member.centrality_score || member.centrality || 0.5
            })
          } else {
            enrichedMembers.push({
              ...member,
              name: member.name || `Author ${member.author_id}`
            })
          }
        } catch (memberError) {
          enrichedMembers.push({
            authorId: member.author_id,
            name: member.name || `Author ${member.author_id}`,
            papers: member.publication_count || 0,
            centrality: member.centrality_score || 0.5
          })
        }
      }
      
      return enrichedMembers
    } catch (error) {
      console.error('Erreur enrichissement membres:', error)
      return members.map(member => ({
        authorId: member.author_id,
        name: member.name || `Author ${member.author_id}`,
        papers: member.publication_count || 0,
        centrality: member.centrality_score || 0.5
      }))
    }
  }

  // Données mockées temporaires
  const getMockCommunities = () => {
    return [
      {
        id: 1,
        name: "Deep Learning Research",
        size: 156,
        density: 0.82,
        topics: ["Computer Science", "AI", "Mathematics"],
        topMembers: [
          { author_id: "3308557", name: "Sepp Hochreiter", publication_count: 45, centrality_score: 0.95 },
          { author_id: "34917892", name: "Djork-Arné Clevert", publication_count: 32, centrality_score: 0.88 }
        ],
        description: "Communauté dédiée aux réseaux de neurones profonds"
      },
      {
        id: 2,
        name: "Computational Physics",
        size: 89,
        density: 0.75,
        topics: ["Physics", "Computational Science"],
        topMembers: [
          { author_id: "145845739", name: "S. Mousavi", publication_count: 31, centrality_score: 0.92 },
          { author_id: "3462562", name: "G. Beroza", publication_count: 28, centrality_score: 0.87 }
        ],
        description: "Physique computationnelle et modélisation numérique"
      }
    ]
  }

  const getMockCommunityDetails = (communityId) => {
    const mockData = {
      1: {
        members: [
          { authorId: '3308557', name: 'Sepp Hochreiter', papers: 45, centrality: 0.95 },
          { authorId: '34917892', name: 'Djork-Arné Clevert', papers: 32, centrality: 0.88 }
        ],
        collaborationNetwork: { nodes: 156, edges: 1200, averageDegree: 8.0 }
      },
      2: {
        members: [
          { authorId: '145845739', name: 'S. Mousavi', papers: 31, centrality: 0.92 },
          { authorId: '3462562', name: 'G. Beroza', papers: 28, centrality: 0.87 }
        ],
        collaborationNetwork: { nodes: 89, edges: 650, averageDegree: 7.3 }
      }
    }
    return mockData[communityId] || {
      members: [],
      collaborationNetwork: { nodes: 0, edges: 0, averageDegree: 0 }
    }
  }

  const getDensityColor = (density) => {
    if (density >= 0.7) return '#27ae60'
    if (density >= 0.4) return '#f39c12'
    return '#e74c3c'
  }

  const getCentralityColor = (centrality) => {
    if (centrality >= 0.8) return '#e74c3c'
    if (centrality >= 0.6) return '#f39c12'
    return '#27ae60'
  }

  // Rendu des statistiques
  const getStats = () => {
    if (summary) {
      return {
        totalCommunities: summary.total_communities || communities.length,
        totalAuthors: summary.total_authors || communities.reduce((acc, comm) => acc + (comm.size || 0), 0),
        averageDensity: summary.average_density 
          ? `${(summary.average_density * 100).toFixed(1)}%` 
          : communities.length > 0
            ? `${(communities.reduce((acc, comm) => acc + (comm.density || 0), 0) / communities.length * 100).toFixed(1)}%`
            : '0%',
        largestSize: summary.largest_community?.size || 
          (communities.length > 0 ? Math.max(...communities.map(c => c.size || 0)) : 0)
      }
    }
    
    return {
      totalCommunities: communities.length,
      totalAuthors: communities.reduce((acc, comm) => acc + (comm.size || 0), 0),
      averageDensity: communities.length > 0 
        ? `${(communities.reduce((acc, comm) => acc + (comm.density || 0), 0) / communities.length * 100).toFixed(1)}%`
        : '0%',
      largestSize: communities.length > 0 ? Math.max(...communities.map(c => c.size || 0)) : 0
    }
  }

  const stats = getStats()

  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1>🏢 Communautés Scientifiques</h1>
        <p style={{ color: '#666' }}>
          {summary?.description || 'Découvrez les communautés de chercheurs détectées par notre algorithme de Graph Machine Learning'}
        </p>
      </div>

      {/* Statistiques globales */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '1rem',
        marginBottom: '2rem'
      }}>
        <StatCard 
          title="Communautés" 
          value={stats.totalCommunities} 
          icon="🏢"
          color="#3498db"
        />
        <StatCard 
          title="Chercheurs" 
          value={stats.totalAuthors} 
          icon="👥"
          color="#27ae60"
        />
        <StatCard 
          title="Densité moyenne" 
          value={stats.averageDensity} 
          icon="🕸️"
          color="#f39c12"
        />
        <StatCard 
          title="Plus grande communauté" 
          value={stats.largestSize} 
          icon="📊"
          color="#9b59b6"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem', minHeight: '600px' }}>
        {/* Liste des communautés */}
        <div style={{
          border: '1px solid #ddd',
          borderRadius: '8px',
          padding: '1rem',
          background: 'white',
          maxHeight: '600px',
          overflowY: 'auto'
        }}>
          <h2 style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            Liste des Communautés
            <span style={{ fontSize: '0.8rem', color: '#666' }}>
              {communities.length} trouvées
            </span>
          </h2>

          {loading ? (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <div className="loading-spinner"></div>
              <p>🔄 Chargement des communautés...</p>
            </div>
          ) : communities.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
              <p>📭 Aucune communauté trouvée</p>
              <button 
                onClick={() => window.location.reload()}
                style={{
                  marginTop: '1rem',
                  padding: '0.5rem 1rem',
                  background: '#3498db',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Réessayer
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {communities.map(community => (
                <CommunityListItem
                  key={community.id}
                  community={community}
                  isSelected={selectedCommunity?.id === community.id}
                  onClick={() => loadCommunityDetails(community)}
                  densityColor={getDensityColor(community.density)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Détails de la communauté sélectionnée */}
        <div style={{
          border: '1px solid #ddd',
          borderRadius: '8px',
          padding: '1.5rem',
          background: 'white'
        }}>
          {!selectedCommunity ? (
            <div style={{ 
              textAlign: 'center', 
              padding: '3rem',
              color: '#666'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🏢</div>
              <h3>Sélectionnez une communauté</h3>
              <p>Choisissez une communauté dans la liste pour voir ses détails</p>
            </div>
          ) : (
            <>
              {/* En-tête de la communauté */}
              <div style={{ marginBottom: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                  <div>
                    <h2 style={{ margin: '0 0 0.5rem 0', color: '#2c3e50' }}>
                      {selectedCommunity.name}
                    </h2>
                    <p style={{ color: '#666', margin: '0' }}>
                      {selectedCommunity.description}
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      onClick={() => setViewMode('list')}
                      style={{
                        padding: '0.5rem 1rem',
                        background: viewMode === 'list' ? '#3498db' : '#ecf0f1',
                        color: viewMode === 'list' ? 'white' : '#2c3e50',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      📋 Liste
                    </button>
                    <button
                      onClick={() => setViewMode('graph')}
                      style={{
                        padding: '0.5rem 1rem',
                        background: viewMode === 'graph' ? '#3498db' : '#ecf0f1',
                        color: viewMode === 'graph' ? 'white' : '#2c3e50',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      🕸️ Graphique
                    </button>
                  </div>
                </div>

                {/* Métriques de la communauté */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                  gap: '1rem',
                  marginBottom: '1rem'
                }}>
                  <MetricCard 
                    label="Taille" 
                    value={selectedCommunity.size} 
                    icon="👥"
                  />
                  <MetricCard 
                    label="Densité" 
                    value={`${(selectedCommunity.density * 100).toFixed(1)}%`} 
                    icon="🕸️"
                    color={getDensityColor(selectedCommunity.density)}
                  />
                  <MetricCard 
                    label="Membres top" 
                    value={selectedCommunity.topMembers?.length || 0} 
                    icon="⭐"
                  />
                </div>

                {/* Domaines de recherche */}
                {selectedCommunity.topics && selectedCommunity.topics.length > 0 && (
                  <div style={{ marginBottom: '1rem' }}>
                    <strong>Domaines: </strong>
                    {selectedCommunity.topics.map((topic, index) => (
                      <span
                        key={index}
                        style={{
                          display: 'inline-block',
                          background: '#ecf0f1',
                          padding: '0.25rem 0.5rem',
                          borderRadius: '12px',
                          fontSize: '0.8rem',
                          margin: '0.25rem',
                          color: '#2c3e50'
                        }}
                      >
                        {topic}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {loadingDetails ? (
                <div style={{ textAlign: 'center', padding: '2rem' }}>
                  <div className="loading-spinner"></div>
                  <p>🔄 Chargement des détails...</p>
                </div>
              ) : viewMode === 'list' ? (
                <CommunityMembersList 
                  communityDetails={communityDetails} 
                  centralityColor={getCentralityColor}
                />
              ) : (
                <CommunityGraphView 
                  community={selectedCommunity}
                  communityDetails={communityDetails}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// Composants auxiliaires (restent identiques)
const StatCard = ({ title, value, icon, color }) => (
  <div style={{
    background: 'white',
    padding: '1.5rem',
    borderRadius: '8px',
    border: '1px solid #ddd',
    textAlign: 'center',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  }}>
    <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>{icon}</div>
    <h3 style={{ margin: '0', color: color }}>{value}</h3>
    <p style={{ margin: '0', color: '#666', fontSize: '0.9rem' }}>{title}</p>
  </div>
)

const CommunityListItem = ({ community, isSelected, onClick, densityColor }) => (
  <div
    onClick={onClick}
    style={{
      padding: '1rem',
      border: `2px solid ${isSelected ? '#3498db' : '#ecf0f1'}`,
      borderRadius: '6px',
      background: isSelected ? '#ebf5fb' : 'white',
      cursor: 'pointer',
      transition: 'all 0.2s',
      '&:hover': {
        background: '#f5f5f5'
      }
    }}
  >
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
      <h4 style={{ margin: '0', color: '#2c3e50' }}>{community.name}</h4>
      <span style={{
        padding: '0.25rem 0.5rem',
        background: densityColor,
        color: 'white',
        borderRadius: '12px',
        fontSize: '0.7rem',
        fontWeight: 'bold'
      }}>
        {(community.density * 100).toFixed(0)}%
      </span>
    </div>
    <p style={{ margin: '0 0 0.5rem 0', fontSize: '0.8rem', color: '#666' }}>
      {community.size} membres
    </p>
    <div style={{ fontSize: '0.7rem', color: '#999', maxHeight: '2em', overflow: 'hidden' }}>
      {community.description}
    </div>
  </div>
)

const MetricCard = ({ label, value, icon, color }) => (
  <div style={{
    background: '#f8f9fa',
    padding: '0.75rem',
    borderRadius: '6px',
    textAlign: 'center'
  }}>
    <div style={{ fontSize: '1.2rem', marginBottom: '0.25rem' }}>{icon}</div>
    <div style={{ 
      fontSize: '1.1rem', 
      fontWeight: 'bold',
      color: color || '#2c3e50',
      marginBottom: '0.25rem'
    }}>
      {value}
    </div>
    <div style={{ fontSize: '0.8rem', color: '#666' }}>{label}</div>
  </div>
)

const CommunityMembersList = ({ communityDetails, centralityColor }) => (
  <div>
    <h3 style={{ marginBottom: '1rem' }}>Membres Principaux</h3>
    {communityDetails?.members?.length > 0 ? (
      <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
        {communityDetails.members.map((member, index) => (
          <div
            key={member.authorId || index}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '0.75rem',
              border: '1px solid #ecf0f1',
              borderRadius: '4px',
              marginBottom: '0.5rem',
              background: 'white'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{
                width: '30px',
                height: '30px',
                background: '#3498db',
                color: 'white',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 'bold',
                fontSize: '0.8rem'
              }}>
                #{index + 1}
              </div>
              <div>
                <div style={{ fontWeight: 'bold', color: '#2c3e50' }}>
                  {member.name}
                </div>
                <div style={{ fontSize: '0.8rem', color: '#666' }}>
                  {member.papers} relations
                </div>
              </div>
            </div>
            <div style={{
              padding: '0.25rem 0.75rem',
              background: centralityColor(member.centrality),
              color: 'white',
              borderRadius: '12px',
              fontSize: '0.7rem',
              fontWeight: 'bold'
            }}>
              Centralité: {(member.centrality * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    ) : (
      <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
        <p>📭 Aucun membre à afficher</p>
      </div>
    )}
  </div>
)

const CommunityGraphView = ({ community, communityDetails }) => (
  <div style={{
    height: '400px',
    background: '#f8f9fa',
    borderRadius: '6px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: '2px dashed #ddd'
  }}>
    <div style={{ textAlign: 'center', color: '#666' }}>
      <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🕸️</div>
      <h3>Visualisation du Graphe</h3>
      <p>Visualisation interactive de la communauté {community.name}</p>
      <p style={{ fontSize: '0.9rem' }}>
        {communityDetails?.collaborationNetwork?.nodes || 0} nœuds • 
        {communityDetails?.collaborationNetwork?.edges || 0} connexions
      </p>
      <p style={{ fontSize: '0.8rem', fontStyle: 'italic' }}>
        À implémenter avec D3.js ou Vis.js
      </p>
    </div>
  </div>
)

export default Communities