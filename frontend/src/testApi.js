import axios from 'axios'

async function testAllAPIs() {
  console.log('🧪 Test des APIs...')
  
  try {
    // Test API ML directe
    console.log('1. Test API ML (port 8000)...')
    const mlResponse = await axios.get('http://localhost:8000/api/communities')
    console.log('✅ API ML OK:', mlResponse.data.communities?.length, 'communautés')
    
    // Test backend Neo4j
    console.log('2. Test Backend Neo4j (port 5000)...')
    const backendResponse = await axios.get('http://localhost:5000/api/test')
    console.log('✅ Backend OK:', backendResponse.data)
    
    // Test recommandations
    console.log('3. Test Recommandations...')
    const recResponse = await axios.get('http://localhost:8000/recommendations/3308557?filter_hubs=true&top_k=5')
    console.log('✅ Recommandations OK:', recResponse.data.recommendations?.length, 'recommandations')
    
  } catch (error) {
    console.error('❌ Erreur test:', error.message)
    console.error('URL échouée:', error.config?.url)
  }
}

// Exécuter le test
testAllAPIs()