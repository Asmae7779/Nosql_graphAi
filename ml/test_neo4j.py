import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  
ml_api_dir = os.path.join(current_dir, "ml-api")  
sys.path.insert(0, ml_api_dir)

from api import Neo4jClient

def test_neo4j_connection():
    """Tester la connexion Neo4j"""
    client = Neo4jClient()
    
    print("🔍 Test de connexion Neo4j...")
    success = client.test_connection()
    
    if success:
        print("\n✅ Connexion Neo4j OK")
        
        # Tester la récupération de noms
        test_ids = ["3308557", "34917892", "2465270", "145845739", "3462562"]
        print(f"\n🔍 Test récupération noms pour {len(test_ids)} auteurs...")
        names = client.get_author_names(test_ids)
        
        for author_id in test_ids:
            print(f"  {author_id}: {names.get(author_id, 'NON TROUVÉ')}")
    else:
        print("\n❌ Connexion Neo4j échouée")
    
    client.close()

if __name__ == "__main__":
    test_neo4j_connection()