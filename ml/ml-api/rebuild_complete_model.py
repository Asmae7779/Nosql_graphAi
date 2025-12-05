"""
Script pour recréer node_to_idx avec TOUS les auteurs et collaborations
"""
import pickle
import torch
from neo4j import GraphDatabase
import networkx as nx

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "h20031717"

def rebuild_complete_mapping():
    """Créer un graphe complet avec tous les auteurs de Neo4j"""
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    G = nx.Graph()
    
    print("📥 Chargement de tous les auteurs depuis Neo4j...")

    with driver.session() as session:
        
        # Charger tous les auteurs (IDs = authorId)
        result = session.run("""
            MATCH (a:authors)
            RETURN a.authorId AS id
        """)

        for record in result:
            if record["id"] is not None:
                G.add_node(str(record["id"]))  # toujours string

        print(f"✅ {G.number_of_nodes()} auteurs chargés")
        
        # Charger toutes les collaborations via papers
        print("📥 Chargement des collaborations (AUTHORED)...")

        result = session.run("""
            MATCH (a1:authors)-[:AUTHORED]->(p:papers)<-[:AUTHORED]-(a2:authors)
            WHERE a1.authorId IS NOT NULL AND a2.authorId IS NOT NULL AND a1.authorId <> a2.authorId
            RETURN DISTINCT a1.authorId AS src, a2.authorId AS dst
        """)

        for record in result:
            src = str(record["src"])
            dst = str(record["dst"])
            if src != dst:
                G.add_edge(src, dst)

        print(f"✅ {G.number_of_edges()} collaborations ajoutées")

    driver.close()
    
    # Sauvegarde
    print("💾 Sauvegarde du graphe complet...")
    with open("../data/graph_complete.pkl", "wb") as f:
        pickle.dump(G, f)
    
    print("\n🎉 Graphe complet reconstruit avec succès !")
    print(f"   Nœuds  : {G.number_of_nodes()}")
    print(f"   Arêtes : {G.number_of_edges()}")
    print("\n📝 Mettre dans config.yaml :")
    print("graph:")
    print("  cache_path: '../data/graph_complete.pkl'")

if __name__ == "__main__":
    rebuild_complete_mapping()
