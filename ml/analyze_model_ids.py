import torch
import networkx as nx

def analyze_model():
    """Analyser les IDs dans le modèle étendu"""
    model_path = "models/link_prediction_model_extended.pkl"
    
    print("🔍 Analyse approfondie du modèle étendu")
    print("=" * 60)
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 1. Analyser node_to_idx
    node_to_idx = checkpoint['node_to_idx']
    print(f"1. node_to_idx: {len(node_to_idx)} entrées")
    
    # Échantillon d'IDs
    sample_ids = list(node_to_idx.keys())[:20]
    print(f"   Échantillon (20 premiers):")
    for i, node_id in enumerate(sample_ids):
        print(f"     {i+1:2d}. {node_id}")
    
    # 2. Analyser G_full (le graphe)
    G = checkpoint['G_full']
    print(f"\n2. Graphe G_full: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    
    # Vérifier les attributs des nœuds
    print(f"   Attributs du premier nœud:")
    first_node = list(G.nodes())[0]
    print(f"     ID: {first_node}")
    print(f"     Attributs: {G.nodes[first_node]}")
    
    # 3. Vérifier si les nœuds ont des noms
    nodes_with_names = [n for n in G.nodes() if 'name' in G.nodes[n]]
    print(f"\n3. Nœuds avec attribut 'name': {len(nodes_with_names)}")
    if nodes_with_names:
        print(f"   Exemple: {G.nodes[nodes_with_names[0]]['name']}")
    
    # 4. Comparer avec Neo4j
    print(f"\n4. IDs Neo4j vs IDs Modèle:")
    
    # IDs connus de Neo4j
    neo4j_ids = ["3308557", "34917892", "2465270", "145845739", "3462562"]
    
    for neo_id in neo4j_ids:
        try:
            neo_id_int = int(neo_id)
            if neo_id_int in node_to_idx:
                print(f"   ✅ {neo_id} présent dans le modèle")
            else:
                print(f"   ❌ {neo_id} ABSENT du modèle")
        except:
            if neo_id in node_to_idx:
                print(f"   ✅ {neo_id} présent dans le modèle")
            else:
                print(f"   ❌ {neo_id} ABSENT du modèle")
    
    # 5. Vérifier la correspondance ID -> Nom
    print(f"\n5. Correspondance ID -> Nom dans le graphe:")
    for i, node_id in enumerate(list(G.nodes())[:5]):
        attrs = G.nodes[node_id]
        if 'name' in attrs:
            print(f"   {node_id} → {attrs['name']}")
        else:
            print(f"   {node_id} → (pas de nom)")
    
    # 6. Trouver l'ID le plus connecté
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n6. Top 5 nœuds les plus connectés:")
    for node_id, deg in top_nodes:
        name = G.nodes[node_id].get('name', 'inconnu')
        print(f"   {node_id}: {deg} connexions ({name})")

if __name__ == "__main__":
    analyze_model()