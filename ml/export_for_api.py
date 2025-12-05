import pickle
import json
import networkx as nx

def export_communities_for_api():
    
    #Charger les meilleures communautes
    with open('results/communities/louvain_stats.pkl', 'rb') as f:
        data = pickle.load(f)
    
    communities = data['communities']
    partition = data['partition']
    algorithm = data['algorithm']
    modularity = data['modularity']
    
    #Charger le graphe
    with open('notebooks/data/graph_cache.pkl', 'rb') as f:
        G = pickle.load(f)
    
    # Préparer les données pour l'API
    api_data = {
        "metadata": {
            "algorithm": algorithm,
            "modularity": float(modularity),
            "num_communities": len(communities),
            "num_authors": G.number_of_nodes()
        },
        "communities": []
    }
    
    # Pour chaque communauté
    for comm_id, comm_nodes in enumerate(communities):
        # Créer un sous-graphe pour cette communauté
        subgraph = G.subgraph(comm_nodes)
        
        # Trouver les membres les plus connectés
        top_members = sorted(
            subgraph.degree(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Top 10
        
        # Préparer les données de la communauté
        community_data = {
            "id": comm_id,
            "size": len(comm_nodes),
            "members": [
                {
                    "authorId": node,
                    "name": G.nodes[node].get('name', 'Unknown'),
                    "degree": G.degree(node),
                    "degree_in_community": subgraph.degree(node)
                }
                for node in comm_nodes
            ],
            "top_members": [
                {
                    "authorId": node,
                    "name": G.nodes[node].get('name', 'Unknown'),
                    "degree_in_community": degree
                }
                for node, degree in top_members
            ]
        }
        
        api_data["communities"].append(community_data)
    
    # 5. Sauvegarder en JSON
    with open('results/communities/communities_api.json', 'w', encoding='utf-8') as f:
        json.dump(api_data, f, indent=2, ensure_ascii=False)
    
    print("Export terminé: results/communities/communities_api.json")
    print(f"{len(communities)} communautés exportées")
    
    # Créer aussi un mapping auteur -> communauté
    author_community_map = {}
    for author_id, comm_id in partition.items():
        author_community_map[author_id] = {
            "community_id": comm_id,
            "name": G.nodes[author_id].get('name', 'Unknown')
        }
    
    with open('results/communities/author_community_map.json', 'w', encoding='utf-8') as f:
        json.dump(author_community_map, f, indent=2, ensure_ascii=False)
    
    print("Mapping auteur->communauté: results/communities/author_community_map.json")
    
    # Créer un résumé simple
    summary = {
        "algorithm": algorithm,
        "modularity": float(modularity),
        "num_communities": len(communities),
        "community_sizes": [len(c) for c in communities],
        "largest_community": max(len(c) for c in communities),
        "smallest_community": min(len(c) for c in communities)
    }
    
    with open('results/communities/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Résumé: results/communities/summary.json")

if __name__ == "__main__":
    export_communities_for_api()