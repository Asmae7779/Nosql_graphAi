import json
import pickle
import numpy as np
from pathlib import Path
import networkx as nx

def export_communities_data(analytics, output_dir="../results"):
    """Exporter les données des communautés au format JSON pour le frontend"""
    
    # Créer le répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("📤 Exportation des données des communautés...")
    
    # 1. Données des communautés complètes
    communities_dict = {}
    for node, comm_id in analytics.partition.items():
        if comm_id not in communities_dict:
            communities_dict[comm_id] = []
        communities_dict[comm_id].append(node)
    
    communities_list = []
    for comm_id, nodes in communities_dict.items():
        subgraph = analytics.G_nx.subgraph(nodes)
        
        # Top membres
        top_members = []
        for node in nodes:
            if node in analytics.metrics_cache:
                metrics = analytics.metrics_cache[node]
                top_members.append({
                    "author_id": str(node),
                    "name": f"Author {node}",
                    "publication_count": metrics.get("degree", 0),
                    "centrality_score": metrics.get("degree_centrality", 0)
                })
        
        top_members.sort(key=lambda x: x["centrality_score"], reverse=True)
        
        communities_list.append({
            "id": comm_id,
            "name": f"Research Community {comm_id}",
            "size": len(nodes),
            "density": round(nx.density(subgraph) if subgraph.number_of_edges() > 0 else 0, 3),
            "topics": ["Computer Science", "Physics", "Mathematics"][:comm_id % 3 + 1],  # Exemple
            "top_members": top_members[:10],
            "internal_edges": subgraph.number_of_edges(),
            "description": f"Community of {len(nodes)} researchers in collaborative network"
        })
    
    # Trier par taille
    communities_list.sort(key=lambda x: x["size"], reverse=True)
    
    # Sauvegarder
    communities_path = output_path / "communities_api.json"
    with open(communities_path, 'w', encoding='utf-8') as f:
        json.dump({"communities": communities_list}, f, indent=2)
    print(f"✅ Communautés sauvegardées : {communities_path}")
    
    # 2. Map auteur -> communauté
    author_map = {}
    for node, comm_id in analytics.partition.items():
        author_map[str(node)] = {
            "name": f"Author {node}",
            "community_id": comm_id
        }
    
    author_map_path = output_path / "author_community_map.json"
    with open(author_map_path, 'w', encoding='utf-8') as f:
        json.dump(author_map, f, indent=2)
    print(f"✅ Map auteur->communauté sauvegardée : {author_map_path}")
    
    # 3. Résumé
    summary = {
        "total_communities": len(communities_list),
        "total_authors": sum(c["size"] for c in communities_list),
        "average_size": sum(c["size"] for c in communities_list) / len(communities_list),
        "average_density": sum(c["density"] for c in communities_list) / len(communities_list),
        "largest_community": {
            "id": communities_list[0]["id"] if communities_list else None,
            "name": communities_list[0]["name"] if communities_list else None,
            "size": communities_list[0]["size"] if communities_list else 0
        },
        "description": "Communities detected from research collaboration network"
    }
    
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Résumé sauvegardé : {summary_path}")
    
    # 4. Exporter des recommandations d'exemple
    recommendations_dir = output_path / "recommendations"
    recommendations_dir.mkdir(exist_ok=True)
    
    # Créer quelques recommandations d'exemple
    example_authors = list(analytics.node_to_idx.keys())[:5]
    for author_id in example_authors:
        try:
            recs = analytics.get_recommendations(str(author_id), top_k=5)
            
            recommendations_data = {
                "researcher_id": str(author_id),
                "recommendations": [
                    {
                        "author_id": str(rec["researcher_id"]),
                        "name": f"Author {rec['researcher_id']}",
                        "score": float(rec["collaboration_score"]),
                        "reason": f"ML score: {rec['ml_score']:.3f}",
                        "common_fields": ["Computer Science", "Mathematics"],
                        "common_papers": rec["common_neighbors"],
                        "network_distance": 2
                    }
                    for rec in recs
                ]
            }
            
            rec_path = recommendations_dir / f"{author_id}.json"
            with open(rec_path, 'w', encoding='utf-8') as f:
                json.dump(recommendations_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Erreur pour {author_id}: {e}")
    
    print(f"✅ Recommandations d'exemple sauvegardées dans {recommendations_dir}")
    
    return {
        "communities_path": str(communities_path),
        "author_map_path": str(author_map_path),
        "summary_path": str(summary_path)
    }

if __name__ == "__main__":
    # Tester l'export
    print("🔧 Test d'exportation des données...")
    
    # Vous devrez charger analytics ici si vous voulez tester
    # Pour l'instant, c'est juste un template
    print("⚠️  Exécutez ce script après avoir chargé analytics dans api.py")